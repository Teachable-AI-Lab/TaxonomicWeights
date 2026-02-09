"""
Autoencoders for taxonomic CIFAR-10 and CelebA-HQ architectures.
"""

import torch
import torch.nn as nn
from .encoder import CIFAR10TaxonEncoder, CelebAHQTaxonEncoder
from .decoder import CIFAR10TaxonDecoder, CelebAHQTaxonDecoder


class CIFAR10TaxonAutoencoder(nn.Module):
    """
    Taxonomic Autoencoder for CIFAR-10
    
    Uses hierarchical convolutional filters where parent filters are constrained
    as convex combinations of children, providing natural regularization.
    
    Architecture:
    - Encoder: 3 TaxonConv layers → latent (256-d)
    - Decoder: 3 TaxonDeconv layers → reconstruction (3x32x32)
    
    Parameters:
    -----------
    latent_dim : int
        Dimensionality of the latent space
    temperature : float
        Temperature for alpha sigmoid in taxonomic layers
    encoder_kernel_sizes : list of int or int
        Kernel sizes for encoder TaxonConv layers
    decoder_kernel_sizes : list of int or int
        Kernel sizes for decoder TaxonDeconv layers
    encoder_strides : list of int or int
        Strides for encoder downsampling (default: 2)
    decoder_strides : list of int or int
        Strides for decoder upsampling (default: [2, 2, 1])
    decoder_paddings : list of int or int
        Padding for decoder layers (default: auto-calculated from kernel sizes)
    use_maxpool : bool
        If True, use maxpooling in encoder. If False, use average pooling.
    """
    
    def __init__(self, latent_dim=256, temperature=1.0, encoder_kernel_sizes=3,
                 decoder_kernel_sizes=None, encoder_strides=2, decoder_strides=None,
                 encoder_n_layers=None, decoder_n_layers=None,
                 encoder_n_filters=None, decoder_n_filters=None,
                 encoder_layer_types=None, decoder_layer_types=None,
                 decoder_paddings=None, decoder_output_paddings=None, 
                 use_maxpool=True, random_init_alphas=False,
                 alpha_init_distribution="uniform", alpha_init_range=None,
                 alpha_init_seed=None):
        super(CIFAR10TaxonAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Calculate initial spatial size for decoder based on encoder settings
        initial_size = 32
        if isinstance(encoder_strides, int):
            initial_size = initial_size // encoder_strides // encoder_strides
        elif isinstance(encoder_strides, list):
            for stride in encoder_strides:
                initial_size = initial_size // stride
        else:
            initial_size = initial_size // encoder_strides[0] // encoder_strides[1]
        
        self.encoder = CIFAR10TaxonEncoder(
            latent_dim=latent_dim, 
            temperature=temperature,
            kernel_sizes=encoder_kernel_sizes,
            strides=encoder_strides,
            n_layers=encoder_n_layers,
            n_filters=encoder_n_filters,
            layer_types=encoder_layer_types,
            use_maxpool=use_maxpool,
            random_init_alphas=random_init_alphas,
            alpha_init_distribution=alpha_init_distribution,
            alpha_init_range=alpha_init_range,
            alpha_init_seed=alpha_init_seed
        )
        
        # Get encoder's final channels for decoder input
        encoder_final_channels = self.encoder.final_channels
        
        self.decoder = CIFAR10TaxonDecoder(
            latent_dim=latent_dim, 
            temperature=temperature,
            kernel_sizes=decoder_kernel_sizes,
            strides=decoder_strides,
            paddings=decoder_paddings,
            output_paddings=decoder_output_paddings,
            n_layers=decoder_n_layers,
            n_filters=decoder_n_filters,
            layer_types=decoder_layer_types,
            initial_spatial_size=initial_size,
            encoder_final_channels=encoder_final_channels,
            random_init_alphas=random_init_alphas,
            alpha_init_distribution=alpha_init_distribution,
            alpha_init_range=alpha_init_range,
            alpha_init_seed=alpha_init_seed
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon
    
    def get_hierarchical_features(self, x):
        """
        Extract hierarchical features from encoder layers.
        
        Returns:
        --------
        dict: Dictionary containing intermediate activations from each TaxonConv layer
        """
        features = {}
        
        # Layer 1
        h1 = self.encoder.taxon_conv1(x)
        features['conv1'] = h1
        h1 = torch.relu(h1)
        h1 = torch.nn.functional.max_pool2d(h1, 2)
        
        # Layer 2
        h2 = self.encoder.taxon_conv2(h1)
        features['conv2'] = h2
        h2 = torch.relu(h2)
        h2 = torch.nn.functional.max_pool2d(h2, 2)
        
        # Layer 3
        h3 = self.encoder.taxon_conv3(h2)
        features['conv3'] = h3
        
        return features


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv2d + ReLU layers with padding=1 to preserve spatial size."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


def _upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """ConvTranspose2d stride=2 followed by Conv2d to match the spec."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
class CelebAHQTaxonAutoencoder(nn.Module):
    """
    Taxonomic Autoencoder for CelebA-HQ (256x256x3).
    
    Similar to CIFAR10TaxonAutoencoder but for larger images with more layers.
    Uses hierarchical convolutional filters where parent filters are constrained
    as convex combinations of children.
    
    Architecture:
    - Encoder: 10 layers (5 blocks × 2 layers) → latent (16x16x511)
    - Decoder: 8 layers (4 blocks × 2 layers) → reconstruction (256x256x3)
    
    Parameters:
    -----------
    latent_dim : int
        Dimensionality of the latent space (not used for spatial latent)
    temperature : float
        Temperature for alpha sigmoid in taxonomic layers
    encoder_kernel_sizes : list of int or int
        Kernel sizes for encoder layers
    decoder_kernel_sizes : list of int or int
        Kernel sizes for decoder layers
    encoder_strides : list of int or int
        Strides for encoder downsampling
    decoder_strides : list of int or int
        Strides for decoder upsampling
    decoder_paddings : list of int or int
        Padding for decoder layers
    decoder_output_paddings : list of int or int
        Output padding for decoder transpose convolutions
    use_maxpool : bool
        If True, use maxpooling in encoder. If False, use average pooling.
    encoder_n_layers : list of int
        Number of taxonomic layers for each encoder layer
    decoder_n_layers : list of int
        Number of taxonomic layers for each decoder layer
    encoder_layer_types : list of str
        Layer types for encoder ('taxonomic_conv' or 'conv')
    decoder_layer_types : list of str
        Layer types for decoder ('taxonomic_deconv', 'taxonomic_conv', 'deconv', 'conv')
    output_activation : str
        Output activation ('sigmoid' for [0,1] or 'tanh' for [-1,1])
    """
    
    def __init__(self, latent_dim=256, temperature=1.0, encoder_kernel_sizes=3,
                 decoder_kernel_sizes=None, encoder_strides=None, decoder_strides=None,
                 encoder_n_layers=None, decoder_n_layers=None,
                 encoder_n_filters=None, decoder_n_filters=None,
                 encoder_layer_types=None, decoder_layer_types=None,
                 decoder_paddings=None, decoder_output_paddings=None, 
                 use_maxpool=True, random_init_alphas=False,
                 alpha_init_distribution="uniform", alpha_init_range=None,
                 alpha_init_seed=None, output_activation='sigmoid'):
        super(CelebAHQTaxonAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Default encoder strides (pools after layers: 0,1,2,3)
        if encoder_strides is None:
            encoder_strides = [1, 2, 1, 2, 1, 2, 1, 2, 1, 1]
        
        # Calculate initial spatial size for decoder based on encoder settings
        initial_size = 256
        if isinstance(encoder_strides, int):
            # Simplified: assume 4 pooling operations
            initial_size = 256 // (2 ** 4)
        elif isinstance(encoder_strides, list):
            for stride in encoder_strides:
                if stride > 1:
                    initial_size = initial_size // stride
        else:
            initial_size = 16
        
        self.encoder = CelebAHQTaxonEncoder(
            latent_dim=latent_dim, 
            temperature=temperature,
            kernel_sizes=encoder_kernel_sizes,
            strides=encoder_strides,
            n_layers=encoder_n_layers,
            n_filters=encoder_n_filters,
            layer_types=encoder_layer_types,
            use_maxpool=use_maxpool,
            random_init_alphas=random_init_alphas,
            alpha_init_distribution=alpha_init_distribution,
            alpha_init_range=alpha_init_range,
            alpha_init_seed=alpha_init_seed
        )
        
        # Get encoder's final channels for decoder input
        encoder_final_channels = self.encoder.final_channels
        
        self.decoder = CelebAHQTaxonDecoder(
            latent_dim=latent_dim, 
            temperature=temperature,
            kernel_sizes=decoder_kernel_sizes,
            strides=decoder_strides,
            paddings=decoder_paddings,
            output_paddings=decoder_output_paddings,
            n_layers=decoder_n_layers,
            n_filters=decoder_n_filters,
            layer_types=decoder_layer_types,
            initial_spatial_size=initial_size,
            encoder_final_channels=encoder_final_channels,
            random_init_alphas=random_init_alphas,
            alpha_init_distribution=alpha_init_distribution,
            alpha_init_range=alpha_init_range,
            alpha_init_seed=alpha_init_seed,
            output_activation=output_activation
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

