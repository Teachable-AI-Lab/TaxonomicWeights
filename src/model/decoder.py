"""
Decoder architectures using Taxonomic Deconvolutional Layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .taxon_layers import TaxonDeconv


class CIFAR10TaxonDecoder(nn.Module):
    """
    Taxonomic Decoder for CIFAR-10 images (32x32x3)
    
    Symmetric architecture mirroring the encoder:
    - Latent vector → Unflatten to (255, H, W)
    - TaxonDeconv layer 1: 255→127 channels (n_layers=6), spatial upsampling
    - TaxonDeconv layer 2: 127→31 channels (n_layers=4), spatial upsampling
    - TaxonDeconv layer 3: 31→3 channels (n_layers=1), spatial: 32x32
    
    Taxonomic Constraint: Each deconvolutional layer uses hierarchical filters
    where parent filters are convex combinations of child filters, maintaining
    multi-scale feature generation during upsampling.
    
    Parameters:
    -----------
    latent_dim : int
        Dimensionality of the latent space
    temperature : float
        Temperature for alpha sigmoid in taxonomic layers
    kernel_sizes : list of int or int
        Kernel sizes for each TaxonDeconv layer (default: [4, 4, 3])
    strides : list of int or int
        Strides for each TaxonDeconv layer (default: [2, 2, 1])
    paddings : list of int or int
        Padding for each TaxonDeconv layer (default: calculated based on kernel_size)
    n_layers : list of int or int
        Number of hierarchy layers for each TaxonDeconv (default: [6, 4, 1])
    initial_spatial_size : int
        Spatial size after unflattening latent vector (default: 8 for 8x8)
    """
    
    def __init__(self, latent_dim=256, temperature=1.0, kernel_sizes=None, 
                 strides=None, paddings=None, output_paddings=None, n_layers=None, initial_spatial_size=8):
        super(CIFAR10TaxonDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.initial_spatial_size = initial_spatial_size
        
        # Default kernel sizes and strides
        if kernel_sizes is None:
            kernel_sizes = [4, 4, 3]
        if strides is None:
            strides = [2, 2, 1]
        if n_layers is None:
            n_layers = [6, 4, 1]
        if output_paddings is None:
            output_paddings = [1, 1, 0]
        
        # Handle scalar or list inputs
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * 3
        if isinstance(strides, int):
            strides = [strides] * 3
        if isinstance(n_layers, int):
            n_layers = [n_layers] * 3
        
        # Calculate default paddings based on kernel sizes
        # For kernel_size k: padding = (k // 2) if k is even, else (k // 2)
        # This ensures proper upsampling: k=4→pad=1, k=6→pad=2, k=3→pad=1, k=5→pad=2
        if paddings is None:
            paddings = []
            for k in kernel_sizes:
                if k == 4:
                    paddings.append(1)
                elif k == 6:
                    paddings.append(2)
                elif k == 3:
                    paddings.append(1)
                elif k == 5:
                    paddings.append(2)
                else:
                    paddings.append(k // 2)
        
        if isinstance(paddings, int):
            paddings = [paddings] * 3
        if isinstance(output_paddings, int):
            output_paddings = [output_paddings] * 3
        
        # Calculate channel sizes to match encoder output
        # Decoder mirrors encoder in reverse:
        # If encoder has n_layers=[3,4,5], it produces [7, 15, 31] channels
        # Decoder should start with 31 and go back to 3 (RGB)
        # We need to know the encoder's n_layers to calculate this properly
        # The decoder's n_layers typically mirrors encoder's n_layers in reverse order
        
        # Calculate expected input channels (from encoder's final output)
        # This should be sum(2^i for i in range(encoder_final_n_layers+1))
        # For now, we reverse the decoder n_layers to estimate the encoder pattern
        # The last element of reversed decoder_n_layers corresponds to encoder's final layer
        encoder_n_layers_estimate = list(reversed(n_layers))
        initial_channels = sum(2**i for i in range(encoder_n_layers_estimate[-1]+1))
        
        # Fully connected from latent space
        self.fc = nn.Linear(latent_dim, initial_channels * initial_spatial_size * initial_spatial_size)
        
        self.initial_channels = initial_channels  # Store for use in forward
        
        # Calculate output channels for each decoder layer
        # Each layer outputs sum(2^i for i in range(n_layers[j]+1))
        out_channels_1 = sum(2**i for i in range(n_layers[0]+1))
        out_channels_2 = sum(2**i for i in range(n_layers[1]+1))
        out_channels_3 = sum(2**i for i in range(n_layers[2]+1))
        
        # TaxonDeconv layer 1
        self.taxon_deconv1 = TaxonDeconv(
            in_channels=initial_channels, out_channels=1, kernel_size=kernel_sizes[0], 
            n_layers=n_layers[0], stride=strides[0], padding=paddings[0], 
            output_padding=output_paddings[0], temperature=temperature
        )
        
        # TaxonDeconv layer 2
        self.taxon_deconv2 = TaxonDeconv(
            in_channels=out_channels_1, out_channels=1, kernel_size=kernel_sizes[1],
            n_layers=n_layers[1], stride=strides[1], padding=paddings[1], 
            output_padding=output_paddings[1], temperature=temperature
        )
        
        # TaxonDeconv layer 3 - outputs RGB channels
        self.taxon_deconv3 = TaxonDeconv(
            in_channels=out_channels_2, out_channels=1, kernel_size=kernel_sizes[2],
            n_layers=n_layers[2], stride=strides[2], padding=paddings[2], 
            output_padding=output_paddings[2], temperature=temperature
        )
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.initial_channels, self.initial_spatial_size, self.initial_spatial_size)
        
        x = self.taxon_deconv1(x)
        x = F.relu(x)
        
        x = self.taxon_deconv2(x)
        x = F.relu(x)
        
        x = self.taxon_deconv3(x)
        x = torch.tanh(x)
        
        return x
