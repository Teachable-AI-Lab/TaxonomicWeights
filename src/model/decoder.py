"""
Decoder architectures using Taxonomic or Regular Deconvolutional Layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .taxon_layers import TaxonDeconv, TaxonConv, TaxonDeconvKL, TaxonConvKL


class CIFAR10TaxonDecoder(nn.Module):
    """
    Decoder for CIFAR-10 images (32x32x3) supporting both Taxonomic and Regular layers
    
    Parameters:
    -----------
    latent_dim : int
        Dimensionality of the latent space
    temperature : float
        Temperature for alpha sigmoid in taxonomic layers
    kernel_sizes : list of int or int
        Kernel sizes for each layer
    strides : list of int or int
        Strides for upsampling
    n_layers : list of int or int or None
        Number of hierarchy layers for TaxonDeconv (ignored for regular)
    n_filters : list of int or int or None
        Number of filters for regular ConvTranspose2d (ignored for taxonomic)
    layer_types : list of str or None
        Type of each layer: 'taxonomic' or 'regular'. If None, all taxonomic.
    initial_spatial_size : int
        Spatial size of input from encoder
    encoder_final_channels : int
        Number of channels from encoder output
    """
    
    def __init__(self, latent_dim=256, temperature=1.0, kernel_sizes=None, 
                 strides=None, paddings=None, output_paddings=None, n_layers=None, 
                 n_filters=None, layer_types=None, initial_spatial_size=4,
                 encoder_final_channels=None, use_maxpool=True, random_init_alphas=False,
                 alpha_init_distribution="uniform", alpha_init_range=None,
                 alpha_init_seed=None):
        super(CIFAR10TaxonDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.initial_spatial_size = initial_spatial_size
        self.use_maxpool = use_maxpool
        
        # Determine number of layers
        if n_layers is None and n_filters is None:
            num_layers = 3
            n_layers = [6, 5, 4]
            layer_types = ['taxonomic_deconv'] * 3 if layer_types is None else layer_types
        elif isinstance(n_layers, int):
            num_layers = 3
            n_layers = [n_layers] * 3
        elif isinstance(n_layers, list):
            num_layers = len(n_layers)
        elif isinstance(n_filters, list):
            num_layers = len(n_filters)
        else:
            num_layers = 3
        
        # Handle layer_types
        if layer_types is None:
            layer_types = ['taxonomic_deconv'] * num_layers
        elif isinstance(layer_types, str):
            layer_types = [layer_types] * num_layers
        
        # Default values based on num_layers
        if kernel_sizes is None:
            kernel_sizes = [3] * num_layers
        if strides is None:
            strides = [2] * num_layers
        if output_paddings is None:
            output_paddings = [1] * num_layers
        
        # Handle scalar or list inputs
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers
        elif isinstance(kernel_sizes, list) and len(kernel_sizes) < num_layers:
            while len(kernel_sizes) < num_layers:
                kernel_sizes = kernel_sizes + [kernel_sizes[-1]]
        
        if isinstance(strides, int):
            strides = [strides] * num_layers
        elif isinstance(strides, list) and len(strides) < num_layers:
            while len(strides) < num_layers:
                strides = strides + [strides[-1]]
        
        # Calculate default paddings
        if paddings is None:
            paddings = [k // 2 for k in kernel_sizes]
        
        if isinstance(paddings, int):
            paddings = [paddings] * num_layers
        elif isinstance(paddings, list) and len(paddings) < num_layers:
            while len(paddings) < num_layers:
                paddings = paddings + [paddings[-1]]
        
        if isinstance(output_paddings, int):
            output_paddings = [output_paddings] * num_layers
        elif isinstance(output_paddings, list) and len(output_paddings) < num_layers:
            while len(output_paddings) < num_layers:
                output_paddings = output_paddings + [output_paddings[-1]]
        
        # Handle n_layers for taxonomic
        if n_layers is not None:
            if isinstance(n_layers, int):
                n_layers = [n_layers] * num_layers
            elif isinstance(n_layers, list) and len(n_layers) < num_layers:
                while len(n_layers) < num_layers:
                    n_layers = n_layers + [n_layers[-1]]
        
        # Handle n_filters for regular
        if n_filters is not None:
            if isinstance(n_filters, int):
                n_filters = [n_filters] * num_layers
            elif isinstance(n_filters, list) and len(n_filters) < num_layers:
                while len(n_filters) < num_layers:
                    n_filters = n_filters + [n_filters[-1]]
        
        self.num_layers = num_layers
        self.n_layers = n_layers
        self.layer_types = layer_types
        self.strides = strides
        
        # Spatial input from encoder - no linear layer needed
        initial_channels = encoder_final_channels if encoder_final_channels else (
            sum(2**i for i in range((n_layers[0] if n_layers else 6) + 1))
            if n_layers and layer_types[0] == 'taxonomic' else (
                n_filters[0] if n_filters else 128
            )
        )
        
        self.initial_channels = initial_channels
        self.initial_spatial_size = initial_spatial_size
        
        # Build decoder layers dynamically based on layer_type
        self.deconv_layers = nn.ModuleList()
        in_ch = initial_channels
        for i in range(self.num_layers):
            layer_type = layer_types[i]
            layer_seed = None if alpha_init_seed is None else int(alpha_init_seed) + i
            
            if layer_type == 'taxonomic_deconv' or layer_type == 'taxonomic':
                # Always use TaxonDeconv for decoder upsampling
                layer = TaxonDeconv(
                    in_channels=in_ch, 
                    out_channels=1, 
                    kernel_size=kernel_sizes[i], 
                    n_layers=n_layers[i] if n_layers else 4, 
                    stride=strides[i], 
                    padding=paddings[i],
                    output_padding=output_paddings[i], 
                    temperature=temperature,
                    random_init_alphas=random_init_alphas,
                    alpha_init_distribution=alpha_init_distribution,
                    alpha_init_range=alpha_init_range,
                    alpha_init_seed=layer_seed
                )
                out_ch = sum(2**j for j in range((n_layers[i] if n_layers else 4) + 1))
            elif layer_type == 'taxonomic_deconv_kl':
                # Always use TaxonDeconvKL for decoder upsampling
                layer = TaxonDeconvKL(
                    in_channels=in_ch, 
                    out_channels=1, 
                    kernel_size=kernel_sizes[i], 
                    n_layers=n_layers[i] if n_layers else 4, 
                    stride=strides[i], 
                    padding=paddings[i],
                    output_padding=output_paddings[i], 
                    temperature=temperature,
                    random_init_alphas=random_init_alphas,
                    alpha_init_distribution=alpha_init_distribution,
                    alpha_init_range=alpha_init_range,
                    alpha_init_seed=layer_seed
                )
                out_ch = sum(2**j for j in range(1, (n_layers[i] if n_layers else 4) + 1))
            elif layer_type == 'taxonomic_conv':
                # Use TaxonConv for regular convolution
                layer = TaxonConv(
                    in_channels=in_ch,
                    kernel_size=kernel_sizes[i],
                    n_layers=n_layers[i] if n_layers else 4,
                    temperature=temperature,
                    random_init_alphas=random_init_alphas,
                    alpha_init_distribution=alpha_init_distribution,
                    alpha_init_range=alpha_init_range,
                    alpha_init_seed=layer_seed
                )
                out_ch = sum(2**j for j in range((n_layers[i] if n_layers else 4) + 1))
            elif layer_type == 'taxonomic_conv_kl':
                # Use TaxonConvKL (returns output, dkl)
                layer = TaxonConvKL(
                    in_channels=in_ch,
                    kernel_size=kernel_sizes[i],
                    n_layers=n_layers[i] if n_layers else 4,
                    temperature=temperature,
                    random_init_alphas=random_init_alphas,
                    alpha_init_distribution=alpha_init_distribution,
                    alpha_init_range=alpha_init_range,
                    alpha_init_seed=layer_seed
                )
                out_ch = sum(2**j for j in range(1, (n_layers[i] if n_layers else 4) + 1))
            elif layer_type == 'deconv':
                out_ch = n_filters[i] if n_filters else 64
                # Always use ConvTranspose2d for decoder upsampling
                layer = nn.ConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=output_paddings[i],
                    bias=True
                )
            elif layer_type == 'conv':
                # Use regular Conv2d
                out_ch = n_filters[i] if n_filters else 64
                layer = nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_sizes[i],
                    padding=paddings[i],
                    bias=True
                )
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}")
            
            self.deconv_layers.append(layer)
            in_ch = out_ch
        
        # Final Conv2D layer to project to RGB (3 channels)
        self.final_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=3,
            kernel_size=3,
            padding=1
        )
        
    def forward(self, z):
        # z is spatial input from encoder (B, C, H, W)
        x = z
        
        for i, deconv in enumerate(self.deconv_layers):
            
            # KL layers may return either a tensor or (tensor, dkl). Accept both.
            if isinstance(deconv, (TaxonDeconvKL, TaxonConvKL)):
                res = deconv(x)
                if isinstance(res, tuple) and len(res) == 2:
                    x, dkl = res
                else:
                    x = res
                    dkl = getattr(deconv, '_last_dkl', None)
                # KL layers output log-probabilities (always negative); skip ReLU
            else:
                x = deconv(x)
                x = F.relu(x)
        
        # Final conv to RGB (outputs exactly 3 channels)
        x = self.final_conv(x)
        
        # Apply tanh activation for output in [-1, 1] range
        x = torch.tanh(x)
        
        return x


class CelebAHQTaxonDecoder(nn.Module):
    """Decoder for CelebA-HQ images (256x256x3) supporting Taxonomic and Regular layers."""
    
    def __init__(self, latent_dim=256, temperature=1.0, kernel_sizes=None, 
                 strides=None, paddings=None, output_paddings=None, n_layers=None, 
                 n_filters=None, layer_types=None, initial_spatial_size=16,
                 encoder_final_channels=None, use_maxpool=True, random_init_alphas=False,
                 alpha_init_distribution="uniform", alpha_init_range=None,
                 alpha_init_seed=None, output_activation='sigmoid'):
        super(CelebAHQTaxonDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.initial_spatial_size = initial_spatial_size
        self.output_activation = output_activation
        self.use_maxpool = use_maxpool
        
        # Determine number of layers
        if n_layers is None and n_filters is None:
            num_layers = 8
            n_layers = [9, 8, 8, 7, 7, 6, 6, 5]
            layer_types = ['taxonomic_deconv', 'taxonomic_conv', 'taxonomic_deconv', 
                          'taxonomic_conv', 'taxonomic_deconv', 'taxonomic_conv',
                          'taxonomic_deconv', 'taxonomic_conv'] if layer_types is None else layer_types
        elif isinstance(n_layers, int):
            num_layers = 8
            n_layers = [n_layers] * 8
        elif isinstance(n_layers, list):
            num_layers = len(n_layers)
        elif isinstance(n_filters, list):
            num_layers = len(n_filters)
        else:
            num_layers = 8
        
        # Handle layer_types
        if layer_types is None:
            layer_types = ['taxonomic'] * num_layers
        elif isinstance(layer_types, str):
            layer_types = [layer_types] * num_layers
        
        # Default values based on num_layers
        if kernel_sizes is None:
            kernel_sizes = [3] * num_layers
        if strides is None:
            strides = [2, 1, 2, 1, 2, 1, 2, 1]
        if output_paddings is None:
            output_paddings = [1, 0, 1, 0, 1, 0, 1, 0]
        
        # Handle scalar or list inputs
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_layers
        elif isinstance(kernel_sizes, list) and len(kernel_sizes) < num_layers:
            while len(kernel_sizes) < num_layers:
                kernel_sizes = kernel_sizes + [kernel_sizes[-1]]
        
        if isinstance(strides, int):
            strides = [strides] * num_layers
        elif isinstance(strides, list) and len(strides) < num_layers:
            while len(strides) < num_layers:
                strides = strides + [strides[-1]]
        
        # Calculate default paddings
        if paddings is None:
            paddings = [k // 2 for k in kernel_sizes]
        
        if isinstance(paddings, int):
            paddings = [paddings] * num_layers
        elif isinstance(paddings, list) and len(paddings) < num_layers:
            while len(paddings) < num_layers:
                paddings = paddings + [paddings[-1]]
        
        if isinstance(output_paddings, int):
            output_paddings = [output_paddings] * num_layers
        elif isinstance(output_paddings, list) and len(output_paddings) < num_layers:
            while len(output_paddings) < num_layers:
                output_paddings = output_paddings + [output_paddings[-1]]
        
        # Handle n_layers for taxonomic
        if n_layers is not None:
            if isinstance(n_layers, int):
                n_layers = [n_layers] * num_layers
            elif isinstance(n_layers, list) and len(n_layers) < num_layers:
                while len(n_layers) < num_layers:
                    n_layers = n_layers + [n_layers[-1]]
        
        # Handle n_filters for regular
        if n_filters is not None:
            if isinstance(n_filters, int):
                n_filters = [n_filters] * num_layers
            elif isinstance(n_filters, list) and len(n_filters) < num_layers:
                while len(n_filters) < num_layers:
                    n_filters = n_filters + [n_filters[-1]]
        
        self.num_layers = num_layers
        self.n_layers = n_layers
        self.layer_types = layer_types
        self.strides = strides
        
        # Spatial input from encoder - no linear layer needed
        initial_channels = encoder_final_channels if encoder_final_channels else (
            sum(2**i for i in range((n_layers[0] if n_layers else 9) + 1))
            if n_layers and layer_types[0] == 'taxonomic' else (
                n_filters[0] if n_filters else 512
            )
        )
        
        self.initial_channels = initial_channels
        self.initial_spatial_size = initial_spatial_size
        
        # Build decoder layers dynamically based on layer_type
        self.deconv_layers = nn.ModuleList()
        in_ch = initial_channels
        for i in range(self.num_layers):
            layer_type = layer_types[i]
            layer_seed = None if alpha_init_seed is None else int(alpha_init_seed) + i
            
            if layer_type == 'taxonomic_deconv' or layer_type == 'taxonomic':
                # Always use TaxonDeconv for decoder upsampling
                layer = TaxonDeconv(
                    in_channels=in_ch, 
                    out_channels=1, 
                    kernel_size=kernel_sizes[i], 
                    n_layers=n_layers[i] if n_layers else 4, 
                    stride=strides[i], 
                    padding=paddings[i],
                    output_padding=output_paddings[i], 
                    temperature=temperature,
                    random_init_alphas=random_init_alphas,
                    alpha_init_distribution=alpha_init_distribution,
                    alpha_init_range=alpha_init_range,
                    alpha_init_seed=layer_seed
                )
                out_ch = sum(2**j for j in range((n_layers[i] if n_layers else 4) + 1))
            elif layer_type == 'taxonomic_deconv_kl':
                # Always use TaxonDeconvKL for decoder upsampling
                layer = TaxonDeconvKL(
                    in_channels=in_ch, 
                    out_channels=1, 
                    kernel_size=kernel_sizes[i], 
                    n_layers=n_layers[i] if n_layers else 4, 
                    stride=strides[i], 
                    padding=paddings[i],
                    output_padding=output_paddings[i], 
                    temperature=temperature,
                    random_init_alphas=random_init_alphas,
                    alpha_init_distribution=alpha_init_distribution,
                    alpha_init_range=alpha_init_range,
                    alpha_init_seed=layer_seed
                )
                out_ch = sum(2**j for j in range(1, (n_layers[i] if n_layers else 4) + 1))
            elif layer_type == 'taxonomic_conv':
                # Use TaxonConv for regular convolution
                layer = TaxonConv(
                    in_channels=in_ch,
                    kernel_size=kernel_sizes[i],
                    n_layers=n_layers[i] if n_layers else 4,
                    temperature=temperature,
                    random_init_alphas=random_init_alphas,
                    alpha_init_distribution=alpha_init_distribution,
                    alpha_init_range=alpha_init_range,
                    alpha_init_seed=layer_seed
                )
                out_ch = sum(2**j for j in range((n_layers[i] if n_layers else 4) + 1))
            elif layer_type == 'taxonomic_conv_kl':
                # Use TaxonConvKL (returns output, dkl)
                layer = TaxonConvKL(
                    in_channels=in_ch,
                    kernel_size=kernel_sizes[i],
                    n_layers=n_layers[i] if n_layers else 4,
                    temperature=temperature,
                    random_init_alphas=random_init_alphas,
                    alpha_init_distribution=alpha_init_distribution,
                    alpha_init_range=alpha_init_range,
                    alpha_init_seed=layer_seed
                )
                out_ch = sum(2**j for j in range(1, (n_layers[i] if n_layers else 4) + 1))
            elif layer_type == 'deconv':
                out_ch = n_filters[i] if n_filters else 64
                # Always use ConvTranspose2d for decoder upsampling
                layer = nn.ConvTranspose2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    output_padding=output_paddings[i],
                    bias=True
                )
            elif layer_type == 'conv':
                # Use regular Conv2d
                out_ch = n_filters[i] if n_filters else 64
                layer = nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_sizes[i],
                    padding=paddings[i],
                    bias=True
                )
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}")
            
            self.deconv_layers.append(layer)
            in_ch = out_ch
        
        # Final Conv2D layer to project to RGB (3 channels)
        self.final_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=3,
            kernel_size=3,
            padding=1
        )
        
    def forward(self, z):
        # z is spatial input from encoder (B, C, H, W)
        x = z
        
        for i, deconv in enumerate(self.deconv_layers):
            
            # KL layers may return either a tensor or (tensor, dkl). Accept both.
            if isinstance(deconv, (TaxonDeconvKL, TaxonConvKL)):
                res = deconv(x)
                if isinstance(res, tuple) and len(res) == 2:
                    x, dkl = res
                else:
                    x = res
                    dkl = getattr(deconv, '_last_dkl', None)
                # KL layers output log-probabilities (always negative); skip ReLU
            else:
                x = deconv(x)
                x = F.relu(x)
        
        # Final conv to RGB (outputs exactly 3 channels)
        x = self.final_conv(x)
        
        # Apply activation based on config
        if self.output_activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.output_activation == 'tanh':
            x = torch.tanh(x)
        
        return x
