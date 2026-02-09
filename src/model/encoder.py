"""
Encoder architectures using Taxonomic or Regular Convolutional Layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .taxon_layers import TaxonConv


class CIFAR10TaxonEncoder(nn.Module):
    """
    Encoder for CIFAR-10 images (32x32x3) supporting both Taxonomic and Regular Conv layers
    
    Parameters:
    -----------
    latent_dim : int
        Dimensionality of the latent space
    temperature : float
        Temperature for alpha sigmoid in taxonomic layers
    kernel_sizes : list of int or int
        Kernel sizes for each Conv layer
    strides : list of int or int
        Strides for downsampling after each layer
    n_layers : list of int or int or None
        Number of hierarchy layers for TaxonConv (ignored for regular conv)
    n_filters : list of int or int or None
        Number of filters for regular Conv2d layers (ignored for taxonomic)
    layer_types : list of str or None
        Type of each layer: 'taxonomic' or 'regular'. If None, all taxonomic.
    use_maxpool : bool
        If True, use maxpooling for downsampling. If False, use stride in conv layers.
    """
    
    def __init__(self, latent_dim=256, temperature=1.0, kernel_sizes=3, strides=2, 
                 n_layers=None, n_filters=None, layer_types=None, use_maxpool=True):
        super(CIFAR10TaxonEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.use_maxpool = use_maxpool
        
        # Determine number of layers
        if n_layers is None and n_filters is None:
            num_layers = 3
            n_layers = [4, 5, 6]
            layer_types = ['taxonomic'] * 3 if layer_types is None else layer_types
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
            layer_types = ['taxonomic'] * num_layers
        elif isinstance(layer_types, str):
            layer_types = [layer_types] * num_layers
        
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
                strides = strides + [1]
        
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
        self.strides = strides
        self.layer_types = layer_types
        
        # Build conv layers dynamically (taxonomic or regular)
        self.conv_layers = nn.ModuleList()
        in_ch = 3
        for i in range(self.num_layers):
            layer_type = layer_types[i]
            
            if layer_type == 'taxonomic_conv' or layer_type == 'taxonomic':
                # Use TaxonConv
                conv = TaxonConv(
                    in_channels=in_ch, 
                    kernel_size=kernel_sizes[i], 
                    n_layers=n_layers[i] if n_layers else 4, 
                    temperature=temperature
                )
                out_ch = sum(2**j for j in range((n_layers[i] if n_layers else 4) + 1))
            elif layer_type == 'conv':
                # Use regular Conv2d
                out_ch = n_filters[i] if n_filters else 64
                conv = nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_sizes[i],
                    padding=kernel_sizes[i] // 2,
                    bias=True
                )
            else:
                raise ValueError(f"Unknown layer_type in encoder: {layer_type}")
            
            self.conv_layers.append(conv)
            in_ch = out_ch
        
        # Calculate final spatial size and channels (keep for decoder initialization)
        final_size = 32
        for stride in strides[:self.num_layers]:
            final_size = final_size // stride
        
        self.final_channels = in_ch
        self.final_size = final_size
        
    def forward(self, x):
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            x = F.relu(x)
            if self.strides[i] > 1:
                if self.use_maxpool:
                    x = F.max_pool2d(x, self.strides[i])
                else:
                    x = F.avg_pool2d(x, self.strides[i])
        
        # Return spatial features (B, C, H, W)
        return x
