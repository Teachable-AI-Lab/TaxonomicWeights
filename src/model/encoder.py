"""
Encoder architectures using Taxonomic Convolutional Layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .taxon_layers import TaxonConv


class CIFAR10TaxonEncoder(nn.Module):
    """
    Taxonomic Encoder for CIFAR-10 images (32x32x3)
    
    Architecture:
    - TaxonConv layer 1: 3→31 channels (n_layers=4), spatial: 32x32
    - MaxPool/Stride: 32x32 → 16x16
    - TaxonConv layer 2: 31→127 channels (n_layers=6), spatial: 16x16
    - MaxPool/Stride: 16x16 → 8x8
    - TaxonConv layer 3: 127→255 channels (n_layers=7), spatial: 8x8
    - Flatten → Latent vector
    
    Taxonomic Constraint: Each layer uses hierarchical filters where parent
    filters are convex combinations of child filters, learning multi-scale
    features at each level.
    
    Parameters:
    -----------
    latent_dim : int
        Dimensionality of the latent space
    temperature : float
        Temperature for alpha sigmoid in taxonomic layers
    kernel_sizes : list of int or int
        Kernel sizes for each TaxonConv layer (default: 3)
    strides : list of int or int
        Strides for downsampling after each layer (default: [2, 2, 1])
        If only 2 values provided, third defaults to 1 (no pooling)
    n_layers : list of int or int
        Number of hierarchy layers for each TaxonConv (default: [4, 6, 7])
    use_maxpool : bool
        If True, use maxpooling for downsampling. If False, use stride in conv layers.
    """
    
    def __init__(self, latent_dim=256, temperature=1.0, kernel_sizes=3, strides=2, 
                 n_layers=None, use_maxpool=True):
        super(CIFAR10TaxonEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.use_maxpool = use_maxpool
        
        # Handle scalar or list inputs
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * 3
        if isinstance(strides, int):
            strides = [strides, strides, 1]  # Default: pool twice, then no pool
        elif len(strides) == 2:
            strides = strides + [1]  # Add default no-pool for third layer
        if n_layers is None:
            n_layers = [4, 6, 7]
        elif isinstance(n_layers, int):
            n_layers = [n_layers] * 3
        
        # TaxonConv layer 1: outputs sum(2^i for i in range(n_layers[0]+1)) channels
        self.taxon_conv1 = TaxonConv(
            in_channels=3, kernel_size=kernel_sizes[0], n_layers=n_layers[0], temperature=temperature
        )
        out_channels_1 = sum(2**i for i in range(n_layers[0]+1))
        
        # TaxonConv layer 2
        self.taxon_conv2 = TaxonConv(
            in_channels=out_channels_1, kernel_size=kernel_sizes[1], n_layers=n_layers[1], temperature=temperature
        )
        out_channels_2 = sum(2**i for i in range(n_layers[1]+1))
        
        # TaxonConv layer 3
        self.taxon_conv3 = TaxonConv(
            in_channels=out_channels_2, kernel_size=kernel_sizes[2], n_layers=n_layers[2], temperature=temperature
        )
        out_channels_3 = sum(2**i for i in range(n_layers[2]+1))
        
        self.pool_stride1 = strides[0]
        self.pool_stride2 = strides[1]
        self.pool_stride3 = strides[2]
        
        # Calculate final spatial size
        final_size = 32
        final_size = final_size // self.pool_stride1
        final_size = final_size // self.pool_stride2
        final_size = final_size // self.pool_stride3
        
        # Fully connected to latent space
        self.fc = nn.Linear(out_channels_3 * final_size * final_size, latent_dim)
        
    def forward(self, x):
        # Layer 1
        x = self.taxon_conv1(x)
        x = F.relu(x)
        if self.use_maxpool:
            x = F.max_pool2d(x, self.pool_stride1)
        else:
            x = F.avg_pool2d(x, self.pool_stride1)
        
        # Layer 2
        x = self.taxon_conv2(x)
        x = F.relu(x)
        if self.use_maxpool:
            x = F.max_pool2d(x, self.pool_stride2)
        else:
            x = F.avg_pool2d(x, self.pool_stride2)
        
        # Layer 3
        x = self.taxon_conv3(x)
        x = F.relu(x)
        if self.pool_stride3 > 1:
            if self.use_maxpool:
                x = F.max_pool2d(x, self.pool_stride3)
            else:
                x = F.avg_pool2d(x, self.pool_stride3)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
