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
    initial_spatial_size : int
        Spatial size after unflattening latent vector (default: 8 for 8x8)
    """
    
    def __init__(self, latent_dim=256, temperature=1.0, kernel_sizes=None, 
                 strides=None, paddings=None, initial_spatial_size=8):
        super(CIFAR10TaxonDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.initial_spatial_size = initial_spatial_size
        
        # Default kernel sizes and strides
        if kernel_sizes is None:
            kernel_sizes = [4, 4, 3]
        if strides is None:
            strides = [2, 2, 1]
        
        # Handle scalar or list inputs
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * 3
        if isinstance(strides, int):
            strides = [strides] * 3
        
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
        
        # Fully connected from latent space
        self.fc = nn.Linear(latent_dim, 255 * initial_spatial_size * initial_spatial_size)
        
        # TaxonDeconv: 255 input → outputs (1+2+4+8+16+32+64)*out_ch = 127*out_ch
        # Set out_channels=1 to get 127 total output channels
        self.taxon_deconv1 = TaxonDeconv(
            in_channels=255, out_channels=1, kernel_size=kernel_sizes[0], 
            n_layers=6, stride=strides[0], padding=paddings[0], temperature=temperature
        )
        
        # TaxonDeconv: 127 input → outputs (1+2+4+8+16)*out_ch = 31*out_ch
        # Set out_channels=1 to get 31 total output channels
        self.taxon_deconv2 = TaxonDeconv(
            in_channels=127, out_channels=1, kernel_size=kernel_sizes[1],
            n_layers=4, stride=strides[1], padding=paddings[1], temperature=temperature
        )
        
        # TaxonDeconv: 31 input → outputs (1+2)*out_ch = 3*out_ch
        # Set out_channels=1 to get 3 total output channels (RGB)
        self.taxon_deconv3 = TaxonDeconv(
            in_channels=31, out_channels=1, kernel_size=kernel_sizes[2],
            n_layers=1, stride=strides[2], padding=paddings[2], temperature=temperature
        )
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 255, self.initial_spatial_size, self.initial_spatial_size)
        
        x = self.taxon_deconv1(x)
        x = F.relu(x)
        
        x = self.taxon_deconv2(x)
        x = F.relu(x)
        
        x = self.taxon_deconv3(x)
        x = torch.tanh(x)
        
        return x
