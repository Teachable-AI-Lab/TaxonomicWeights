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
    - MaxPool: 32x32 → 16x16
    - TaxonConv layer 2: 31→127 channels (n_layers=6), spatial: 16x16
    - MaxPool: 16x16 → 8x8
    - TaxonConv layer 3: 127→255 channels (n_layers=7), spatial: 8x8
    - Flatten → Latent vector
    
    Taxonomic Constraint: Each layer uses hierarchical filters where parent
    filters are convex combinations of child filters, learning multi-scale
    features at each level.
    """
    
    def __init__(self, latent_dim=256, temperature=1.0):
        super(CIFAR10TaxonEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # TaxonConv: outputs 1+2+4+8+16=31 channels for n_layers=4
        self.taxon_conv1 = TaxonConv(
            in_channels=3, kernel_size=3, n_layers=4, temperature=temperature
        )
        
        # TaxonConv: outputs 1+2+4+8+16+32+64=127 channels for n_layers=6
        self.taxon_conv2 = TaxonConv(
            in_channels=31, kernel_size=3, n_layers=6, temperature=temperature
        )
        
        # TaxonConv: outputs 1+2+4+8+16+32+64+128=255 channels for n_layers=7
        self.taxon_conv3 = TaxonConv(
            in_channels=127, kernel_size=3, n_layers=7, temperature=temperature
        )
        
        # Fully connected to latent space
        self.fc = nn.Linear(255 * 8 * 8, latent_dim)
        
    def forward(self, x):
        x = self.taxon_conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.taxon_conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = self.taxon_conv3(x)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
