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
    
    Architecture:
    - Latent vector → Unflatten to (255, 8, 8)
    - TaxonDeconv layer 1: 255→127 channels (n_layers=6), spatial: 8x8 → 16x16
    - TaxonDeconv layer 2: 127→31 channels (n_layers=4), spatial: 16x16 → 32x32
    - Final conv: 31→3 channels, spatial: 32x32
    
    Taxonomic Constraint: Each deconvolutional layer uses hierarchical filters
    where parent filters are convex combinations of child filters, maintaining
    multi-scale feature generation during upsampling.
    """
    
    def __init__(self, latent_dim=256, temperature=1.0):
        super(CIFAR10TaxonDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Fully connected from latent space
        self.fc = nn.Linear(latent_dim, 255 * 8 * 8)
        
        # TaxonDeconv: 255 input → outputs (1+2+4+8+16+32+64)*out_ch = 127*out_ch
        # Set out_channels=1 to get 127 total output channels
        self.taxon_deconv1 = TaxonDeconv(
            in_channels=255, out_channels=1, kernel_size=4, 
            n_layers=6, stride=2, padding=1, temperature=temperature
        )
        
        # TaxonDeconv: 127 input → outputs (1+2+4+8+16)*out_ch = 31*out_ch
        # Set out_channels=1 to get 31 total output channels
        self.taxon_deconv2 = TaxonDeconv(
            in_channels=127, out_channels=1, kernel_size=4,
            n_layers=4, stride=2, padding=1, temperature=temperature
        )
        
        # Final convolution to RGB
        self.final_conv = nn.Conv2d(31, 3, kernel_size=3, padding=1)
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 255, 8, 8)
        
        x = self.taxon_deconv1(x)
        x = F.relu(x)
        
        x = self.taxon_deconv2(x)
        x = F.relu(x)
        
        x = self.final_conv(x)
        x = torch.tanh(x)
        
        return x
