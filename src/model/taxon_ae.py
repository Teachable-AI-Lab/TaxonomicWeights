"""
Autoencoder using Taxonomic Convolutional and Deconvolutional Layers
"""

import torch
import torch.nn as nn
from .encoder import CIFAR10TaxonEncoder
from .decoder import CIFAR10TaxonDecoder


class CIFAR10TaxonAutoencoder(nn.Module):
    """
    Taxonomic Autoencoder for CIFAR-10
    
    Uses hierarchical convolutional filters where parent filters are constrained
    as convex combinations of children, providing natural regularization.
    
    Architecture:
    - Encoder: 3 TaxonConv layers → latent (256-d)
    - Decoder: 2 TaxonDeconv layers → reconstruction (3x32x32)
    """
    
    def __init__(self, latent_dim=256, temperature=1.0):
        super(CIFAR10TaxonAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.encoder = CIFAR10TaxonEncoder(latent_dim=latent_dim, temperature=temperature)
        self.decoder = CIFAR10TaxonDecoder(latent_dim=latent_dim, temperature=temperature)
        
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


class CIFAR10TaxonVariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) using Taxonomic layers for CIFAR-10
    
    Extends the standard autoencoder with a probabilistic latent space,
    learning a distribution over latent codes rather than deterministic encodings.
    
    Taxonomic Constraint: Maintains hierarchical feature learning while
    enabling generative sampling from the learned latent distribution.
    
    Example:
    --------
    >>> model = CIFAR10TaxonVariationalAutoencoder(latent_dim=256)
    >>> x = torch.randn(4, 3, 32, 32)
    >>> x_recon, mu, logvar = model(x)
    >>> # Sample new images
    >>> z = torch.randn(4, 256)
    >>> samples = model.decode(z)
    """
    
    def __init__(self, latent_dim=256, temperature=1.0):
        super(CIFAR10TaxonVariationalAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Use same encoder backbone
        self.encoder = CIFAR10TaxonEncoder(latent_dim=latent_dim * 2, temperature=temperature)
        self.decoder = CIFAR10TaxonDecoder(latent_dim=latent_dim, temperature=temperature)
        
        # Split encoder output into mu and logvar
        self.fc_mu = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim * 2, latent_dim)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1.0):
        """VAE loss = MSE reconstruction + beta * KL divergence"""
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl_div
