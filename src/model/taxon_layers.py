"""
Taxonomic Layers for Neural Networks

This module implements convolutional layers that enforce taxonomic (hierarchical)
constraints on the learned weights, allowing the network to learn features at
multiple levels of abstraction simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TaxonConv(nn.Module):
    """
    Taxonomic Convolutional Layer
    
    How Taxonomic Constraints Work:
    --------------------------------
    1. TREE STRUCTURE: The layer creates a binary tree of depth n_layers
       - Leaves: 2^n_layers independent filters (finest level)
       - Each parent: weighted average of its 2 children (coarser level)
       - Root: single filter representing the coarsest feature
    
    2. CONVEX COMBINATIONS: Each parent filter is computed as:
       parent = α * child_left + (1-α) * child_right
       where α ∈ [0,1] is learned via sigmoid(α_raw / temperature)
       
       This ensures:
       - Parent filters are ALWAYS valid combinations of children
       - Information flows bottom-up: leaves → intermediate → root
       - Hierarchy is strictly enforced (parents can't diverge from children)
    
    3. TAXONOMIC HIERARCHY EXAMPLE (n_layers=3):
       Level 0 (root):     [Filter_0]                    → 1 filter
                          /          \
       Level 1:      [F_0]            [F_1]              → 2 filters
                     /    \          /    \
       Level 2:    [F_0]  [F_1]    [F_2]  [F_3]         → 4 filters
                   /  \   /  \     /  \   /  \
       Level 3:  [8 leaf filters]                        → 8 filters
       
       Total output channels = 1 + 2 + 4 + 8 = 15
    
    4. CONSTRAINT ENFORCEMENT:
       - Leaf weights: Learned directly (2^n_layers, in_ch, k, k)
       - Alpha parameters: Control mixing at each level
       - Parent weights: Computed dynamically via convex combinations
       - Result: Strict tree hierarchy where coarse features are composites
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels
    kernel_size : int
        Size of the convolutional kernel (k×k)
    n_layers : int
        Depth of the taxonomic tree (creates 2^n_layers leaf filters)
    temperature : float
        Controls the sharpness of alpha values (lower = more binary)
        
    Forward Pass:
    -------------
    Input: (batch, in_channels, H, W)
    Output: (batch, sum(2^i for i in range(n_layers+1)), H, W)
            Concatenation of features from all hierarchy levels
    
    Example:
    --------
    >>> layer = TaxonConv(in_channels=3, kernel_size=3, n_layers=3)
    >>> x = torch.randn(2, 3, 32, 32)
    >>> out = layer(x)
    >>> print(out.shape)  # torch.Size([2, 15, 32, 32])
    """
    
    def __init__(self, in_channels=1, kernel_size=5, n_layers=3, temperature=1.0):
        super(TaxonConv, self).__init__()
        self.in_channels = in_channels
        self.temperature = temperature
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        # Leaf-level weights: (2^n_layers, in_channels, kernel_size, kernel_size)
        self.leaves_weights = nn.Parameter(
            torch.empty(2**n_layers, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.leaves_weights, a=np.sqrt(5))

        self.leaves_bias = nn.Parameter(torch.zeros(2**n_layers))
        nn.init.zeros_(self.leaves_bias)

        # Alpha parameters control mixing at each level (stored root-to-leaves)
        alphas = [nn.Parameter(torch.zeros(2**i, 1)) for i in range(n_layers)]
        alphas.reverse()
        self.alphas = nn.ParameterList(alphas)

    def forward(self, x):
        """
        Forward pass that computes hierarchical features.
        
        Process:
        1. Start with leaf weights (finest level)
        2. For each level moving up the tree:
           - Compute alpha (mixing coefficient) via sigmoid
           - Combine pairs of children into parents using convex combination
        3. Apply convolution at each level
        4. Concatenate all hierarchy levels into output
        """
        B, C, H, W = x.shape
        weights = [self.leaves_weights]
        biases = [self.leaves_bias]
        
        # Build hierarchy bottom-up via convex combinations
        for lvl in range(self.n_layers):
            alpha_raw = torch.sigmoid(self.alphas[lvl] / self.temperature)
            
            # Combine child weights: parent = α*child_0 + (1-α)*child_1
            child_w = weights[-1]
            # Use actual child weight dimensions, not self.in_channels
            num_children, in_ch, kh, kw = child_w.shape
            child_w = child_w.view(
                alpha_raw.shape[0], 2, in_ch, kh, kw
            )
            a_w = alpha_raw.view(alpha_raw.shape[0], 1, 1, 1, 1)
            a_w = torch.cat([a_w, 1 - a_w], dim=1)
            parent_w = torch.sum(a_w * child_w, dim=1)
            weights.append(parent_w)

            # Combine biases
            child_b = biases[-1].view(alpha_raw.shape[0], 2)
            a_b = alpha_raw.view(alpha_raw.shape[0], 1)
            a_b = torch.cat([a_b, 1 - a_b], dim=1)
            parent_b = torch.sum(a_b * child_b, dim=1)
            biases.append(parent_b)

        weights = weights[::-1]
        biases = biases[::-1]

        # Apply convolution at each level and concatenate
        pad = self.kernel_size // 2
        outs = [
            F.conv2d(x, w, bias=b, stride=1, padding=pad)
            for w, b in zip(weights, biases)
        ]
        out = torch.cat(outs, dim=1)

        return out

    def get_hierarchy_weights(self):
        """
        Extract the computed weight tensors for each level of the hierarchy.
        
        Returns:
        --------
        list of torch.Tensor
            Weight tensors ordered from root to leaves:
            [root: (1, in_ch, k, k), 
             level1: (2, in_ch, k, k), 
             ..., 
             leaves: (2^n_layers, in_ch, k, k)]
        """
        weights = [self.leaves_weights]
        for alpha in self.alphas:
            alpha_sig = torch.sigmoid(alpha / self.temperature)
            children = weights[-1]
            children = children.view(
                alpha_sig.shape[0], 2,
                self.in_channels,
                self.kernel_size,
                self.kernel_size
            )
            a = alpha_sig.view(alpha_sig.shape[0], 1, 1, 1, 1)
            a = torch.cat([a, 1 - a], dim=1)
            parent = (a * children).sum(dim=1)
            weights.append(parent)
        return weights[::-1]

    def num_output_channels(self):
        """Calculate total number of output channels."""
        return sum(2**i for i in range(self.n_layers + 1))


class TaxonDeconv(nn.Module):
    """
    Taxonomic Deconvolutional (Transposed Convolutional) Layer
    
    This layer implements the same hierarchical (taxonomic) structure as TaxonConv,
    but uses transposed convolutions for upsampling operations. It's useful for
    decoder architectures in autoencoders, GANs, or segmentation networks.
    
    How Taxonomic Constraints Work:
    --------------------------------
    The hierarchical constraint mechanism is identical to TaxonConv:
    
    1. TREE STRUCTURE: Binary tree of depth n_layers
       - Leaves: 2^n_layers independent filters (finest level)
       - Each parent: weighted average of its 2 children (coarser level)
       - Root: single filter representing the coarsest feature
    
    2. CONVEX COMBINATIONS: Each parent filter is computed as:
       parent = α * child_left + (1-α) * child_right
       where α ∈ [0,1] is learned via sigmoid(α_raw / temperature)
    
    3. TRANSPOSED CONVOLUTION: Instead of F.conv2d, uses F.conv_transpose2d
       - Upsamples spatial dimensions (increases H, W)
       - Useful for decoder/generator architectures
       - Maintains same hierarchical feature learning
    
    4. HIERARCHY EXAMPLE (n_layers=3):
       Level 0 (root):     [Filter_0]                    → 1 filter
                          /          \
       Level 1:      [F_0]            [F_1]              → 2 filters
                     /    \          /    \
       Level 2:    [F_0]  [F_1]    [F_2]  [F_3]         → 4 filters
                   /  \   /  \     /  \   /  \
       Level 3:  [8 leaf filters]                        → 8 filters
       
       Total output channels = 1 + 2 + 4 + 8 = 15
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels (will be multiplied across hierarchy)
    kernel_size : int
        Size of the deconvolutional kernel (k×k)
    n_layers : int
        Depth of the taxonomic tree (creates 2^n_layers leaf filters)
    stride : int
        Stride for the transposed convolution (default: 2 for upsampling)
    padding : int
        Padding for the transposed convolution (default: 1)
    temperature : float
        Controls the sharpness of alpha values (lower = more binary)
        
    Forward Pass:
    -------------
    Input: (batch, in_channels, H, W)
    Output: (batch, out_channels * sum(2^i for i in range(n_layers+1)), H', W')
            where H' and W' depend on stride and padding
            Concatenation of features from all hierarchy levels
    
    Example:
    --------
    >>> layer = TaxonDeconv(in_channels=64, out_channels=32, kernel_size=4, 
    ...                     n_layers=3, stride=2, padding=1)
    >>> x = torch.randn(2, 64, 16, 16)
    >>> out = layer(x)
    >>> print(out.shape)  # torch.Size([2, 480, 32, 32])
    >>> # 480 = 32 * (1 + 2 + 4 + 8) = 32 * 15
    """
    
    def __init__(self, in_channels, out_channels=1, kernel_size=4, n_layers=3, 
                 stride=2, padding=1, output_padding=0, temperature=1.0):
        super(TaxonDeconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temperature = temperature
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Leaf weights for conv_transpose2d: (in_ch, out_ch*2^n_layers, k, k)
        self.leaves_weights = nn.Parameter(
            torch.empty(in_channels, out_channels * (2**n_layers), kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.leaves_weights, a=np.sqrt(5))

        self.leaves_bias = nn.Parameter(torch.zeros(out_channels * (2**n_layers)))
        nn.init.zeros_(self.leaves_bias)

        # Alpha parameters (stored root-to-leaves)
        alphas = [nn.Parameter(torch.zeros(2**i, 1)) for i in range(n_layers)]
        alphas.reverse()
        self.alphas = nn.ParameterList(alphas)

    def forward(self, x):
        """
        Forward pass that computes hierarchical deconvolutional features.
        
        Process:
        1. Start with leaf weights (finest level)
        2. For each level moving up the tree:
           - Compute alpha (mixing coefficient) via sigmoid
           - Combine pairs of children into parents using convex combination
        3. Apply transposed convolution at each level
        4. Concatenate all hierarchy levels into output
        """
        B, C_in, H, W = x.shape
        weights = [self.leaves_weights]
        biases = [self.leaves_bias]
        
        # Build hierarchy bottom-up via convex combinations
        for lvl in range(self.n_layers):
            alpha_raw = torch.sigmoid(self.alphas[lvl] / self.temperature)
            
            # Combine child weights: parent = α*child_0 + (1-α)*child_1
            child_w = weights[-1]
            child_w = child_w.view(
                self.in_channels, alpha_raw.shape[0], 2, 
                self.out_channels, self.kernel_size, self.kernel_size
            )
            a_w = alpha_raw.view(alpha_raw.shape[0], 1, 1, 1, 1)
            a_w = torch.cat([a_w, 1 - a_w], dim=1)
            a_w = a_w.unsqueeze(0)
            parent_w = torch.sum(a_w * child_w, dim=2)
            parent_w = parent_w.view(
                self.in_channels, alpha_raw.shape[0] * self.out_channels,
                self.kernel_size, self.kernel_size
            )
            weights.append(parent_w)

            # Combine biases
            child_b = biases[-1].view(alpha_raw.shape[0], 2, self.out_channels)
            a_b = alpha_raw.view(alpha_raw.shape[0], 1, 1)
            a_b = torch.cat([a_b, 1 - a_b], dim=1)
            parent_b = torch.sum(a_b * child_b, dim=1)
            parent_b = parent_b.view(-1)
            biases.append(parent_b)

        weights = weights[::-1]
        biases = biases[::-1]

        # Apply transposed convolution at each level and concatenate
        outs = [
            F.conv_transpose2d(x, w, bias=b, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
            for w, b in zip(weights, biases)
        ]
        out = torch.cat(outs, dim=1)

        return out

    def get_hierarchy_weights(self):
        """
        Extract the computed weight tensors for each level of the hierarchy.
        
        Returns:
        --------
        list of torch.Tensor
            Weight tensors ordered from root to leaves:
            [root: (in_ch, out_ch, k, k), 
             level1: (in_ch, 2*out_ch, k, k), 
             ..., 
             leaves: (in_ch, 2^n_layers*out_ch, k, k)]
        """
        weights = [self.leaves_weights]
        for alpha in self.alphas:
            alpha_sig = torch.sigmoid(alpha / self.temperature)
            children = weights[-1]
            children = children.view(
                self.in_channels, alpha_sig.shape[0], 2,
                self.out_channels, self.kernel_size, self.kernel_size
            )
            a = alpha_sig.view(alpha_sig.shape[0], 1, 1, 1, 1)
            a = torch.cat([a, 1 - a], dim=1)
            a = a.unsqueeze(0)
            parent = (a * children).sum(dim=2)
            parent = parent.view(
                self.in_channels, alpha_sig.shape[0] * self.out_channels,
                self.kernel_size, self.kernel_size
            )
            weights.append(parent)
        return weights[::-1]

    def num_output_channels(self):
        """Calculate total number of output channels."""
        return self.out_channels * sum(2**i for i in range(self.n_layers + 1))
