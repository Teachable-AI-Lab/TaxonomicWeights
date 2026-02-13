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


def stable_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid implementation.

    This avoids overflow for large magnitude inputs by computing the
    sigmoid piecewise for positive and negative values.
    """
    pos = x >= 0
    neg = ~pos
    out = torch.empty_like(x)
    # safe for large positive x
    out[pos] = 1.0 / (1.0 + torch.exp(-x[pos]))
    # safe for large negative x
    exp_x = torch.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


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
    
    def __init__(self, in_channels=1, kernel_size=5, n_layers=3, temperature=1.0,
                 random_init_alphas=False, alpha_init_distribution="uniform",
                 alpha_init_range=None, alpha_init_seed=None):
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

        self._init_alphas(random_init_alphas, alpha_init_distribution,
                          alpha_init_range, alpha_init_seed)

    def _init_alphas(self, random_init_alphas, distribution, value_range, seed):
        """Initialize alpha parameters with optional randomness.

        If random_init_alphas is False, all alphas stay at 0 so sigmoid → 0.5.
        """
        if not random_init_alphas:
            # Explicitly zero for clarity (sigmoid(0) = 0.5).
            for alpha in self.alphas:
                alpha.data.zero_()
            return

        # Normalize range input
        if value_range and len(value_range) == 2:
            low, high = float(value_range[0]), float(value_range[1])
        else:
            low, high = -0.5, 0.5

        generator = torch.Generator(device=self.leaves_weights.device)
        if seed is not None:
            generator.manual_seed(int(seed))

        dist = (distribution or "uniform").lower()
        with torch.no_grad():
            for alpha in self.alphas:
                if dist in ("uniform", "uniform_weird"):
                    alpha.data.uniform_(low, high, generator=generator)
                elif dist == "normal":
                    mean = 0.5 * (low + high)
                    std = abs(high - low) / 6 if high != low else 1.0
                    alpha.data.normal_(mean, std, generator=generator)
                else:
                    alpha.data.uniform_(low, high, generator=generator)

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
                 stride=2, padding=1, output_padding=0, temperature=1.0,
                 random_init_alphas=False, alpha_init_distribution="uniform",
                 alpha_init_range=None, alpha_init_seed=None):
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

        self._init_alphas(random_init_alphas, alpha_init_distribution,
                          alpha_init_range, alpha_init_seed)

    def _init_alphas(self, random_init_alphas, distribution, value_range, seed):
        if not random_init_alphas:
            for alpha in self.alphas:
                alpha.data.zero_()
            return

        if value_range and len(value_range) == 2:
            low, high = float(value_range[0]), float(value_range[1])
        else:
            low, high = -0.5, 0.5

        generator = torch.Generator(device=self.leaves_weights.device)
        if seed is not None:
            generator.manual_seed(int(seed))

        dist = (distribution or "uniform").lower()
        with torch.no_grad():
            for alpha in self.alphas:
                if dist in ("uniform", "uniform_weird"):
                    alpha.data.uniform_(low, high, generator=generator)
                elif dist == "normal":
                    mean = 0.5 * (low + high)
                    std = abs(high - low) / 6 if high != low else 1.0
                    alpha.data.normal_(mean, std, generator=generator)
                else:
                    alpha.data.uniform_(low, high, generator=generator)

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


class TaxonConvKL(nn.Module):
    """
    Taxonomic Convolutional Layer with KL Divergence Loss
    
    This variant computes KL divergence between the learned distribution and a uniform
    distribution at each layer. The layer outputs log probabilities that sum to 1 across
    channels at each pixel, encouraging a more probabilistic interpretation.
    
    Key Differences from TaxonConv:
    --------------------------------
    1. Outputs log probabilities instead of raw activations
    2. Computes KL divergence against uniform distribution
    3. Returns tuple: (output, dkl_loss)
    4. Each depth creates 2× channels via binary splits
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels
    kernel_size : int
        Size of the convolutional kernel (k×k)
    n_layers : int
        Depth of the taxonomic tree
    temperature : float
        Controls the sharpness of probabilities
        
    Forward Pass:
    -------------
    Input: (batch, in_channels, H, W)
    Output: (tensor, float)
            tensor: (batch, sum(2^i for i in range(1, n_layers+1)), H, W) log probabilities
            float: KL divergence loss
    """
    
    def __init__(self, in_channels=1, kernel_size=7, n_layers=3, temperature=1.0,
                 random_init_alphas=False, alpha_init_distribution="uniform",
                 alpha_init_range=None, alpha_init_seed=None):
        super(TaxonConvKL, self).__init__()
        self.in_channels = in_channels
        self.temperature = temperature
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        # Some code expects `.alphas` on taxon layers; provide an empty list
        # so callers that iterate `for alpha in layer.alphas` will simply skip.
        self.alphas = nn.ParameterList([])

        # Create a conv for each depth i=1..n_layers, out_channels=2**i
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=(1 << i),  # 2**i
                kernel_size=kernel_size,
                stride=1,
                padding=int(kernel_size // 2),
            )
            for i in range(self.n_layers)
        ])

    def forward(self, x):
        """
        Forward pass computing hierarchical log probabilities and KL divergence.
        
        Returns:
        --------
        output : torch.Tensor
            Concatenated log probabilities from all depths
        dkl : float
            Total KL divergence summed across all layers
        """
        outputs = []
        prev = None
        # initialize dkl as a tensor on the same device/dtype as the input
        dkl = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for idx, conv in enumerate(self.convs):
            # Compute log-likelihood
            ll = conv(x) / self.temperature
            # use numerically-stable sigmoid
            ll = stable_sigmoid(ll)
            
            # Clamp to avoid numerical issues
            ll = torch.clamp(ll, 1e-6, 1 - 1e-6)
            
            # Stack [p, 1-p] and flatten to create binary splits
            ll = torch.stack([ll, 1 - ll], dim=2).flatten(1, 2)  # (B, 2*C, H, W)
            logp = ll.log()  # Convert to log-probabilities
            
            # If not root, add the parent log-probs, repeating to match 2× expansion
            if idx == 0:
                out = logp
            else:
                out = logp + prev.repeat_interleave(2, dim=1)
            
            # Expected uniform distribution at this depth
            out_expected = torch.full_like(out, 0.5 / (2**idx))
            
            # Compute KL divergence: KL(out || out_expected)
            dkl_raw = F.kl_div(
                input=out,
                target=out_expected,
                reduction='none',
                log_target=False
            )
            # Normalize by averaging over all dimensions (batch, channels, H, W)
            dkl = dkl + dkl_raw.mean()
            
            outputs.append(out)
            prev = out

        # Concatenate all depths. For backward compatibility with code that
        # expects a plain tensor (not a (tensor, dkl) tuple), return only
        # the concatenated tensor and store the KL on the module.
        out_tensor = torch.cat(outputs, dim=1)
        try:
            # Ensure dkl is a tensor
            if not isinstance(dkl, torch.Tensor):
                dkl = torch.tensor(dkl, device=out_tensor.device, dtype=out_tensor.dtype)
        except Exception:
            dkl = torch.tensor(0.0, device=out_tensor.device, dtype=out_tensor.dtype)
        self._last_dkl = dkl
        return out_tensor, dkl

    def num_output_channels(self):
        """Calculate total number of output channels."""
        # Each depth i produces 2^(i+1) channels due to binary splitting
        # Depths go from 0 to n_layers-1, producing 2^1, 2^2, ..., 2^n_layers
        return sum(2**i for i in range(1, self.n_layers + 1))

    def get_hierarchy_weights(self):
        """
        Return the convolution weight tensors for each depth ordered from root
        (coarsest) to leaves (finest).

        Returns:
        --------
        list of torch.Tensor
            Each element is the `.weight` tensor from the corresponding
            `nn.Conv2d` in `self.convs` (root -> leaves).
        """
        return [conv.weight.detach() for conv in self.convs]


class TaxonDeconvKL(nn.Module):
    """
    Taxonomic Deconvolutional Layer with KL Divergence Loss
    
    Transposed convolution version of TaxonConvKL. Uses the same probabilistic
    formulation with KL divergence but performs upsampling.
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels per depth (will be multiplied)
    kernel_size : int
        Size of the deconvolutional kernel
    n_layers : int
        Depth of the taxonomic tree
    stride : int
        Stride for upsampling
    padding : int
        Padding for transposed convolution
    output_padding : int
        Additional output padding
    temperature : float
        Controls the sharpness of probabilities
        
    Forward Pass:
    -------------
    Input: (batch, in_channels, H, W)
    Output: (tensor, float)
            tensor: Upsampled log probabilities
            float: KL divergence loss
    """
    
    def __init__(self, in_channels, out_channels=1, kernel_size=4, n_layers=3,
                 stride=2, padding=1, output_padding=0, temperature=1.0,
                 random_init_alphas=False, alpha_init_distribution="uniform",
                 alpha_init_range=None, alpha_init_seed=None):
        super(TaxonDeconvKL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temperature = temperature
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Provide an empty `.alphas` for compatibility with visualization helpers
        self.alphas = nn.ParameterList([])

        # Create a deconv for each depth i=1..n_layers, out_channels=2**i
        self.deconvs = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels * (1 << i),  # out_channels * 2**i
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
            for i in range(self.n_layers)
        ])

    def forward(self, x):
        """
        Forward pass computing hierarchical log probabilities and KL divergence.
        
        Returns:
        --------
        output : torch.Tensor
            Concatenated upsampled log probabilities from all depths
        dkl : float
            Total KL divergence summed across all layers
        """
        outputs = []
        prev = None
        # initialize dkl as a tensor on the same device/dtype as the input
        dkl = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for idx, deconv in enumerate(self.deconvs):
            # Compute log-likelihood
            ll = deconv(x) / self.temperature
            # use numerically-stable sigmoid
            ll = stable_sigmoid(ll)
            
            # Clamp to avoid numerical issues
            ll = torch.clamp(ll, 1e-6, 1 - 1e-6)
            
            # Stack [p, 1-p] and flatten to create binary splits
            ll = torch.stack([ll, 1 - ll], dim=2).flatten(1, 2)
            logp = ll.log()  # Convert to log-probabilities
            
            # If not root, add the parent log-probs, repeating to match 2× expansion
            if idx == 0:
                out = logp
            else:
                # Upsample previous-level log-probs to match current spatial dims,
                # then repeat-interleave channels to match binary expansion.
                prev_upsampled = F.interpolate(prev, size=logp.shape[2:], mode='nearest')
                out = logp + prev_upsampled.repeat_interleave(2, dim=1)
            
            # Expected uniform distribution at this depth
            out_expected = torch.full_like(out, 0.5 / (2**idx))
            
            # Compute KL divergence
            dkl_raw = F.kl_div(
                input=out,
                target=out_expected,
                reduction='none',
                log_target=False
            )
            # Normalize by averaging over all dimensions (batch, channels, H, W)
            dkl = dkl + dkl_raw.mean()
            
            outputs.append(out)
            prev = out

        # Concatenate all depths. Return only the tensor for compatibility and
        # store the KL divergence on the module as `_last_dkl`.
        out_tensor = torch.cat(outputs, dim=1)
        try:
            if not isinstance(dkl, torch.Tensor):
                dkl = torch.tensor(dkl, device=out_tensor.device, dtype=out_tensor.dtype)
        except Exception:
            dkl = torch.tensor(0.0, device=out_tensor.device, dtype=out_tensor.dtype)
        self._last_dkl = dkl
        return out_tensor, dkl

    def num_output_channels(self):
        """Calculate total number of output channels."""
        # Each depth i produces 2^(i+1) channels due to binary splitting
        # Depths go from 0 to n_layers-1, producing 2^1, 2^2, ..., 2^n_layers
        return self.out_channels * sum(2**i for i in range(1, self.n_layers + 1))

    def get_hierarchy_weights(self):
        """
        Return the transposed-convolution weight tensors for each depth ordered
        from root (coarsest) to leaves (finest).

        Returns:
        --------
        list of torch.Tensor
            Each element is the `.weight` tensor from the corresponding
            `nn.ConvTranspose2d` in `self.deconvs` (root -> leaves).
        """
        return [deconv.weight.detach() for deconv in self.deconvs]
