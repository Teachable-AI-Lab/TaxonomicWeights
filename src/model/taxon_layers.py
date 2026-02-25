"""
Taxonomic Layers for Neural Networks

This module implements convolutional layers that enforce taxonomic (hierarchical)
constraints on the learned weights, allowing the network to learn features at
multiple levels of abstraction simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid.

    Delegates directly to ``torch.sigmoid`` which uses a piecewise-stable
    CUDA/CPU kernel internally and preserves the input dtype — including fp16
    when Automatic Mixed Precision (AMP) is active.  The previous hand-rolled
    boolean-index implementation broke under AMP because ``torch.empty_like``
    produced an fp16 tensor while the RHS arithmetic was promoted to fp32,
    causing "Index put requires source and destination dtypes match" at runtime.
    """
    return torch.sigmoid(x)


class ResidualConvBlock(nn.Module):
    """Pre-activated residual block: BN -> Conv -> ReLU -> BN -> Conv + skip.

    This is the building block used by :class:`TaxonConv`.  Each depth
    of the probabilistic hierarchy has its own ``ResidualConvBlock`` whose
    output is mapped through sigmoid to produce conditional Bernoulli
    probabilities.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    stride : int
        Stride applied to the *first* convolution (and the skip projection)
        for spatial downsampling.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            if (in_channels != out_channels) or (stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


class ResidualDeconvBlock(nn.Module):
    """Pre-activated residual block with transposed convolution for upsampling.

    Mirrors :class:`ResidualConvBlock` exactly, but uses ``ConvTranspose2d``
    for spatial upsampling instead of strided ``Conv2d`` for downsampling.

    Architecture: BN -> ConvTranspose2d(stride) -> ReLU -> BN -> Conv2d(1) + skip.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    stride : int
        Stride for the transposed convolution.
    padding : int
    output_padding : int
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 padding=1, output_padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                               padding, output_padding, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1,
                      padding=kernel_size // 2, bias=False),
        )
        self.skip = (
            nn.ConvTranspose2d(in_channels, out_channels, 1, stride,
                               padding=0, output_padding=output_padding, bias=False)
            if (in_channels != out_channels) or (stride != 1)
            else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


class TaxonConv(nn.Module):
    """Taxonomic Convolutional Layer with Residual Blocks and KL Divergence.

    Based on the design in ``taxon-conv-zekun.ipynb``.  Each depth of the
    probabilistic hierarchy uses a :class:`ResidualConvBlock` (BN-Conv-ReLU-
    BN-Conv + skip) instead of a plain ``nn.Conv2d``.  Every residual block
    takes the **original input** ``x`` and produces conditional Bernoulli
    probabilities via sigmoid.  Parent log-probabilities are accumulated
    down the binary tree, and KL divergence against a uniform distribution
    is computed as regularisation.

    Output channels = ``2 + 4 + … + 2^n_layers``
        (same channel layout as :class:`TaxonConv`).

    Returns ``(output_tensor, dkl)`` — treat identically to
    :class:`TaxonConv` in training loops and analysis scripts.

    Parameters
    ----------
    in_channels : int
    kernel_size : int
    n_layers : int
        Depth of the probabilistic tree.
    stride : int
        Spatial stride baked into every :class:`ResidualConvBlock`.
    temperature : float
    """

    def __init__(self, in_channels=1, kernel_size=3, n_layers=3, stride=1,
                 temperature=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.temperature = temperature
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride

        # One ResidualConvBlock per depth: out_channels = 2^i before binary
        # split doubles them to 2^(i+1).
        self.convs = nn.ModuleList([
            ResidualConvBlock(
                in_channels=self.in_channels,
                out_channels=(1 << i),      # 2**i
                kernel_size=kernel_size,
                stride=self.stride,
            )
            for i in range(self.n_layers)
        ])

    def forward(self, x):
        outputs = []
        prev = None
        dkl = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for idx, conv in enumerate(self.convs):
            # Residual block → sigmoid → probabilities
            ll = conv(x) / self.temperature
            ll = torch.sigmoid(ll)

            # Force float32 for log to avoid float16 underflow
            ll = ll.float()
            ll = torch.clamp(ll, 1e-6, 1 - 1e-6)

            # Binary split: [p, 1-p] → doubles channels
            ll = torch.stack([ll, 1 - ll], dim=2).flatten(1, 2)
            logp = ll.log()

            # Accumulate parent log-probs
            if idx == 0:
                out = logp
            else:
                out = logp + prev.repeat_interleave(2, dim=1)

            # KL against uniform at this depth
            out_expected = torch.full_like(out, 0.5 / (2 ** idx))
            dkl_raw = F.kl_div(
                input=out, target=out_expected,
                reduction='none', log_target=False,
            )
            dkl = dkl + dkl_raw.sum(dim=1).mean()

            outputs.append(out)
            prev = out

        return torch.cat(outputs, dim=1), dkl

    def num_output_channels(self):
        """Total output channels: 2 + 4 + … + 2^n_layers."""
        return sum(2 ** i for i in range(1, self.n_layers + 1))

    def get_hierarchy_weights(self):
        """Return the first Conv2d weight tensor from each depth's residual block."""
        weights = []
        for conv_block in self.convs:
            for m in conv_block.block:
                if isinstance(m, nn.Conv2d):
                    weights.append(m.weight.detach())
                    break
        return weights


class TaxonDeconv(nn.Module):
    """Taxonomic Deconvolutional Layer with Residual Blocks and KL Divergence.

    Transposed-convolution analogue of :class:`TaxonConv`.  Uses
    :class:`ResidualDeconvBlock` at each depth for upsampling.  Returns
    ``(output_tensor, dkl)`` — treat identically to :class:`TaxonDeconv`.

    Output channels = ``out_channels × (2 + 4 + … + 2^n_layers)``.
    """

    def __init__(self, in_channels, out_channels=1, kernel_size=3, n_layers=3,
                 stride=2, padding=1, output_padding=0, temperature=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temperature = temperature
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # One ResidualDeconvBlock per depth
        self.deconvs = nn.ModuleList([
            ResidualDeconvBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels * (1 << i),   # out_ch * 2**i
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
            for i in range(self.n_layers)
        ])

    def forward(self, x):
        outputs = []
        prev = None
        dkl = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for idx, deconv in enumerate(self.deconvs):
            ll = deconv(x) / self.temperature
            ll = torch.sigmoid(ll)

            # Force float32 for log
            ll = ll.float()
            ll = torch.clamp(ll, 1e-6, 1 - 1e-6)

            # Binary split
            ll = torch.stack([ll, 1 - ll], dim=2).flatten(1, 2)
            logp = ll.log()

            if idx == 0:
                out = logp
            else:
                # Upsample previous-level log-probs to match current spatial dims
                prev_upsampled = F.interpolate(prev, size=logp.shape[2:], mode='nearest')
                out = logp + prev_upsampled.repeat_interleave(2, dim=1)

            # KL against uniform
            out_expected = torch.full_like(out, 0.5 / (2 ** idx))
            dkl_raw = F.kl_div(
                input=out, target=out_expected,
                reduction='none', log_target=False,
            )
            dkl = dkl + dkl_raw.sum(dim=1).mean()

            outputs.append(out)
            prev = out

        return torch.cat(outputs, dim=1), dkl

    def num_output_channels(self):
        return self.out_channels * sum(2 ** i for i in range(1, self.n_layers + 1))

    def get_hierarchy_weights(self):
        """Return the first ConvTranspose2d weight tensor from each depth."""
        weights = []
        for deconv_block in self.deconvs:
            for m in deconv_block.block:
                if isinstance(m, nn.ConvTranspose2d):
                    weights.append(m.weight.detach())
                    break
        return weights



