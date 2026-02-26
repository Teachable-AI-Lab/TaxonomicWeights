"""
Taxonomic Layers for Neural Networks

This module implements convolutional layers that enforce taxonomic (hierarchical)
constraints on the learned weights, allowing the network to learn features at
multiple levels of abstraction simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


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
    """Pre-activated residual block with resize-convolution upsampling.

    Mirrors :class:`ResidualConvBlock`, but replaces transposed convolution
    with explicit nearest-neighbor upsampling followed by ``Conv2d`` to avoid
    checkerboard artifacts from uneven overlap.

    Architecture: BN -> Upsample(stride) -> Conv2d -> ReLU -> BN -> Conv2d + skip.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    stride : int
        Spatial upsampling factor.
    padding : int
        Kept for API compatibility. Main branch uses ``kernel_size // 2``.
    output_padding : int
        Kept for API compatibility; unused in resize-convolution mode.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 padding=1, output_padding=0):
        super().__init__()
        del padding, output_padding
        self.upsample = (
            nn.Upsample(scale_factor=stride, mode='nearest')
            if stride != 1 else nn.Identity()
        )
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            self.upsample,
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1,
                      padding=kernel_size // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1,
                      padding=kernel_size // 2, bias=False),
        )
        if (in_channels != out_channels) or (stride != 1):
            self.skip = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode='nearest')
                if stride != 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


def _taxonomic_regularization(
    out: torch.Tensor,
    idx: int,
) -> tuple:
    """Compute the two-part taxonomic regularization for one depth level.

    Parameters
    ----------
    out : Tensor, shape ``[B, 2^(idx+1), H, W]``
        Joint (accumulated) log-probabilities at this depth.
    idx : int
        Current depth index (0-based).

    Returns
    -------
    (entropy, batch_kl) : tuple of scalar Tensors
        1. **Per-instance entropy** (to minimize) — encourages each sample
           to commit to few branches (sparse routing).
        2. **Batch-marginal KL to uniform** (to minimize) — encourages
           all branches to be used equally across the batch.

        Both are returned **unweighted**; the training loop applies the
        ``entropy_weight`` and ``batch_kl_weight`` coefficients from config.
    """
    n_branches = out.shape[1]  # 2^(idx+1)

    # Convert joint log-probs to a proper per-instance distribution over
    # branches at this depth via softmax across the channel dimension.
    instance_probs = torch.softmax(out, dim=1)  # [B, n_branches, H, W]

    # ---- Term 1: per-instance entropy (minimize → sparse per sample) ----
    instance_entropy = -(instance_probs * instance_probs.clamp(min=1e-8).log()).sum(dim=1)
    entropy = instance_entropy.mean()

    # ---- Term 2: batch-marginal KL to uniform (minimize → diverse) ------
    batch_marginal = instance_probs.mean(dim=(0, 2, 3))  # [n_branches]
    batch_marginal = batch_marginal / batch_marginal.sum()  # re-normalise
    uniform = torch.full_like(batch_marginal, 1.0 / n_branches)
    # KL(batch_marginal || uniform)
    batch_kl = F.kl_div(
        input=uniform.log(),
        target=batch_marginal,
        reduction='sum',
        log_target=False,
    )

    return entropy, batch_kl


class TaxonConv(nn.Module):
    """Taxonomic Convolutional Layer with Residual Blocks.

    Based on the design in ``taxon-conv-zekun.ipynb``.  Each depth of the
    probabilistic hierarchy uses a :class:`ResidualConvBlock` (BN-Conv-ReLU-
    BN-Conv + skip) instead of a plain ``nn.Conv2d``.  Every residual block
    takes the **original input** ``x`` and produces conditional Bernoulli
    probabilities via sigmoid.  Parent log-probabilities are accumulated
    down the binary tree.

    Regularisation (returned as second element of the forward tuple):
        * **Per-instance entropy minimization** — each sample should commit
          to a sparse subset of branches (decisive routing).
        * **Batch-marginal KL to uniform** — across the batch, all branches
          should be used equally (no branch collapse).

    Output channels = ``2 + 4 + … + 2^n_layers``.

    Returns ``(output_tensor, entropy_reg, batch_kl_reg)``.

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
        total_entropy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        total_batch_kl = torch.tensor(0.0, device=x.device, dtype=x.dtype)

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

            # Taxonomic regularization: sparse per-instance + uniform per-batch
            # Gradient checkpointing frees large softmax/entropy intermediates
            # during forward; they are recomputed on-the-fly during backward.
            ent, bkl = grad_checkpoint(
                _taxonomic_regularization, out, idx, use_reentrant=False
            )
            total_entropy = total_entropy + ent
            total_batch_kl = total_batch_kl + bkl

            outputs.append(out)
            prev = out

        return torch.cat(outputs, dim=1), total_entropy, total_batch_kl

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
    """Taxonomic Deconvolutional Layer with Residual Blocks.

    Upsampling analogue of :class:`TaxonConv`. Uses
    :class:`ResidualDeconvBlock` at each depth. Returns
    ``(output_tensor, entropy_reg, batch_kl_reg)``.

    Regularisation (returned as second element of the forward tuple):
        * **Per-instance entropy minimization** — each sample should commit
          to a sparse subset of branches (decisive routing).
        * **Batch-marginal KL to uniform** — across the batch, all branches
          should be used equally (no branch collapse).

    Output channels = ``out_channels × (2 + 4 + … + 2^n_layers)``.

    Parameters
    ----------
    (see :class:`TaxonConv` for shared parameters)
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
        total_entropy = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        total_batch_kl = torch.tensor(0.0, device=x.device, dtype=x.dtype)

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

            # Taxonomic regularization: sparse per-instance + uniform per-batch
            # Gradient checkpointing frees large softmax/entropy intermediates
            # during forward; they are recomputed on-the-fly during backward.
            ent, bkl = grad_checkpoint(
                _taxonomic_regularization, out, idx, use_reentrant=False
            )
            total_entropy = total_entropy + ent
            total_batch_kl = total_batch_kl + bkl

            outputs.append(out)
            prev = out

        return torch.cat(outputs, dim=1), total_entropy, total_batch_kl

    def num_output_channels(self):
        return self.out_channels * sum(2 ** i for i in range(1, self.n_layers + 1))

    def get_hierarchy_weights(self):
        """Return the first Conv2d weight tensor from each depth."""
        weights = []
        for deconv_block in self.deconvs:
            for m in deconv_block.block:
                if isinstance(m, nn.Conv2d):
                    weights.append(m.weight.detach())
                    break
        return weights

