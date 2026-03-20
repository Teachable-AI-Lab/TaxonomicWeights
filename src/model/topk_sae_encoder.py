"""TopK Convolutional SAE encoder.

Replaces the L1 / KL sparsity penalty of :class:`~src.model.sae_encoder.ConvSAEEncoder`
with a **hard TopK activation** (Gao et al., "Scaling and Evaluating Sparse
Autoencoders", OpenAI 2024).

For each (batch, spatial) position the top-k channels by absolute value are
kept; all others are zeroed.  Sparsity is therefore *directly controlled* by
``topk_k`` rather than through a penalty weight — making the sparsity /
reconstruction trade-off much easier to tune.

An optional **AuxK auxiliary loss** (from the same paper) is computed to
discourage dead latent features.  The AuxK loss is the reconstruction error
attributable to the next-best ``k_aux`` features for inputs where those
features are not in the main TopK selection.  Pass ``use_aux_loss=False`` to
disable it; in that case ``info["sparsity"]`` is always 0.

Interface is identical to :class:`ConvSAEEncoder`:

    ``latent, info = encoder(x)``

where ``info["sparsity"]`` is the raw (unweighted) auxiliary scalar.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResidualConvBlock, resolve_resnet_stage_blocks
from .sae_encoder import ConvSAEStage


# ---------------------------------------------------------------------------
# TopK activation
# ---------------------------------------------------------------------------

def topk_activation(x: torch.Tensor, k: int) -> torch.Tensor:
    """Hard TopK activation applied channel-wise per spatial location.

    For each ``[b, :, h, w]`` slice (a C-dimensional vector), only the top-k
    entries by *value* are kept; the remainder are set to zero.  Pre-ReLU
    clipping is applied so negative values cannot enter the latent (matching
    the convention that latent features are non-negative).

    Args:
        x: Feature map of shape ``[B, C, H, W]``.
        k: Number of active channels to retain per position.

    Returns:
        Sparse feature map of the same shape as ``x``.
    """
    B, C, H, W = x.shape
    k = min(k, C)

    # Permute to [B*H*W, C] for efficient topk
    flat = x.permute(0, 2, 3, 1).reshape(-1, C)   # [N, C] where N = B*H*W

    # Apply ReLU first (feature values must be non-negative, as in Gao et al.)
    flat_relu = F.relu(flat)

    if k >= C:
        # All features active — just return relu-activated map
        return flat_relu.reshape(B, H, W, C).permute(0, 3, 1, 2)

    # Select top-k per position
    _, topk_idx = flat_relu.topk(k, dim=1)          # [N, k]
    mask = torch.zeros_like(flat_relu)
    mask.scatter_(1, topk_idx, 1.0)

    # Straight-through estimator for the mask (gradient flows through
    # activated entries; mask stops gradient on the selection decision)
    latent = flat_relu * mask

    return latent.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# AuxK auxiliary loss
# ---------------------------------------------------------------------------

def auxk_loss(
    pre_relu: torch.Tensor,
    latent: torch.Tensor,
    k: int,
    k_aux: int,
    dead_threshold: float = 1e-3,
) -> torch.Tensor:
    """AuxK auxiliary loss to prevent feature death.

    For features that are underutilised (mean activation below
    ``dead_threshold``), compute how much reconstruction error they could
    explain and penalise their non-activation proportionally.

    This is a simplified convolutional adaptation of the Gao et al. AuxK:

    1. Find "dead" channel indices: those whose mean absolute activation
       across the current batch & spatial dims is below ``dead_threshold``.
    2. Among those dead channels, keep the top-``k_aux`` by *pre-relu*
       activation at each position.
    3. AuxK loss = mean squared pre-relu value of those top-``k_aux`` dead
       channel activations (encourages them to grow larger so they will
       eventually be selected by the main TopK).

    Args:
        pre_relu: Pre-ReLU feature map ``[B, C, H, W]``.
        latent:   Main TopK latent (post-activation) ``[B, C, H, W]``.
        k:        Number of features selected by main TopK.
        k_aux:    Number of auxiliary features for AuxK loss.
        dead_threshold: Mean abs-activation below which a channel is "dead".

    Returns:
        Scalar auxiliary loss (0 when no dead features exist).
    """
    B, C, H, W = pre_relu.shape
    k_aux = min(k_aux, max(1, C - k))

    # Identify dead channels (per batch)
    mean_act = latent.abs().mean(dim=(0, 2, 3))   # [C]
    dead_mask = (mean_act < dead_threshold)        # [C] bool

    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return pre_relu.new_zeros(())

    # Extract dead channel pre-relu activations
    dead_idx = dead_mask.nonzero(as_tuple=True)[0]  # [n_dead]
    dead_pre = pre_relu[:, dead_idx, :, :]           # [B, n_dead, H, W]

    # TopK-aux per position among dead features
    k_aux_eff = min(k_aux, n_dead)
    flat_dead = dead_pre.permute(0, 2, 3, 1).reshape(-1, n_dead)  # [N, n_dead]
    topk_aux_vals, _ = flat_dead.top_k(k_aux_eff, dim=1) if False else (
        flat_dead.topk(k_aux_eff, dim=1)
    )
    # AuxK loss: encourage these values to grow
    return (topk_aux_vals ** 2).mean()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TopKSAEEncoder(nn.Module):
    """ResNet-style encoder for the TopK Sparse Convolutional Autoencoder.

    Identical backbone to :class:`~src.model.sae_encoder.ConvSAEEncoder` but
    the L1/KL activation + penalty is replaced with a hard TopK activation.

    Args:
        in_channels: Input image channels.
        resnet_variant: ResNet preset for stage block counts.
        stage_channels: Output channel count per stage.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit per-stage block counts (overrides
            ``resnet_variant`` if provided).
        topk_k: Number of active channels to retain per spatial position.
            If ``<= 0`` the TopK is disabled and plain ReLU is used.
        k_aux: Number of auxiliary features for the optional AuxK loss.
            Defaults to ``topk_k``.
        use_aux_loss: Whether to compute and return the AuxK auxiliary loss.
        dead_threshold: Mean activation threshold below which a channel is
            treated as dead for the AuxK computation.
        kernel_size: Conv kernel size throughout.
        use_stem: Include the 7×7 conv stem.
        stem_channels: Output channels of the stem.
        stem_stride: Stride of the stem conv.
        use_stem_maxpool: Append 3×3 max-pool after stem.
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        topk_k: int = 64,
        k_aux: Optional[int] = None,
        use_aux_loss: bool = True,
        dead_threshold: float = 1e-3,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
    ) -> None:
        super().__init__()

        resolved_blocks = resolve_resnet_stage_blocks(
            resnet_variant=resnet_variant,
            stage_blocks=stage_blocks,
        )

        if not (len(resolved_blocks) == len(stage_channels) == len(stage_strides)):
            raise ValueError(
                "stage_blocks, stage_channels, and stage_strides must have equal length. "
                f"Got {len(resolved_blocks)}, {len(stage_channels)}, {len(stage_strides)}."
            )

        self.in_channels = in_channels
        self.resnet_variant = str(resnet_variant)
        self.stage_blocks = tuple(int(v) for v in resolved_blocks)
        self.stage_channels = tuple(int(v) for v in stage_channels)
        self.stage_strides = tuple(int(v) for v in stage_strides)
        self.topk_k = int(topk_k)
        self.k_aux = int(k_aux) if k_aux is not None else int(topk_k)
        self.use_aux_loss = bool(use_aux_loss)
        self.dead_threshold = float(dead_threshold)
        self.sparsity_type = "topk"   # for compatibility with analysis / compare scripts
        self.use_stem = bool(use_stem)

        # ----- Stem --------------------------------------------------------
        if use_stem:
            stem_ops: List[nn.Module] = [
                nn.Conv2d(
                    in_channels, stem_channels,
                    kernel_size=7, stride=stem_stride, padding=3, bias=False,
                ),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            ]
            if use_stem_maxpool:
                stem_ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.stem = nn.Sequential(*stem_ops)
            current_channels = stem_channels
            self.stem_total_stride = stem_stride * (2 if use_stem_maxpool else 1)
        else:
            self.stem = nn.Identity()
            current_channels = in_channels
            self.stem_total_stride = 1

        self.stem_channels_out = stem_channels if use_stem else in_channels

        # ----- Stages ------------------------------------------------------
        self.stage_input_channels: List[int] = []
        self.stages = nn.ModuleList()

        for n_blocks, out_ch, stride in zip(self.stage_blocks, self.stage_channels, self.stage_strides):
            self.stage_input_channels.append(current_channels)
            self.stages.append(
                ConvSAEStage(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    n_blocks=n_blocks,
                    stride=stride,
                    kernel_size=kernel_size,
                )
            )
            current_channels = out_ch

        self.final_channels = current_channels

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode an image batch to a TopK-sparse latent feature map.

        Args:
            x: Input image tensor ``[B, C, H, W]``.
            return_details: If ``True``, populate ``shape_trace`` in details.

        Returns:
            ``(latent, info)`` where ``info["sparsity"]`` is the AuxK
            auxiliary scalar (0 when ``use_aux_loss=False``).
        """
        details: Dict[str, object] = {"input_shape": tuple(x.shape), "shape_trace": []}

        x = self.stem(x)
        if return_details:
            details["shape_trace"].append(("stem", tuple(x.shape)))

        for idx, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if return_details:
                details["shape_trace"].append((f"stage{idx}", tuple(x.shape)))

        # Save pre-ReLU activations for AuxK
        pre_relu = x

        # Apply TopK activation
        k_eff = self.topk_k if self.topk_k > 0 else int(x.shape[1])
        latent = topk_activation(pre_relu, k_eff)

        # Compute AuxK auxiliary loss
        if self.use_aux_loss and self.training:
            sparsity = auxk_loss(pre_relu, latent, k_eff, self.k_aux, self.dead_threshold)
        else:
            sparsity = pre_relu.new_zeros(())

        details["latent_shape"] = tuple(latent.shape)
        details["sparsity"] = sparsity
        details["topk_k"] = k_eff

        return latent, details


__all__ = [
    "topk_activation",
    "auxk_loss",
    "TopKSAEEncoder",
]
