"""Matryoshka Batch-TopK Convolutional SAE encoder.

Two key innovations over the standard TopK SAE encoder:

1. **Batch TopK**: Instead of applying TopK independently per spatial position
   (``[b, :, h, w]``), a shared global threshold is determined from the entire
   batch jointly.  The *total* number of active (position, channel) pairs is
   exactly ``k × B × H × W``, but individual positions may use more or fewer
   features depending on image content.

2. **Matryoshka nesting**: The encoder produces sparse latents at every level
   in ``k_values`` (stored descending, e.g. ``[64, 32, 16]``).  Because all
   levels share the same global ranking of activations, the top-k_small latent
   is a *strict subset* of the top-k_large latent — the matryoshka nesting
   property comes for free.

Interface:

    latent, info = encoder(x)                     # largest-k latent (inference)
    latents_list, info = encoder.forward_matryoshka(x)  # all k-level latents

``info["sparsity"]`` is the raw (unweighted) AuxK auxiliary scalar computed
on the largest-k latent, matching the convention in
:mod:`~src.model.topk_sae_encoder`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..taxon.encoder import resolve_resnet_stage_blocks
from .sae_encoder import ConvSAEStage


# ---------------------------------------------------------------------------
# Batch-global TopK activation
# ---------------------------------------------------------------------------

def batch_topk_activation(x: torch.Tensor, k: int) -> torch.Tensor:
    """Batch-global TopK activation.

    Selects the *globally* top-``k × (B × H × W)`` (position, channel) pairs
    across all batch items and spatial locations jointly, zeroing all others.
    This allows the model to allocate more features to complex regions and
    fewer to homogeneous areas — unlike per-position TopK which forces exactly
    ``k`` active channels at every location.

    Args:
        x: Pre-ReLU feature map ``[B, C, H, W]``.
        k: Target active channels per spatial position **on average**.

    Returns:
        Sparse feature map of the same shape as ``x``.
    """
    B, C, H, W = x.shape
    N = B * H * W
    flat = x.permute(0, 2, 3, 1).reshape(N, C)   # [N, C]
    flat_relu = F.relu(flat)

    k_eff = min(k, C)
    k_total = k_eff * N
    total_elements = N * C

    if k_total >= total_elements:
        return flat_relu.reshape(B, H, W, C).permute(0, 3, 1, 2)

    # Identify the exact top-k_total (position, channel) pairs globally.
    # Using index-based masking avoids threshold-tie ambiguity.
    flat_all = flat_relu.reshape(-1)               # [N*C]
    _, top_idxs = flat_all.topk(k_total)
    mask_flat = torch.zeros_like(flat_all)
    mask_flat.scatter_(0, top_idxs, 1.0)
    mask = mask_flat.reshape(N, C)

    latent = flat_relu * mask
    return latent.reshape(B, H, W, C).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# AuxK auxiliary loss (batch-global variant)
# ---------------------------------------------------------------------------

def batch_auxk_loss(
    pre_relu: torch.Tensor,
    latent: torch.Tensor,
    k: int,
    k_aux: int,
    dead_threshold: float = 1e-3,
) -> torch.Tensor:
    """AuxK auxiliary loss for the Batch-TopK setting.

    Encourages dead channels (mean activation below ``dead_threshold`` across
    the entire batch and spatial dims) to produce larger pre-activation values
    so they can eventually compete in the global selection.

    Args:
        pre_relu: Pre-ReLU feature map ``[B, C, H, W]``.
        latent: Batch-TopK latent (post-activation) ``[B, C, H, W]``.
        k: Active channels per position used by main TopK.
        k_aux: Number of auxiliary features for AuxK loss.
        dead_threshold: Mean abs-activation threshold for "dead" channels.

    Returns:
        Scalar auxiliary loss (0 when no dead features exist).
    """
    B, C, H, W = pre_relu.shape
    k_aux = min(k_aux, max(1, C - k))

    mean_act = latent.abs().mean(dim=(0, 2, 3))   # [C]
    dead_mask = mean_act < dead_threshold

    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return pre_relu.new_zeros(())

    dead_idx = dead_mask.nonzero(as_tuple=True)[0]     # [n_dead]
    dead_pre = pre_relu[:, dead_idx, :, :]              # [B, n_dead, H, W]

    k_aux_eff = min(k_aux, n_dead)
    flat_dead = dead_pre.permute(0, 2, 3, 1).reshape(-1, n_dead)  # [N, n_dead]
    topk_aux_vals, _ = flat_dead.topk(k_aux_eff, dim=1)

    return (topk_aux_vals ** 2).mean()


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class MatryoshkaBatchTopKSAEEncoder(nn.Module):
    """ResNet-style encoder for the Matryoshka Batch-TopK Sparse Conv AE.

    Architecture is identical to :class:`~src.model.topk_sae_encoder.TopKSAEEncoder`
    but uses a **batch-global** threshold rather than per-position TopK, and
    produces latents at *multiple* sparsity levels for the Matryoshka loss.

    Args:
        in_channels: Input image channels.
        resnet_variant: ResNet preset for stage block counts.
        stage_channels: Output channel count per stage.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit per-stage block counts (overrides
            ``resnet_variant`` if provided).
        k_values: Sequence of active-channel counts (will be sorted descending).
            E.g. ``[64, 32, 16]``.  The first (largest) value is used for the
            primary latent returned by :meth:`forward`.
        k_aux: Number of auxiliary features for the AuxK loss.
            Defaults to the largest k in ``k_values``.
        use_aux_loss: Whether to compute and return the AuxK auxiliary loss.
        dead_threshold: Mean activation threshold below which a channel is
            treated as dead for AuxK.
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
        k_values: Sequence[int] = (64, 32, 16),
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

        # Sort k_values descending (largest k first)
        self.k_values: Tuple[int, ...] = tuple(sorted(k_values, reverse=True))
        if len(self.k_values) < 1:
            raise ValueError("k_values must contain at least one element.")

        self.in_channels = in_channels
        self.resnet_variant = str(resnet_variant)
        self.stage_blocks = tuple(int(v) for v in resolved_blocks)
        self.stage_channels = tuple(int(v) for v in stage_channels)
        self.stage_strides = tuple(int(v) for v in stage_strides)
        self.k_aux = int(k_aux) if k_aux is not None else int(self.k_values[0])
        self.use_aux_loss = bool(use_aux_loss)
        self.dead_threshold = float(dead_threshold)
        self.sparsity_type = "matryoshka_batch_topk"
        self.use_stem = bool(use_stem)

        # ----- Stem --------------------------------------------------------
        stem_ops: List[nn.Module]
        if use_stem:
            stem_ops = [
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

        for n_blocks, out_ch, stride in zip(
            self.stage_blocks, self.stage_channels, self.stage_strides
        ):
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
    # Internal helpers
    # -----------------------------------------------------------------------

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run stem + all stages, returning the pre-activation feature map."""
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x

    # -----------------------------------------------------------------------
    # Forward (primary — largest k, for inference / analysis)
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode to the largest-k sparse latent (used during inference).

        Args:
            x: Input image tensor ``[B, C, H, W]``.
            return_details: If ``True``, populate ``shape_trace`` in details.

        Returns:
            ``(latent, info)`` where ``info["sparsity"]`` is the AuxK scalar
            computed at the largest k (0 when ``use_aux_loss=False``).
        """
        details: Dict[str, object] = {"input_shape": tuple(x.shape), "shape_trace": []}

        pre_relu = self._backbone(x)
        if return_details:
            details["shape_trace"].append(("backbone", tuple(pre_relu.shape)))

        k0 = self.k_values[0]
        latent = batch_topk_activation(pre_relu, k0)

        if self.use_aux_loss and self.training:
            sparsity = batch_auxk_loss(
                pre_relu, latent, k0, self.k_aux, self.dead_threshold
            )
        else:
            sparsity = pre_relu.new_zeros(())

        details["latent_shape"] = tuple(latent.shape)
        details["sparsity"] = sparsity
        details["k_values"] = self.k_values

        return latent, details

    # -----------------------------------------------------------------------
    # Forward Matryoshka (training — all k levels)
    # -----------------------------------------------------------------------

    def forward_matryoshka(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Dict[str, object]]:
        """Encode to all k-level latents simultaneously.

        The latents have the matryoshka nesting property:
        ``latents[i]`` (smaller k) is a strict subset of ``latents[i-1]``
        (larger k) because they are computed from the same global ranking.

        Args:
            x: Input image tensor ``[B, C, H, W]``.

        Returns:
            ``(latents_list, info)`` where ``latents_list[i]`` is the sparse
            feature map at ``self.k_values[i]`` and ``info["sparsity"]`` is
            the AuxK scalar (computed at the largest k).
        """
        pre_relu = self._backbone(x)

        latents: List[torch.Tensor] = []
        for k in self.k_values:
            latents.append(batch_topk_activation(pre_relu, k))

        if self.use_aux_loss and self.training:
            sparsity = batch_auxk_loss(
                pre_relu, latents[0], self.k_values[0], self.k_aux, self.dead_threshold
            )
        else:
            sparsity = pre_relu.new_zeros(())

        info: Dict[str, object] = {
            "latent_shape": tuple(latents[0].shape),
            "sparsity": sparsity,
            "k_values": self.k_values,
        }
        return latents, info


__all__ = [
    "batch_topk_activation",
    "batch_auxk_loss",
    "MatryoshkaBatchTopKSAEEncoder",
]
