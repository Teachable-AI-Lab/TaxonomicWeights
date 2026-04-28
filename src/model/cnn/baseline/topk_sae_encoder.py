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

from ..taxon.encoder import ResidualConvBlock, resolve_resnet_stage_blocks
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

def make_dead_latent(
    pre_relu: torch.Tensor,
    steps_since_active: torch.Tensor,
    k_aux: int,
    k: int,
    dead_steps: int,
) -> Optional[torch.Tensor]:
    """Build a "dead-only" sparse latent for proper AuxK reconstruction.

    The result has the same shape as ``pre_relu`` but only the top-``k_aux``
    pre-ReLU values **among dead channels** are kept at each spatial
    position; everything else is zero.  Returning this tensor lets the
    model decode it through the same decoder and compute

        aux_loss = MSE(decode(dead_latent), (x - recon).detach())

    which is the standard Anthropic / Gao-et-al. AuxK formulation
    (encourages dead features to actually predict residual error, instead
    of merely growing in magnitude).

    A channel is "dead" iff its post-TopK output has been zero for at
    least ``dead_steps`` consecutive training steps (tracked by the
    encoder's ``_steps_since_active`` buffer).

    Returns ``None`` when no channels are dead.
    """
    B, C, H, W = pre_relu.shape
    dead_mask = steps_since_active >= dead_steps        # [C] bool
    n_dead = int(dead_mask.sum().item())
    if n_dead == 0:
        return None
    k_aux_eff = min(k_aux, n_dead, max(1, C - k))

    dead_idx = dead_mask.nonzero(as_tuple=True)[0]      # [n_dead]
    dead_pre = pre_relu[:, dead_idx, :, :]              # [B, n_dead, H, W]

    # Top-k_aux per spatial position among dead channels.
    flat = dead_pre.permute(0, 2, 3, 1).reshape(-1, n_dead)   # [N, n_dead]
    top_vals, top_pos = flat.topk(k_aux_eff, dim=1)
    keep = torch.zeros_like(flat)
    keep.scatter_(1, top_pos, top_vals.relu())
    dead_kept = keep.reshape(B, H, W, n_dead).permute(0, 3, 1, 2)  # [B, n_dead, H, W]

    dead_latent = pre_relu.new_zeros(B, C, H, W)
    dead_latent[:, dead_idx, :, :] = dead_kept
    return dead_latent


def auxk_loss(
    pre_relu: torch.Tensor,
    latent: torch.Tensor,
    k: int,
    k_aux: int,
    dead_threshold: float = 1e-3,
) -> torch.Tensor:
    """Deprecated.

    Old per-batch magnitude formulation kept only for backward import
    compatibility.  Returns 0; do **not** call from new code.  Use
    :func:`make_dead_latent` together with a decoder-based residual MSE
    in the model's forward pass instead.
    """
    return pre_relu.new_zeros(())


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
        dead_steps: int = 200,
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
        self.dead_steps = int(dead_steps)
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

        # ----- Track per-channel deadness over training time -----
        C = int(latent.shape[1])
        if not hasattr(self, "_steps_since_active") or self._steps_since_active.numel() != C:
            self.register_buffer(
                "_steps_since_active",
                torch.zeros(C, dtype=torch.long, device=latent.device),
                persistent=False,
            )
        if self.training:
            with torch.no_grad():
                any_active = latent.amax(dim=(0, 2, 3)) > self.dead_threshold
                self._steps_since_active[any_active] = 0
                self._steps_since_active[~any_active] += 1

        # ----- AuxK: emit a dead-only sparse latent (decoded by the model) -----
        if self.use_aux_loss and self.training:
            dead_latent = make_dead_latent(
                pre_relu, self._steps_since_active,
                k_aux=self.k_aux, k=k_eff, dead_steps=self.dead_steps,
            )
        else:
            dead_latent = None

        details["latent_shape"] = tuple(latent.shape)
        details["sparsity"] = pre_relu.new_zeros(())   # back-compat (model computes real aux_loss)
        details["dead_latent"] = dead_latent
        details["n_dead"] = int((self._steps_since_active >= self.dead_steps).sum().item())
        details["topk_k"] = k_eff

        return latent, details


__all__ = [
    "topk_activation",
    "auxk_loss",
    "make_dead_latent",
    "TopKSAEEncoder",
]
