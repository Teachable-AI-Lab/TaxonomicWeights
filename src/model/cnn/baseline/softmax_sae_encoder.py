"""Convolutional encoder with flat softmax gating and DKL/entropy regularizers.

No taxonomy constraint — all latent channels compete equally via a single
flat softmax over the channel dimension at each spatial position.

This provides the same DKL (coverage) and entropy (sparsity) regularization
as the Taxon SAE, but without any hierarchical tree structure, making it a
direct ablation to test whether the taxonomic inductive bias adds value.

Forward returns::

    (latent, {"dkl": dkl, "entropy": entropy})

where:

- ``latent``:  ``[B, C, H, W]`` — feature map weighted by softmax probability
  (same gating as ``TaxonResNetStage`` but flat, no tree)
- ``dkl``:     ``KL(marginal || uniform)``  — coverage penalty (lower → all
  features used equally)
- ``entropy``: ``-H(prob)``  — negative entropy scalar; adding
  ``entropy_weight * entropy`` to the loss encourages concentrated, sparse
  routing (lower entropy = more peaked softmax = sparser latent)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..taxon.encoder import resolve_resnet_stage_blocks
from .sae_encoder import ConvSAEStage


class SoftmaxSAEEncoder(nn.Module):
    """ResNet-style encoder with flat-softmax sparsity gating.

    Same stem + stage layout as :class:`ConvSAEEncoder`.  After the final
    residual stage, a *flat* softmax over the channel dimension converts raw
    feature maps into per-position occupancy probabilities, and the output is

    .. math::

        \\text{latent}_{b,c,h,w} = x_{b,c,h,w} \\cdot \\text{softmax}(x / \\tau)_{b,c,h,w}

    which is the same gating formula used by :class:`TaxonResNetStage` on
    each binary split — applied here globally without any tree structure.

    Args:
        in_channels: Input image channels.
        resnet_variant: ResNet preset (``"18"``, ``"34"``, …) for block counts.
        stage_channels: Per-stage output channel count.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit block counts (overrides ``resnet_variant``).
        temperature: Softmax temperature τ; lower → sharper / more sparse.
        kernel_size: Conv kernel size throughout the network.
        use_stem: Include 7×7 conv + BN + ReLU stem.
        stem_channels: Stem output channels.
        stem_stride: Spatial stride of the stem conv.
        use_stem_maxpool: Append 3×3 max-pool (stride 2) after the stem.
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        temperature: float = 1.0,
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
        self.temperature = float(temperature)
        self.use_stem = bool(use_stem)

        # ----- Stem ---------------------------------------------------------
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

        # ----- Stages -------------------------------------------------------
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
    # Regularisation helpers (mirror TaxonResNetStage formulae, flat version)
    # -----------------------------------------------------------------------

    def _regularization_terms(
        self,
        prob: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute negative entropy and coverage DKL from flat softmax probs.

        Args:
            prob: Softmax probability tensor ``[B, C, H, W]``.

        Returns:
            ``(entropy, dkl)`` where:

            - ``entropy`` = ``-H(prob)`` mean over B, H, W — adding a positive
              weight to the loss minimizes entropy (encourages sparsity).
            - ``dkl`` = ``KL(marginal || uniform)`` — coverage penalty;
              minimizing it encourages all features to be used equally.
        """
        logp = prob.clamp_min(eps).log()

        # Local negative entropy (per spatial position, averaged over batch+space)
        entropy = -(prob * logp).sum(dim=1).mean()   # scalar

        # Coverage KL: marginal distribution should be uniform
        marginal = prob.mean(dim=(0, 2, 3))           # [C]
        marginal = marginal / marginal.sum().clamp_min(eps)
        uniform_logp = -math.log(prob.shape[1])
        dkl = (marginal * (marginal.clamp_min(eps).log() - uniform_logp)).sum()

        return entropy, dkl

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode an image batch to a softmax-gated sparse latent feature map.

        Args:
            x: Input image tensor ``[B, C, H, W]``.
            return_details: Populate ``shape_trace`` in the info dict.

        Returns:
            ``(latent, info)`` where ``info`` contains ``"dkl"`` and
            ``"entropy"`` scalars (same keys as :class:`TaxonResNetEncoder`).
        """
        details: Dict[str, object] = {"input_shape": tuple(x.shape), "shape_trace": []}

        x = self.stem(x)
        if return_details:
            details["shape_trace"].append(("stem", tuple(x.shape)))  # type: ignore[attr-defined]

        for idx, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if return_details:
                details["shape_trace"].append((f"stage{idx}", tuple(x.shape)))  # type: ignore[attr-defined]

        # Flat softmax gating — same output formula as TaxonResNetStage per depth
        prob = torch.softmax(x / self.temperature, dim=1)   # [B, C, H, W]
        latent = x * prob                                     # soft gating

        entropy, dkl = self._regularization_terms(prob)

        details["latent_shape"] = tuple(latent.shape)
        details["dkl"] = dkl
        details["entropy"] = entropy

        return latent, details


__all__ = ["SoftmaxSAEEncoder"]
