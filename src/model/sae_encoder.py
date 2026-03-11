"""Convolutional SAE encoder mimicking the TaxonResNetEncoder.

Replaces taxonomy-constrained hierarchical routing with a simple sparsity
penalty on the final latent feature map.  Two sparsity modes are supported:

- ``"l1"``:  ``sparsity = mean(|h|)``  (ReLU latent activation recommended)
- ``"kl"``:  ``sparsity = Σ_j KL(ρ || ρ̂_j)``  (sigmoid latent activation)

The raw (unweighted) sparsity term is returned alongside the latent so the
caller can scale it by an external coefficient — exactly the same convention
used by :class:`~src.model.taxon_ae.TaxonAutoencoder` for ``dkl`` / ``entropy``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResidualConvBlock, resolve_resnet_stage_blocks


# ---------------------------------------------------------------------------
# Stage (plain residual conv, no taxonomy)
# ---------------------------------------------------------------------------

class ConvSAEStage(nn.Module):
    """Plain residual CNN stage — N consecutive :class:`ResidualConvBlock` blocks.

    This is structurally identical to :class:`~src.model.encoder.TaxonResNetStage`
    minus the taxonomy routing.  The first block applies the spatial stride;
    subsequent blocks keep stride 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        for idx in range(n_blocks):
            block_in = in_channels if idx == 0 else out_channels
            block_stride = stride if idx == 0 else 1
            blocks.append(
                ResidualConvBlock(
                    in_channels=block_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=block_stride,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ConvSAEEncoder(nn.Module):
    """ResNet-style encoder for the Sparse Convolutional Autoencoder.

    Stem and stage layout mirror :class:`~src.model.encoder.TaxonResNetEncoder`
    (same 7×7 stem with optional max-pool, same residual block structure).
    The only difference is that taxonomy routing is replaced by a
    sparsity-inducing activation and regularizer on the final feature map.

    Args:
        in_channels: Number of input image channels.
        resnet_variant: ResNet preset (``"18"``, ``"34"``, …) for block counts.
        stage_channels: Output channel count for each encoder stage.
        stage_strides: Spatial stride for the first block of each stage.
        stage_blocks: Explicit per-stage block counts (overrides
            ``resnet_variant`` if provided).
        sparsity_type: ``"l1"`` or ``"kl"``.
        sparsity_target: Target sparsity level ``ρ`` used by KL mode (0–1).
        latent_activation: Activation applied to the final feature map before
            sparsity is measured — ``"relu"`` (default) or ``"sigmoid"``.
        kernel_size: Conv kernel size throughout the network.
        use_stem: Include the 7×7 conv stem (same as TaxonResNetEncoder).
        stem_channels: Number of channels produced by the stem conv.
        stem_stride: Spatial stride of the stem conv.
        use_stem_maxpool: Append a 3×3 max-pool (stride 2) after the stem.
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        sparsity_type: str = "l1",
        sparsity_target: float = 0.05,
        latent_activation: str = "relu",
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
        self.sparsity_type = sparsity_type.lower()
        self.sparsity_target = float(sparsity_target)
        self.latent_activation = latent_activation.lower()
        self.use_stem = bool(use_stem)

        if self.sparsity_type not in {"l1", "kl"}:
            raise ValueError(f"sparsity_type must be 'l1' or 'kl', got '{sparsity_type}'.")
        if self.latent_activation not in {"relu", "sigmoid", "none", "identity"}:
            raise ValueError(
                f"latent_activation must be 'relu', 'sigmoid', or 'none', got '{latent_activation}'."
            )

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
        # stage_input_channels[i] = number of channels fed into stage i
        # (used by TaxonResNetDecoder to wire up the reverse pathway).
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
    # Internal helpers
    # -----------------------------------------------------------------------

    def _apply_latent_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_activation == "relu":
            return F.relu(x)
        if self.latent_activation == "sigmoid":
            return torch.sigmoid(x)
        return x   # "none" / "identity" — pass through unchanged

    def _compute_sparsity(self, h: torch.Tensor) -> torch.Tensor:
        """Return the *raw* (unweighted) sparsity regularization term.

        L1 mode:
            ``mean(|h|)``

        KL mode (SAE.md formulation):
            ``Σ_j KL(ρ || ρ̂_j)``  where ``ρ̂_j = mean_j(h)`` over batch &
            spatial dims.  Requires ``h`` to be in ``(0, 1)`` — use sigmoid
            latent activation.
        """
        if self.sparsity_type == "l1":
            return h.abs().mean()

        # KL-divergence sparsity
        rho_hat = h.mean(dim=(0, 2, 3)).clamp(1e-6, 1.0 - 1e-6)   # [C]
        rho = self.sparsity_target
        kl = (
            rho * math.log(rho) - rho * rho_hat.log()
            + (1.0 - rho) * math.log(1.0 - rho)
            - (1.0 - rho) * (1.0 - rho_hat).log()
        )
        return kl.sum()

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode an image batch to a sparse latent feature map.

        Args:
            x: Input image tensor ``[B, C, H, W]``.
            return_details: If ``True``, populate the ``shape_trace`` entry.

        Returns:
            ``(latent, info)`` where ``latent`` is the activated feature map
            (shape ``[B, final_channels, H', W']``) and ``info["sparsity"]``
            is the raw (unweighted) sparsity scalar.
        """
        details: Dict[str, object] = {"input_shape": tuple(x.shape), "shape_trace": []}

        x = self.stem(x)
        if return_details:
            details["shape_trace"].append(("stem", tuple(x.shape)))  # type: ignore[attr-defined]

        for idx, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if return_details:
                details["shape_trace"].append((f"stage{idx}", tuple(x.shape)))  # type: ignore[attr-defined]

        latent = self._apply_latent_activation(x)
        sparsity = self._compute_sparsity(latent)

        details["latent_shape"] = tuple(latent.shape)
        details["sparsity"] = sparsity

        return latent, details


__all__ = [
    "ConvSAEStage",
    "ConvSAEEncoder",
]
