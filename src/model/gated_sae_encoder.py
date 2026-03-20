"""Gated Convolutional SAE encoder.

Implements the **Gated SAE** architecture (Rajamanoharan et al., Google
DeepMind 2024 — "Improving Dictionary Learning with Gated Sparse
Autoencoders").

Core idea: decouple *which* features fire (a gating decision) from *how
strongly* they fire (a magnitude decision).  This breaks the shrinkage bias
of L1 SAEs, where large activations are penalised proportionally and the
model must choose between reconstruction quality and sparsity.

Architecture (per spatial location for the convolutional adaptation):
- Shared backbone: same ResNet-style residual stages as
  :class:`~src.model.sae_encoder.ConvSAEEncoder`.
- Two parallel 1×1 projection heads on the final stage output ``h``:
    - ``gate_proj``:  ``h → W_g h + b_g → ReLU``   — gate activations π
    - ``mag_proj``:   ``h → W_m h + b_m → ReLU``    — magnitude activations μ
- Sparse latent: ``f = π × μ``  (element-wise)
- Sparsity loss: ``L1(π)`` — penalty on the *gate* only, not the magnitude.
  This means large-magnitude features are not penalised for being active.

The L1 penalty is applied to the *gate pre-activations* (before ReLU) so
that the gate can grow freely without being immediately killed off.

A Straight-Through Estimator (STE) variant is optionally available
(``use_gate_ste=True``): during the forward pass the gate is binarised as
``g_hard = (π > 0).float()`` but gradients flow through ``π`` as if it were
the continuous gate.  This gives a cleaner binary interpretation at the cost
of a slightly noisier gradient signal.

Interface is identical to :class:`ConvSAEEncoder`:

    ``latent, info = encoder(x)``

where ``info["sparsity"]`` is the raw L1(π) value.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResidualConvBlock, resolve_resnet_stage_blocks
from .sae_encoder import ConvSAEStage


class GatedSAEEncoder(nn.Module):
    """ResNet-style encoder for the Gated Sparse Convolutional Autoencoder.

    Backbone is identical to :class:`~src.model.sae_encoder.ConvSAEEncoder`.
    The final latent is produced by two parallel 1×1 conv heads:

    - Gate head  → ReLU → π  (sparsity L1 penalty applied here)
    - Magnitude head → ReLU → μ
    - Latent  = π × μ

    Args:
        in_channels: Input image channels.
        resnet_variant: ResNet preset for stage block counts.
        stage_channels: Output channel count per stage.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit per-stage block counts.
        use_gate_ste: If ``True``, binarise the gate with a
            Straight-Through Estimator during training.
        kernel_size: Conv kernel size throughout.
        use_stem: Include the 7×7 conv + BN + ReLU stem.
        stem_channels: Output channels of the stem conv.
        stem_stride: Spatial stride of the stem conv.
        use_stem_maxpool: Append a 3×3 max-pool after the stem.
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        use_gate_ste: bool = False,
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
        self.use_gate_ste = bool(use_gate_ste)
        self.sparsity_type = "gated"   # for compatibility with analysis / compare scripts
        self.latent_activation = "gated"
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

        # ----- Residual stages (same as ConvSAEEncoder) --------------------
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

        # ----- Gating heads (parallel 1×1 projections) ---------------------
        # Both project from final_channels → final_channels.
        # We use bias=True here so the gate and magnitude can be independently
        # biased away from zero.
        self.gate_proj = nn.Conv2d(self.final_channels, self.final_channels, kernel_size=1, bias=True)
        self.mag_proj  = nn.Conv2d(self.final_channels, self.final_channels, kernel_size=1, bias=True)

        # Initialise gate_proj bias slightly negative to start near-zero
        # (encourages initial sparsity and avoids all features being active).
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -0.1)
        nn.init.zeros_(self.mag_proj.weight)
        nn.init.zeros_(self.mag_proj.bias)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode an image batch to a gated-sparse latent feature map.

        Args:
            x: Input image tensor ``[B, C, H, W]``.
            return_details: If ``True``, populate ``shape_trace`` in details.

        Returns:
            ``(latent, info)`` where ``info["sparsity"]`` is the raw L1 gate
            penalty (L1(π), before the sparsity weight is applied).
        """
        details: Dict[str, object] = {"input_shape": tuple(x.shape), "shape_trace": []}

        x = self.stem(x)
        if return_details:
            details["shape_trace"].append(("stem", tuple(x.shape)))

        for idx, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if return_details:
                details["shape_trace"].append((f"stage{idx}", tuple(x.shape)))

        # Parallel gate and magnitude projections
        gate_pre = self.gate_proj(x)   # [B, C, H, W]
        mag_pre  = self.mag_proj(x)    # [B, C, H, W]

        # Gate: ReLU activation (optionally binarised with STE)
        gate = F.relu(gate_pre)
        if self.use_gate_ste and self.training:
            # Straight-Through Estimator: use binary gate on forward pass
            # but let gradient flow through the continuous gate (ReLU).
            gate_binary = (gate > 0).float()
            gate = gate_binary - gate.detach() + gate

        # Magnitude: ReLU activation
        mag = F.relu(mag_pre)

        # Gated latent: sparse because gate is often zero
        latent = gate * mag

        # Sparsity penalty: L1 on the gate pre-activation (before ReLU)
        # Applying L1 to gate_pre (not gate) prevents magnitude from being
        # penalised and matches the Rajamanoharan et al. formulation.
        sparsity = gate_pre.abs().mean()

        details["latent_shape"] = tuple(latent.shape)
        details["sparsity"]     = sparsity
        details["gate"]         = gate
        details["magnitude"]    = mag
        details["gate_mean"]    = float(gate.mean().detach().cpu())
        details["mag_mean"]     = float(mag.mean().detach().cpu())

        return latent, details


__all__ = ["GatedSAEEncoder"]
