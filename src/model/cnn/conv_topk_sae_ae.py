"""Strided-conv autoencoder with a channel-wise TopK bottleneck (no ResNet).

Architecture
------------
Encoder:  StridedConvEncoder  3 → 64 → 128 → 256 → 512 → 1024  (8×8 spatial at 256px)
Bottleneck: channel-wise TopK on [B, 1024, 8, 8]
Decoder:  StridedConvDecoder  1024 → 512 → 256 → 128 → 64 → 3

Dead-node tracking and AuxK revival follow the same convention as
TopKTaxonResNetStage: channels not selected by TopK for `dead_steps`
consecutive steps are revived via a pixel-space reconstruction-error objective.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_ae import StridedConvDecoder, StridedConvEncoder


class ConvTopKSAEAutoencoder(nn.Module):
    """Strided-conv AE with channel-wise TopK bottleneck.

    Parameters
    ----------
    topk_k:
        Number of channels kept per spatial position (or per batch-position if
        ``use_batch_topk=True``).
    k_aux:
        Number of dead channels used in AuxK loss. Defaults to ``topk_k``.
    dead_steps:
        A channel is "dead" after not being selected for this many steps.
    use_batch_topk:
        If True, select a global threshold so on average ``topk_k`` channels
        are active per position (batch-level TopK). Otherwise: per-position
        exact TopK.
    warmup_steps:
        Linearly anneal from all-channels-active to ``topk_k`` over this many
        training steps (0 = no warmup).
    output_activation:
        ``"none"`` / ``"tanh"`` / ``"sigmoid"``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        topk_k: int = 64,
        k_aux: Optional[int] = None,
        dead_steps: int = 2000,
        output_activation: str = "none",
        use_batch_topk: bool = True,
        warmup_steps: int = 0,
    ) -> None:
        super().__init__()

        self.encoder_net = StridedConvEncoder(in_channels=in_channels)
        latent_ch = self.encoder_net.out_channels  # 1024

        self.topk_k = int(topk_k)
        self.k_aux = int(k_aux) if k_aux is not None else self.topk_k
        self.dead_steps = int(dead_steps)
        self.use_batch_topk = bool(use_batch_topk)
        self.warmup_steps = int(warmup_steps)
        self.latent_channels = latent_ch

        self.decoder_net = StridedConvDecoder(
            in_channels=latent_ch,
            out_channels=in_channels,
            output_activation=output_activation,
        )

        # AuxK projection: dead latent features → pixel-space error (3-ch)
        self.auxk_proj = nn.Conv2d(latent_ch, in_channels, kernel_size=1, bias=True)

        # Dead-node tracking
        self.register_buffer("_steps_since_active", torch.zeros(latent_ch, dtype=torch.long))
        self.register_buffer("_warmup_step", torch.zeros(1, dtype=torch.long))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _topk_gate(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply channel-wise TopK and return (gated_z, bool_sel_mask [B,C,H,W])."""
        B, C, H, W = z.shape
        k_eff = min(self.topk_k, C)

        if self.training and self.warmup_steps > 0:
            step = int(self._warmup_step.item())
            progress = min(1.0, step / self.warmup_steps)
            k_current = max(k_eff, int(C * (1.0 - progress) + k_eff * progress))
            self._warmup_step += 1
        else:
            k_current = k_eff

        flat = z.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]

        if k_current < C:
            if self.use_batch_topk:
                flat_all = flat.reshape(-1)
                total_keep = k_current * flat.shape[0]
                if total_keep < flat_all.numel():
                    threshold = flat_all.topk(total_keep).values[-1]
                    mask_flat = (flat >= threshold).float()
                else:
                    mask_flat = torch.ones_like(flat)
            else:
                _, idx = flat.topk(k_current, dim=1)
                mask_flat = torch.zeros_like(flat)
                mask_flat.scatter_(1, idx, 1.0)
        else:
            mask_flat = torch.ones_like(flat)

        gated = (flat * mask_flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
        sel_mask = mask_flat.reshape(B, H, W, C).permute(0, 3, 1, 2).bool()
        return gated, sel_mask

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        z = self.encoder_net(x)
        z_gated, sel_mask = self._topk_gate(z)

        if self.training:
            any_active = sel_mask.amax(dim=(0, 2, 3))
            self._steps_since_active[any_active] = 0
            self._steps_since_active[~any_active] += 1

        dead_frac = (self._steps_since_active >= self.dead_steps).float().mean()
        recon = self.decoder_net(z_gated, output_size=x.shape[-2:])

        details: Dict = {}
        if return_details:
            details = {
                "dead_frac": dead_frac,
                "latent_shape": tuple(z.shape),
                "k": self.topk_k,
            }
        return recon, dead_frac, details

    def compute_auxk_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
    ) -> torch.Tensor:
        """AuxK revival loss: top-k_aux dead channels predict reconstruction error."""
        dead_mask = self._steps_since_active >= self.dead_steps
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return x.new_zeros(())

        z = self.encoder_net(x)                     # [B, C, H, W]
        dead_float = dead_mask.float().view(1, -1, 1, 1)
        dead_vals = z * dead_float

        B, C, H, W = dead_vals.shape
        flat = dead_vals.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        # Mask live channels to -inf so TopK only picks from dead channels
        flat_for_rank = flat.masked_fill(
            (~dead_mask).view(1, -1).expand_as(flat), float("-inf")
        )
        k_aux_eff = min(self.k_aux, n_dead)
        _, aux_idx = flat_for_rank.topk(k_aux_eff, dim=-1)
        aux_mask = torch.zeros_like(flat)
        aux_mask.scatter_(1, aux_idx, 1.0)
        aux_mask = aux_mask.reshape(B, H, W, C).permute(0, 3, 1, 2)

        error = (x - recon).detach()
        dead_signal = self.auxk_proj(dead_vals * aux_mask)
        if error.shape[-2:] != dead_signal.shape[-2:]:
            error = F.interpolate(
                error, size=dead_signal.shape[-2:], mode="bilinear", align_corners=False
            )
        return F.mse_loss(dead_signal, error)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z = self.encoder_net(x)
        z_gated, _ = self._topk_gate(z)
        return z_gated, {}

    def decode(
        self, z: torch.Tensor, output_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        return self.decoder_net(z, output_size=output_size)
