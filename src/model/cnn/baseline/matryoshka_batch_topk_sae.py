"""Matryoshka Batch-TopK Sparse Convolutional Autoencoder.

Composes:
- :class:`~src.model.matryoshka_batch_topk_sae_encoder.MatryoshkaBatchTopKSAEEncoder`:
  batch-global TopK encoder with Matryoshka nesting.
- :class:`~src.model.decoder.TaxonResNetDecoder`: mirror decoder (reused
  unchanged across all sparsity levels).

Two operating modes
-------------------
**Inference** (``model(x)`` or ``model.encode`` / ``model.decode``)
    Uses the largest k in ``k_values``.  Forward returns ``(recon, aux_loss)``
    — identical signature to :class:`TopKSparseConvAutoencoder`.

**Training** (``model.forward_matryoshka(x)``)
    Returns one reconstruction per k-level plus the AuxK loss.  The training
    script computes the weighted sum
    ``Σ_i loss_weights[i] * MSE(recon_i, x) + sparsity_weight * aux_loss``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matryoshka_batch_topk_sae_encoder import MatryoshkaBatchTopKSAEEncoder
from ..taxon.decoder import TaxonResNetDecoder


class MatryoshkaBatchTopKSparseConvAutoencoder(nn.Module):
    """Matryoshka Batch-TopK Sparse Convolutional Autoencoder.

    A single encoder + single decoder are shared across all sparsity levels.
    During training, the decoder is applied once per k-level (producing nested
    reconstructions from coarse to fine).  The Matryoshka loss encourages each
    level to produce a useful reconstruction on its own.

    Args:
        in_channels: Input image channels (default 3).
        resnet_variant: ResNet preset for stage block counts.
        stage_channels: Output channel count per encoder stage.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit block counts (overrides ``resnet_variant``).
        k_values: Sequence of active-channel counts (sorted descending).
            E.g. ``[64, 32, 16]``.  The largest value is used for inference.
        k_aux: AuxK features for the auxiliary dead-neuron loss.
        use_aux_loss: Enable the AuxK auxiliary loss during training.
        dead_threshold: Activation threshold for dead-neuron detection.
        kernel_size: Conv kernel size throughout.
        use_stem: Include 7×7 stem conv.
        stem_channels: Stem output channels.
        stem_stride: Stem conv stride.
        use_stem_maxpool: Append 3×3 max-pool after stem.
        output_activation: Activation on reconstruction output
            (``"none"`` / ``"tanh"`` / ``"sigmoid"``).
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
        dead_steps: int = 200,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
    ) -> None:
        super().__init__()

        self.output_activation = output_activation.lower()

        self.encoder = MatryoshkaBatchTopKSAEEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_channels=stage_channels,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            k_values=k_values,
            k_aux=k_aux,
            use_aux_loss=use_aux_loss,
            dead_threshold=dead_threshold,
            dead_steps=dead_steps,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
        )

        self.decoder = TaxonResNetDecoder(
            latent_channels=self.encoder.final_channels,
            stage_input_channels=self.encoder.stage_input_channels,
            stage_blocks=self.encoder.stage_blocks,
            stage_strides=self.encoder.stage_strides,
            out_channels=in_channels,
            use_stem=use_stem,
            stem_total_stride=self.encoder.stem_total_stride,
            kernel_size=kernel_size,
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation in {"identity", "none", "linear"}:
            return x
        raise ValueError(
            f"Unsupported output_activation='{self.output_activation}'. "
            "Use 'none'/'identity'/'linear', 'tanh', or 'sigmoid'."
        )

    # -----------------------------------------------------------------------
    # Public encode / decode API
    # -----------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode to the largest-k sparse latent (for inference / analysis)."""
        return self.encoder(x, return_details=return_details)

    def decode(
        self,
        z: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Decode a sparse latent to image space."""
        return self.decoder(z, output_size=output_size, return_details=return_details)

    # -----------------------------------------------------------------------
    # Standard forward (inference — largest k)
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ):
        """Full autoencoder pass using the largest k.

        Args:
            x: Input image tensor ``[B, C, H, W]``.
            return_details: Return shape-trace details dict.

        Returns:
            ``(recon, aux_loss)`` normally, or
            ``(recon, aux_loss, details)`` when ``return_details=True``.
        """
        latent, enc_details = self.encode(x, return_details=return_details)
        recon, dec_details = self.decode(
            latent, output_size=x.shape[-2:], return_details=return_details
        )
        recon = self._apply_output_activation(recon)
        dead_latent = enc_details.get("dead_latent")
        if dead_latent is not None and self.training:
            dead_recon, _ = self.decoder(dead_latent, output_size=x.shape[-2:])
            error = (x - recon).detach()
            aux_loss = F.mse_loss(dead_recon, error)
        else:
            aux_loss = enc_details["sparsity"]

        if not return_details:
            return recon, aux_loss

        details = {
            "input_shape":  tuple(x.shape),
            "latent_shape": tuple(latent.shape),
            "recon_shape":  tuple(recon.shape),
            "encoder":      enc_details,
            "decoder":      dec_details,
        }
        return recon, aux_loss, details

    # -----------------------------------------------------------------------
    # Matryoshka forward (training — all k levels)
    # -----------------------------------------------------------------------

    def forward_matryoshka(
        self,
        x: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Full autoencoder pass at every Matryoshka k-level.

        The decoder is applied once per level, producing one reconstruction
        per k-value.  The training loss is the weighted sum of per-level MSEs.

        Args:
            x: Input image tensor ``[B, C, H, W]``.

        Returns:
            ``(recons, aux_loss)`` where ``recons[i]`` is the reconstruction
            at ``self.encoder.k_values[i]`` (descending) and ``aux_loss``
            is the raw AuxK scalar from the largest-k latent.
        """
        latents, info = self.encoder.forward_matryoshka(x)
        H, W = x.shape[-2:]

        recons: List[torch.Tensor] = []
        for z in latents:
            recon, _ = self.decoder(z, output_size=(H, W))
            recon = self._apply_output_activation(recon)
            recons.append(recon)

        # Decoder-based AuxK against the largest-k residual.
        dead_latent = info.get("dead_latent")
        if dead_latent is not None and self.training:
            dead_recon, _ = self.decoder(dead_latent, output_size=(H, W))
            error = (x - recons[0]).detach()
            aux_loss = F.mse_loss(dead_recon, error)
        else:
            aux_loss = info["sparsity"]

        return recons, aux_loss


__all__ = ["MatryoshkaBatchTopKSparseConvAutoencoder"]
