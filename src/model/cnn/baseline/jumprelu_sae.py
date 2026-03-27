"""JumpReLU Sparse Convolutional Autoencoder wrapper.

Composes:
- :class:`~src.model.jumprelu_sae_encoder.JumpReLUSAEEncoder`: JumpReLU encoder.
- :class:`~src.model.decoder.TaxonResNetDecoder`: mirror decoder (reused).

The forward signature mirrors :class:`SparseConvAutoencoder`::

    recon, sparsity = model(x)

where ``sparsity`` is ``(L̂₀ − target_l0)²`` — the raw sparsity penalty
(multiply by ``sparsity_weight`` in the training loss).

For full debugging info::

    recon, sparsity, details = model(x, return_details=True)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .jumprelu_sae_encoder import JumpReLUSAEEncoder
from ..taxon.decoder import TaxonResNetDecoder


class JumpReLUSparseConvAutoencoder(nn.Module):
    """JumpReLU Sparse Convolutional Autoencoder.

    Same architecture as the other SAE variants but with a *learned
    per-channel threshold* (Rajamanoharan et al. 2024) instead of an L1
    penalty, hard TopK, or gating.  Sparsity is encouraged by penalising
    the mean active-feature count (L̂₀) deviating from ``target_l0``:

        L_sparse = (L̂₀ − target_l0)²

    Scale via an external ``sparsity_weight`` in the training loss.

    Args:
        in_channels: Input image channels (default 3).
        resnet_variant: ResNet preset for stage block counts.
        stage_channels: Output channel count per encoder stage.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit block counts (overrides ``resnet_variant``).
        target_l0: Desired mean active channels per spatial position.
        bandwidth: STE window width for threshold-gradient approximation.
        theta_init: Initial threshold value (stored as ``log(theta_init)``).
        kernel_size: Conv kernel size throughout.
        use_stem: Include 7×7 stem conv.
        stem_channels: Stem output channels.
        stem_stride: Stem conv stride.
        use_stem_maxpool: Append 3×3 max-pool after stem.
        output_activation: Output activation (``"none"`` / ``"tanh"`` /
            ``"sigmoid"``).
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        target_l0: float = 64.0,
        bandwidth: float = 0.001,
        theta_init: float = 0.1,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
    ) -> None:
        super().__init__()

        self.output_activation = output_activation.lower()

        self.encoder = JumpReLUSAEEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_channels=stage_channels,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            target_l0=target_l0,
            bandwidth=bandwidth,
            theta_init=theta_init,
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
    # Public encode / decode API (mirrors SparseConvAutoencoder)
    # -----------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode an image batch to a JumpReLU-sparse latent feature map."""
        return self.encoder(x, return_details=return_details)

    def decode(
        self,
        z: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Decode a latent feature map to image space."""
        return self.decoder(z, output_size=output_size, return_details=return_details)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ):
        """Full autoencoder pass.

        Args:
            x: Input image tensor ``[B, C, H, W]``.
            return_details: Return shape-trace details dict.

        Returns:
            ``(recon, sparsity)`` normally, or
            ``(recon, sparsity, details)`` when ``return_details=True``.

            ``sparsity`` is the raw ``(L̂₀ − target_l0)²`` scalar.
            Multiply by ``sparsity_weight`` in the training loss.
        """
        latent, enc_details = self.encode(x, return_details=return_details)
        recon, dec_details = self.decode(
            latent, output_size=x.shape[-2:], return_details=return_details
        )
        recon = self._apply_output_activation(recon)

        sparsity = enc_details["sparsity"]

        if not return_details:
            return recon, sparsity

        details = {
            "input_shape":  tuple(x.shape),
            "latent_shape": tuple(latent.shape),
            "recon_shape":  tuple(recon.shape),
            "encoder":      enc_details,
            "decoder":      dec_details,
        }
        return recon, sparsity, details


__all__ = ["JumpReLUSparseConvAutoencoder"]
