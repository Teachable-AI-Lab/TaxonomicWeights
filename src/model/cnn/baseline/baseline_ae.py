"""Baseline Convolutional Autoencoder — no sparsity, no taxonomy.

A plain ResNet-style encoder + mirror decoder.  Serves as a reconstruction
quality baseline against TaxonAutoencoder and SparseConvAutoencoder.

The encoder is structurally identical to :class:`ConvSAEEncoder` but with:
- No sparsity penalty (returns ``torch.tensor(0.0)`` as a placeholder so
  the training loop interface stays the same).
- Plain identity (no clamping) latent activation — the raw encoder output
  is used as the latent representation.

Forward signature mirrors :class:`SparseConvAutoencoder`:

    ``recon, zero = model(x)``

where ``zero`` is always 0.0 so the training script can treat it identically
to the SAE without any conditional logic.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .sae_encoder import ConvSAEEncoder
from ..taxon.decoder import TaxonResNetDecoder


class BaselineConvAutoencoder(nn.Module):
    """Plain convolutional autoencoder — no sparsity, no taxonomy.

    Shares the same stem + residual-stage encoder and mirror decoder as
    :class:`SparseConvAutoencoder`.  The sparsity regulariser is simply
    removed so that the model trains purely on reconstruction loss.

    Args:
        in_channels: Number of input image channels (default 3).
        resnet_variant: ResNet preset for stage block counts.
        stage_channels: Output channel count per encoder stage.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit block counts (overrides ``resnet_variant``).
        kernel_size: Kernel size used throughout encoder and decoder.
        use_stem: Include the 7×7 conv + BN + ReLU stem.
        stem_channels: Output channels of the stem conv.
        stem_stride: Spatial stride of the stem conv.
        use_stem_maxpool: Append a 3×3 max-pool (stride 2) after the stem.
        output_activation: Final activation applied to the reconstruction
            (``"none"`` / ``"tanh"`` / ``"sigmoid"``).
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
    ) -> None:
        super().__init__()

        self.output_activation = output_activation.lower()

        # Reuse ConvSAEEncoder with an identity latent activation and a
        # negligible sparsity type (l1 — result is discarded).
        self.encoder = ConvSAEEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_channels=stage_channels,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            sparsity_type="l1",        # computed but thrown away
            sparsity_target=0.0,
            latent_activation="none",  # raw latent, no squashing
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

        # Pre-register zero so it lives on the right device automatically.
        self.register_buffer("_zero", torch.zeros(1))

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        return x

    # -----------------------------------------------------------------------
    # Encode / decode
    # -----------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """Encode image batch to latent feature map.

        Returns ``(z, details)`` where ``z`` is the raw (unactivated) latent
        and ``details`` is an empty dict (for interface compatibility).
        """
        z, details = self.encoder(x, return_details=return_details)
        return z, details

    def decode(
        self,
        z: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
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

        Returns:
            ``(recon, zero)`` where ``zero`` is always 0.0 (no regulariser).
            When ``return_details=True``: ``(recon, zero, details)``.
        """
        z, enc_details = self.encoder(x, return_details=return_details)
        recon_raw, dec_details = self.decoder(
            z, output_size=x.shape[-2:], return_details=return_details
        )
        recon = self._apply_output_activation(recon_raw)

        if return_details:
            details = {**enc_details, **dec_details}
            return recon, self._zero.squeeze(), details
        return recon, self._zero.squeeze()
