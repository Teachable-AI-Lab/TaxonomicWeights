"""Sparse Convolutional Autoencoder (SAE) wrapper.

Composes:
- :class:`~src.model.sae_encoder.ConvSAEEncoder`: ResNet-style encoder that
  produces a sparsity-regularised latent feature map.
- :class:`~src.model.decoder.TaxonResNetDecoder`: mirror decoder (reused
  from the taxon autoencoder — no taxonomy-specific logic present).

The forward signature intentionally mirrors :class:`TaxonAutoencoder`:

    ``recon, sparsity = model(x)``

where ``sparsity`` is the *raw* (unweighted) regularisation term.  Scale it
from the training script with an external coefficient, just as
``dkl_weight * dkl`` is applied for the taxon model.

For full debugging info::

    recon, sparsity, details = model(x, return_details=True)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sae_encoder import ConvSAEEncoder
from .decoder import TaxonResNetDecoder


class SparseConvAutoencoder(nn.Module):
    """Sparse Convolutional Autoencoder for images.

    Architecture deliberately mirrors :class:`TaxonAutoencoder` — same stem,
    same residual stage layout, same decoder — with taxonomy routing replaced
    by an L1 or KL-divergence sparsity penalty on the latent feature map.

    Args:
        in_channels: Number of input image channels (default 3).
        resnet_variant: ResNet preset for stage block counts.
        stage_channels: Output channel count per encoder stage.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit block counts (overrides ``resnet_variant``).
        sparsity_type: ``"l1"`` or ``"kl"``.
        sparsity_target: Target sparsity ``ρ`` for KL mode.
        latent_activation: ``"relu"`` (default) or ``"sigmoid"``.
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
        sparsity_type: str = "l1",
        sparsity_target: float = 0.05,
        latent_activation: str = "relu",
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
    ) -> None:
        super().__init__()

        self.output_activation = output_activation.lower()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = ConvSAEEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_channels=stage_channels,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            sparsity_type=sparsity_type,
            sparsity_target=sparsity_target,
            latent_activation=latent_activation,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        # Re-use TaxonResNetDecoder unchanged — it is a pure convolutional
        # mirror decoder with residual upsampling blocks.
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
    # Public encode / decode (mirrors TaxonAutoencoder API)
    # -----------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode an image batch to a sparse latent feature map."""
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
            return_details: If ``True``, return shape-trace details dict.

        Returns:
            ``(recon, sparsity)`` normally, or
            ``(recon, sparsity, details)`` when ``return_details=True``.

            ``sparsity`` is the raw (unweighted) regularisation scalar —
            multiply by your ``sparsity_weight`` in the training loop.
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
            "input_shape": tuple(x.shape),
            "latent_shape": tuple(latent.shape),
            "recon_shape": tuple(recon.shape),
            "encoder": enc_details,
            "decoder": dec_details,
        }
        return recon, sparsity, details


__all__ = ["SparseConvAutoencoder"]
