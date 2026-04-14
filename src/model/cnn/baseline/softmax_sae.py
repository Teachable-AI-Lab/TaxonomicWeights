"""Flat-Softmax Sparse Convolutional Autoencoder (SoftmaxSAE).

Composes:
- :class:`SoftmaxSAEEncoder`: ResNet-style encoder that applies a flat
  softmax gating over the channel dimension — same DKL + entropy regularizers
  as the Taxonomic AE, but *without* any tree structure.
- :class:`~src.model.taxon.decoder.TaxonResNetDecoder`: mirror decoder reused
  from the taxon autoencoder (no taxonomy-specific logic present).

This model is the direct ablation test for the taxonomic inductive bias:
it uses the identical softmax-gating objective (`feature × softmax(feature/τ)`)
and the same DKL / entropy penalties, but node organisation does not influence
activations.

Forward signature intentionally mirrors :class:`~src.model.taxon.taxon_ae.TaxonAutoencoder`::

    recon, dkl, entropy = model(x)

so that training and comparison scripts can treat both models identically.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .softmax_sae_encoder import SoftmaxSAEEncoder
from ..taxon.decoder import TaxonResNetDecoder


class SoftmaxSparseConvAutoencoder(nn.Module):
    """Flat-softmax sparse convolutional autoencoder.

    Architecture mirrors :class:`~src.model.taxon.taxon_ae.TaxonAutoencoder`
    — same stem, same residual stage layout, same decoder — with taxonomy
    routing replaced by a flat softmax over the latent channel dimension.

    Args:
        in_channels: Number of input image channels (default 3).
        resnet_variant: ResNet preset for stage block counts.
        stage_channels: Output channel count per encoder stage.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit block counts (overrides ``resnet_variant``).
        temperature: Softmax temperature τ (default 1.0). Lower → sparser.
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
        temperature: float = 1.0,
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
        self.encoder = SoftmaxSAEEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_channels=stage_channels,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            temperature=temperature,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
        )

        # ── Decoder ──────────────────────────────────────────────────────────
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
        """Encode an image batch to a softmax-gated sparse latent feature map."""
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
            ``(recon, dkl, entropy)`` normally, or
            ``(recon, dkl, entropy, details)`` when ``return_details=True``.

            - ``dkl``: raw coverage KL (multiply by ``dkl_weight`` in training).
            - ``entropy``: raw negative entropy (multiply by ``entropy_weight``
              in training; positive weight → sparsity pressure).

            The signature is identical to
            :class:`~src.model.taxon.taxon_ae.TaxonAutoencoder` so training
            and comparison scripts can treat both models identically.
        """
        latent, enc_details = self.encode(x, return_details=return_details)
        recon, dec_details = self.decode(
            latent, output_size=x.shape[-2:], return_details=return_details
        )
        recon = self._apply_output_activation(recon)

        dkl     = enc_details["dkl"]
        entropy = enc_details["entropy"]

        if not return_details:
            return recon, dkl, entropy

        details = {
            "input_shape":  tuple(x.shape),
            "latent_shape": tuple(latent.shape),
            "recon_shape":  tuple(recon.shape),
            "encoder":      enc_details,
            "decoder":      dec_details,
        }
        return recon, dkl, entropy, details


__all__ = ["SoftmaxSparseConvAutoencoder"]
