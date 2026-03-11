"""Taxon ResNet autoencoder wrapper.

This module composes:
- ``TaxonResNetEncoder``: ResNet-style encoder with taxonomy constraints.
- ``TaxonResNetDecoder``: mirror decoder.

Forward output:
- ``reconstruction``
- ``dkl``: aggregated coverage KL regularizer
- ``entropy``: local path entropy regularizer
- optional ``details`` dictionary for debugging/visualization
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import TaxonResNetDecoder
from .encoder import TaxonResNetEncoder


class TaxonAutoencoder(nn.Module):
    """ResNet-style taxonomic autoencoder.

    The autoencoder can optionally use the attention-enhanced encoder variant by
    specifying ``attn_heads`` &gt; 0.  This keeps the external API the same so
    that existing scripts and config files continue to work; old configs simply
    omit ``attn_heads`` or set it to ``0``.  Analysis routines that reconstruct
    the model from a config will also automatically pick up the correct
    encoder class.
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_taxonomy_layers: Sequence[int] = (5, 6, 7, 8),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        temperature: float = 1.0,
        hard: bool = False,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
        depth_decay: float = 0.5,
        attn_heads: int = 0,
    ) -> None:
        super().__init__()

        self.default_hard = bool(hard)
        self.output_activation = output_activation.lower()

        encoder_cls = (
            TaxonResNetEncoderWithAttention if (attn_heads and attn_heads > 0) else TaxonResNetEncoder
        )
        encoder_kwargs = dict(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_taxonomy_layers=stage_taxonomy_layers,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            temperature=temperature,
            hard=hard,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
            depth_decay=depth_decay,
        )
        if encoder_cls is TaxonResNetEncoderWithAttention:
            encoder_kwargs["attn_heads"] = attn_heads

        self.encoder = encoder_cls(**encoder_kwargs)

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

    def encode(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode image tensor into taxonomy-aware latent tensor."""
        return self.encoder(x, hard=hard, return_details=return_details)

    def decode(
        self,
        z: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Decode latent tensor into reconstructed image tensor."""
        return self.decoder(z, output_size=output_size, return_details=return_details)

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation in {"identity", "none", "linear"}:
            return x
        raise ValueError(
            f"Unsupported output_activation={self.output_activation}. Use none/identity/linear, tanh, or sigmoid."
        )

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ):
        """Run full autoencoder pass.

        Args:
            x: Input image tensor of shape ``[B, C, H, W]``.
            hard: Optional override for pairwise softmax hard routing.
            return_details: If ``True``, also return detailed stage traces.

        Returns:
            - ``recon``
            - ``dkl``
            - ``entropy``
            - optional details dict when ``return_details=True``
        """
        if hard is None:
            hard = self.default_hard

        z, enc_details = self.encode(x, hard=hard, return_details=return_details)
        recon, dec_details = self.decode(z, output_size=x.shape[-2:], return_details=return_details)
        recon = self._apply_output_activation(recon)

        dkl = enc_details["dkl"]
        entropy = enc_details["entropy"]

        if not return_details:
            return recon, dkl, entropy

        details = {
            "input_shape": tuple(x.shape),
            "latent_shape": tuple(z.shape),
            "recon_shape": tuple(recon.shape),
            "hard": bool(hard),
            "encoder": enc_details,
            "decoder": dec_details,
        }
        return recon, dkl, entropy, details


__all__ = ["TaxonAutoencoder"]
