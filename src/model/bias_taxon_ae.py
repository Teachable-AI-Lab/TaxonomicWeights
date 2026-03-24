"""Sigmoid + Bias (Loss-Free Balancing) taxon autoencoder wrapper.

Composes :class:`~.encoder.BiasTaxonResNetEncoder` with
:class:`~.decoder.TaxonResNetDecoder`.

Instead of softmax and DKL / entropy losses, uses sigmoid routing with
non-differentiable per-leaf biases that heuristically balance leaf usage.
Forward output:

* ``reconstruction``
* optional ``details`` dictionary

**No regularisation terms at all** — diversity is maintained purely through
the bias update rule.

Reference: Wang et al., "Auxiliary-Loss-Free Load Balancing Strategy
for Mixture-of-Experts", arXiv:2408.15664.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import TaxonResNetDecoder
from .encoder import BiasTaxonResNetEncoder


class BiasTaxonAutoencoder(nn.Module):
    """ResNet-style taxonomic autoencoder with sigmoid + bias routing."""

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_taxonomy_layers: Sequence[int] = (5, 6, 7, 8),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        k: Optional[int] = None,
        bias_update_rate: float = 0.001,
        bias_ema_decay: float = 0.99,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
        temperature: float = 1.0,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()

        self.output_activation = output_activation.lower()

        self.encoder = BiasTaxonResNetEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_taxonomy_layers=stage_taxonomy_layers,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            k=k,
            bias_update_rate=bias_update_rate,
            bias_ema_decay=bias_ema_decay,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
            temperature=temperature,
            depth_decay=depth_decay,
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

    def encode(self, x, hard=None, return_details=False):
        return self.encoder(x, hard=hard, return_details=return_details)

    def decode(self, z, output_size=None, return_details=False):
        return self.decoder(z, output_size=output_size, return_details=return_details)

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation in {"identity", "none", "linear"}:
            return x
        raise ValueError(
            f"Unsupported output_activation={self.output_activation}."
        )

    def forward(self, x, hard=None, return_details=False):
        """Run full autoencoder pass.

        Returns:
            - ``recon``  — reconstructed image ``[B, C, H, W]``
            - optional details dict when ``return_details=True``
        """
        z, enc_details = self.encode(x, hard=hard, return_details=return_details)
        recon, dec_details = self.decode(z, output_size=x.shape[-2:], return_details=return_details)
        recon = self._apply_output_activation(recon)

        if not return_details:
            return recon,

        details = {
            "input_shape": tuple(x.shape),
            "latent_shape": tuple(z.shape),
            "recon_shape": tuple(recon.shape),
            "encoder": enc_details,
            "decoder": dec_details,
        }
        return recon, details


__all__ = ["BiasTaxonAutoencoder"]
