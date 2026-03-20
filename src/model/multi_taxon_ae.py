"""Multi-hierarchy taxon autoencoder wrapper.

Composes :class:`~.encoder.MultiTaxonResNetEncoder` with
:class:`~.decoder.TaxonResNetDecoder`.

Each encoder stage applies K independent taxonomy hierarchies and an
inter-hierarchy gate.  Forward output exposes four regularization scalars:

* ``dkl``          — within-hierarchy path-marginal KL (summed over stages × K)
* ``entropy``      — within-hierarchy path entropy   (summed over stages × K)
* ``gate_dkl``     — gate-marginal KL(marginal || uniform K)  (summed over stages)
* ``gate_entropy`` — per-spatial gate entropy (summed over stages; lower → spikier)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import TaxonResNetDecoder
from .encoder import MultiTaxonResNetEncoder


class MultiTaxonAutoencoder(nn.Module):
    """ResNet-style autoencoder with K independent taxonomy hierarchies per stage."""

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_taxonomy_layers: Sequence[int] = (5, 6, 7, 8),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        n_hierarchies: int = 3,
        temperature: float = 1.0,
        hard: bool = False,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()

        self.default_hard = bool(hard)
        self.n_hierarchies = int(n_hierarchies)
        self.output_activation = output_activation.lower()

        self.encoder = MultiTaxonResNetEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_taxonomy_layers=stage_taxonomy_layers,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            n_hierarchies=n_hierarchies,
            temperature=temperature,
            hard=hard,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
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

    def encode(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ):
        """Encode image tensor into multi-hierarchy latent."""
        return self.encoder(x, hard=hard, return_details=return_details)

    def decode(
        self,
        z: torch.Tensor,
        output_size=None,
        return_details: bool = False,
    ):
        """Decode latent tensor into reconstructed image."""
        return self.decoder(z, output_size=output_size, return_details=return_details)

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation in {"identity", "none", "linear"}:
            return x
        raise ValueError(
            f"Unsupported output_activation={self.output_activation}. "
            "Use none/identity/linear, tanh, or sigmoid."
        )

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ):
        """Run full multi-hierarchy autoencoder pass.

        Returns:
            - ``recon``          — reconstructed image ``[B, C, H, W]``
            - ``dkl``            — within-hierarchy path KL scalar
            - ``entropy``        — within-hierarchy path entropy scalar
            - ``gate_dkl``       — inter-hierarchy gate coverage KL scalar
            - ``gate_entropy``   — inter-hierarchy gate entropy scalar
            - optional details dict when ``return_details=True``
        """
        if hard is None:
            hard = self.default_hard

        z, enc_details = self.encoder(x, hard=hard, return_details=return_details)
        recon, dec_details = self.decoder(
            z, output_size=x.shape[-2:], return_details=return_details
        )
        recon = self._apply_output_activation(recon)

        dkl          = enc_details["dkl"]
        entropy      = enc_details["entropy"]
        gate_dkl     = enc_details["gate_dkl"]
        gate_entropy = enc_details["gate_entropy"]

        if return_details:
            return recon, dkl, entropy, gate_dkl, gate_entropy, {
                "encoder": enc_details,
                "decoder": dec_details,
            }
        return recon, dkl, entropy, gate_dkl, gate_entropy
