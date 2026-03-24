"""TopK + AuxK multi-hierarchy taxon autoencoder wrapper.

Composes :class:`~.encoder.TopKMultiTaxonResNetEncoder` with
:class:`~.decoder.TaxonResNetDecoder`.

Each encoder stage applies K independent TopK taxonomy hierarchies with a
TopK inter-hierarchy gate.  Forward output:

* ``reconstruction``
* ``dead_frac``  — total dead-leaf fraction across all stages and hierarchies
* optional ``details`` dictionary

Reference: Gao et al., "Scaling and Evaluating Sparse Autoencoders",
arXiv:2406.04093.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import TaxonResNetDecoder
from .encoder import TopKMultiTaxonResNetEncoder


class TopKMultiTaxonAutoencoder(nn.Module):
    """ResNet-style autoencoder with K TopK+AuxK taxonomy hierarchies per stage."""

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_taxonomy_layers: Sequence[int] = (3, 4, 5, 6),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        n_hierarchies: int = 3,
        k: Optional[int] = None,
        k_aux: Optional[int] = None,
        dead_steps: int = 2000,
        gate_k: int = 1,
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

        self.n_hierarchies = int(n_hierarchies)
        self.output_activation = output_activation.lower()

        self.encoder = TopKMultiTaxonResNetEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_taxonomy_layers=stage_taxonomy_layers,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            n_hierarchies=n_hierarchies,
            k=k,
            k_aux=k_aux,
            dead_steps=dead_steps,
            gate_k=gate_k,
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

    def compute_auxk_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """Aggregate AuxK loss across all encoder stages and hierarchies.

        Replays the encoder forward pass to obtain the correct intermediate
        input for each stage (stages beyond 0 do not receive the raw image).
        """
        total = x.new_zeros(())
        h = self.encoder.stem(x)
        for stage in self.encoder.multi_taxon_stages:
            for hierarchy in stage.hierarchies:
                total = total + hierarchy.compute_auxk_loss(h, x, recon)
            # Replay the forward pass to advance h to the next stage's input.
            with torch.no_grad():
                h, _, _ = stage(h)
        return total

    def forward(self, x, hard=None, return_details=False):
        """Run full autoencoder pass.

        Returns:
            - ``recon``     — reconstructed image
            - ``dead_frac`` — total dead-leaf fraction scalar
            - optional details dict when ``return_details=True``
        """
        z, enc_details = self.encode(x, hard=hard, return_details=return_details)
        recon, dec_details = self.decode(z, output_size=x.shape[-2:], return_details=return_details)
        recon = self._apply_output_activation(recon)

        dead_frac = enc_details["dead_frac"]

        if not return_details:
            return recon, dead_frac

        details = {
            "input_shape": tuple(x.shape),
            "latent_shape": tuple(z.shape),
            "recon_shape": tuple(recon.shape),
            "encoder": enc_details,
            "decoder": dec_details,
        }
        return recon, dead_frac, details


__all__ = ["TopKMultiTaxonAutoencoder"]
