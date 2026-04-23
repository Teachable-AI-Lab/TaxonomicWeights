"""Bottleneck-only multi-hierarchy (regular softmax+DKL) taxon autoencoder.

Plain ResNet stages followed by a single :class:`MultiTaxonResNetStage` at the
bottleneck.  Returns within-hierarchy DKL/entropy AND inter-hierarchy
gate_dkl/gate_entropy regs.  ``hard=True`` selects exactly one hierarchy per
spatial position via the inter-hierarchy gate (straight-through argmax).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import TaxonResNetDecoder
from .encoder import BottleneckMultiTaxonResNetEncoder


class BottleneckMultiTaxonAutoencoder(nn.Module):
    """ResNet AE with a single multi-hierarchy (softmax+DKL) taxon stage at the bottleneck."""

    def __init__(
        self,
        in_channels: int = 3,
        plain_stage_channels: Sequence[int] = (64, 128, 256),
        plain_stage_blocks: Sequence[int] = (2, 2, 2),
        plain_stage_strides: Sequence[int] = (1, 2, 2),
        bottleneck_n_taxonomy_layers: int = 6,
        bottleneck_n_blocks: int = 2,
        bottleneck_stride: int = 2,
        n_hierarchies: int = 4,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
        k_leaves: int = 0,
        use_gate_value: bool = False,
    ) -> None:
        super().__init__()
        self.n_hierarchies = int(n_hierarchies)
        self.output_activation = output_activation.lower()
        self.default_hard = bool(hard)

        self.encoder = BottleneckMultiTaxonResNetEncoder(
            in_channels=in_channels,
            plain_stage_channels=plain_stage_channels,
            plain_stage_blocks=plain_stage_blocks,
            plain_stage_strides=plain_stage_strides,
            bottleneck_n_taxonomy_layers=bottleneck_n_taxonomy_layers,
            bottleneck_n_blocks=bottleneck_n_blocks,
            bottleneck_stride=bottleneck_stride,
            n_hierarchies=n_hierarchies,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
            temperature=temperature,
            hard=hard,
            depth_decay=depth_decay,
            k_leaves=k_leaves,
            use_gate_value=use_gate_value,
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
        raise ValueError(f"Unsupported output_activation={self.output_activation}.")

    def forward(self, x, hard=None, return_details=False):
        if hard is None:
            hard = self.default_hard
        z, enc_details = self.encode(x, hard=hard, return_details=return_details)
        recon, dec_details = self.decode(z, output_size=x.shape[-2:], return_details=return_details)
        recon = self._apply_output_activation(recon)
        dkl = enc_details["dkl"]
        entropy = enc_details["entropy"]
        gate_dkl = enc_details["gate_dkl"]
        gate_entropy = enc_details["gate_entropy"]
        if return_details:
            return recon, dkl, entropy, gate_dkl, gate_entropy, {
                "encoder": enc_details, "decoder": dec_details,
            }
        return recon, dkl, entropy, gate_dkl, gate_entropy

    def forward_matryoshka(
        self, x: torch.Tensor, hard: Optional[bool] = None,
    ) -> Tuple[List[torch.Tensor], dict]:
        if hard is None:
            hard = self.default_hard
        z, enc_details = self.encode(x, hard=hard)
        last = self.encoder.bottleneck_stage
        n_layers = last.n_taxonomy_layers
        layer_ch = last.layer_channels
        hier_ch = last.hierarchy_out_channels
        K = self.n_hierarchies
        prefix_recons: List[torch.Tensor] = []
        for d in range(n_layers):
            prefix_ch = sum(layer_ch[: d + 1])
            mask = torch.zeros(1, K * hier_ch, 1, 1, device=z.device)
            for k in range(K):
                start = k * hier_ch
                mask[:, start : start + prefix_ch] = 1.0
            recon_d, _ = self.decode(z * mask, output_size=x.shape[-2:])
            prefix_recons.append(self._apply_output_activation(recon_d))
        return prefix_recons, enc_details


__all__ = ["BottleneckMultiTaxonAutoencoder"]
