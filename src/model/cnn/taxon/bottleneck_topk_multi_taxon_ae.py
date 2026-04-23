"""Bottleneck-only TopK multi-taxon autoencoder.

Composes :class:`~.encoder.BottleneckTopKMultiTaxonResNetEncoder` (plain ResNet
stages followed by a single TopK multi-taxon stage at the bottleneck) with
:class:`~.decoder.TaxonResNetDecoder`.

With ``k_leaves=1`` and ``use_inter_hierarchy_gate=False``, every spatial
location of the latent has exactly ``K * L`` non-zero channels — one full
depth-L taxonomic path per hierarchy.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import TaxonResNetDecoder
from .encoder import BottleneckTopKMultiTaxonResNetEncoder


class BottleneckTopKMultiTaxonAutoencoder(nn.Module):
    """ResNet AE with a single TopK multi-taxon stage at the bottleneck."""

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
        topk_k_multiplier: float = 1.0,
        k_aux: Optional[int] = None,
        dead_steps: int = 2000,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
        use_batch_topk: bool = True,
        warmup_steps: int = 0,
        k_leaves: int = 1,
        use_gate_value: bool = False,
        use_inter_hierarchy_gate: bool = False,
        gate_k: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.n_hierarchies = int(n_hierarchies)
        self.output_activation = output_activation.lower()
        self.default_hard = hard

        self.encoder = BottleneckTopKMultiTaxonResNetEncoder(
            in_channels=in_channels,
            plain_stage_channels=plain_stage_channels,
            plain_stage_blocks=plain_stage_blocks,
            plain_stage_strides=plain_stage_strides,
            bottleneck_n_taxonomy_layers=bottleneck_n_taxonomy_layers,
            bottleneck_n_blocks=bottleneck_n_blocks,
            bottleneck_stride=bottleneck_stride,
            n_hierarchies=n_hierarchies,
            topk_k_multiplier=topk_k_multiplier,
            k_aux=k_aux,
            dead_steps=dead_steps,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
            temperature=temperature,
            hard=hard,
            depth_decay=depth_decay,
            use_batch_topk=use_batch_topk,
            warmup_steps=warmup_steps,
            out_channels=in_channels,
            k_leaves=k_leaves,
            use_gate_value=use_gate_value,
            use_inter_hierarchy_gate=use_inter_hierarchy_gate,
            gate_k=gate_k,
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

    def compute_auxk_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """AuxK loss summed across the K hierarchies of the bottleneck stage."""
        h = self.encoder.stem(x)
        for stage in self.encoder.plain_stages:
            with torch.no_grad():
                h = stage(h)
        total = x.new_zeros(())
        for hierarchy in self.encoder.bottleneck_stage.hierarchies:
            total = total + hierarchy.compute_auxk_loss(h, x, recon)
        return total

    def forward(self, x, hard=None, return_details=False):
        if hard is None:
            hard = self.default_hard
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

    def forward_matryoshka(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[List[torch.Tensor], dict]:
        """Return one reconstruction per depth prefix at the bottleneck stage.

        Prefix d keeps depths 1..d+1 of every hierarchy.
        """
        if hard is None:
            hard = self.default_hard
        z, enc_details = self.encode(x, hard=hard)

        last_stage = self.encoder.bottleneck_stage
        n_layers = last_stage.n_taxonomy_layers
        layer_ch = last_stage.layer_channels
        hier_ch = last_stage.hierarchy_out_channels
        K = self.n_hierarchies

        prefix_recons: List[torch.Tensor] = []
        for d in range(n_layers):
            prefix_ch = sum(layer_ch[: d + 1])
            mask = torch.zeros(1, K * hier_ch, 1, 1, device=z.device)
            for k in range(K):
                start = k * hier_ch
                mask[:, start : start + prefix_ch] = 1.0
            z_prefix = z * mask
            recon_d, _ = self.decode(z_prefix, output_size=x.shape[-2:])
            prefix_recons.append(self._apply_output_activation(recon_d))

        return prefix_recons, enc_details


__all__ = ["BottleneckTopKMultiTaxonAutoencoder"]
