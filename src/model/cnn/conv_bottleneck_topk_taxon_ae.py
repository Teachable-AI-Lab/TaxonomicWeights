"""Strided-conv autoencoder with a TopK taxonomy bottleneck (no ResNet backbone).

Architecture
------------
Encoder:  StridedConvEncoder  3 → 64 → 128 → 256 → 512 → 1024  (8×8 spatial)
Bottleneck: TopKTaxonResNetStage(in=1024, L=L, n_blocks=1)
            → [B, 2^(L+1)-2, 8, 8]   (e.g. L=9 → 1022 ch)
Decoder:  StridedConvDecoder  taxon_ch → 512 → 256 → 128 → 64 → 3

The single ResNet block inside TopKTaxonResNetStage acts only as a channel
projection (1024 → taxon_out_channels); the backbone contains no residual
connections.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .conv_ae import StridedConvDecoder, StridedConvEncoder
from .taxon.encoder import TopKTaxonResNetStage


class ConvBottleneckTopKTaxonAutoencoder(nn.Module):
    """Strided-conv AE with a hierarchical pairwise-softmax + TopK bottleneck.

    Parameters
    ----------
    bottleneck_n_taxonomy_layers:
        Depth of the binary taxonomy tree at the bottleneck.
        Latent channels = 2^(L+1)-2  (e.g. L=9 → 1022).
    bottleneck_n_blocks:
        ResNet blocks *inside* the taxonomy stage only (channel projection).
        1 is the minimum useful value; set to 2 for extra capacity.
    topk_k_multiplier:
        Fraction of taxonomy channels kept by TopK  (1.0 = keep L channels).
    """

    def __init__(
        self,
        in_channels: int = 3,
        bottleneck_n_taxonomy_layers: int = 9,
        bottleneck_n_blocks: int = 1,
        topk_k_multiplier: float = 1.0,
        k_aux: Optional[int] = None,
        dead_steps: int = 2000,
        temperature: float = 0.5,
        hard: bool = False,
        depth_decay: float = 0.5,
        use_batch_topk: bool = True,
        warmup_steps: int = 0,
        output_activation: str = "none",
        k_leaves: int = 0,
        use_gate_value: bool = False,
    ) -> None:
        super().__init__()
        self.default_hard = bool(hard)
        self.output_activation = output_activation.lower()
        self.in_channels = in_channels

        self.encoder_net = StridedConvEncoder(in_channels=in_channels)
        enc_ch = self.encoder_net.out_channels  # 1024

        self.bottleneck_stage = TopKTaxonResNetStage(
            in_channels=enc_ch,
            n_taxonomy_layers=bottleneck_n_taxonomy_layers,
            n_blocks=bottleneck_n_blocks,
            stride=1,
            kernel_size=3,
            topk_k_multiplier=topk_k_multiplier,
            k_aux=k_aux,
            dead_steps=dead_steps,
            temperature=temperature,
            hard=hard,
            depth_decay=depth_decay,
            use_batch_topk=use_batch_topk,
            warmup_steps=warmup_steps,
            out_channels=in_channels,
            k_leaves=k_leaves,
            use_gate_value=use_gate_value,
        )

        taxon_ch = TopKTaxonResNetStage.output_channels(bottleneck_n_taxonomy_layers)
        self.taxon_channels = taxon_ch

        self.decoder_net = StridedConvDecoder(
            in_channels=taxon_ch,
            out_channels=in_channels,
            output_activation=output_activation,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        h = self.encoder_net(x)
        z, logp, regs = self.bottleneck_stage(h, hard=hard)
        recon = self.decoder_net(z, output_size=x.shape[-2:])
        details = {**regs, "logp": logp} if return_details else {}
        return recon, regs["dead_frac"], details

    def compute_auxk_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to the bottleneck stage's AuxK revival loss."""
        h = self.encoder_net(x)
        return self.bottleneck_stage.compute_auxk_loss(h, x, recon)

    def encode(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        h = self.encoder_net(x)
        z, logp, regs = self.bottleneck_stage(h, hard=hard)
        details = {**regs, "logp": logp} if return_details else {}
        return z, details

    def decode(
        self,
        z: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        recon = self.decoder_net(z, output_size=output_size)
        return recon, {}

    def forward_matryoshka(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[list, Dict]:
        """Prefix reconstructions matching BottleneckTopKTaxonAutoencoder.forward_matryoshka.

        For each depth d, decodes with all channels beyond depth d zeroed out
        (prefix-channel mask on the post-TopK latent), so the figure layout is
        identical to the DKL taxon models.
        """
        z, enc_details = self.encode(x, hard=hard)
        stage = self.bottleneck_stage
        layer_ch = stage.layer_channels
        n_layers = stage.n_taxonomy_layers
        prefix_recons = []
        for d in range(n_layers):
            prefix_ch = sum(layer_ch[: d + 1])
            mask = torch.zeros(1, z.size(1), 1, 1, device=z.device, dtype=z.dtype)
            mask[:, :prefix_ch] = 1.0
            recon_d, _ = self.decode(z * mask, output_size=x.shape[-2:])
            prefix_recons.append(recon_d)
        return prefix_recons, enc_details
