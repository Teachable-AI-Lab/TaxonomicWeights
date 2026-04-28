"""Bottleneck-only JumpReLU single-taxon autoencoder.

Plain ResNet stages followed by a single :class:`JumpReLUTaxonResNetStage` at
the bottleneck — same hierarchical pairwise-softmax routing and DKL/entropy
regularisers as :class:`BottleneckTopKTaxonAutoencoder`, but the per-path
sparsity is enforced by a learned per-channel JumpReLU threshold instead of
batch-TopK + AuxK revival.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import TaxonResNetDecoder
from .encoder import _BottleneckEncoderBase
from .jumprelu_taxon_stage import JumpReLUTaxonResNetStage


class BottleneckJumpReLUTaxonResNetEncoder(_BottleneckEncoderBase):
    """Plain ResNet stages followed by a single JumpReLU taxon bottleneck stage."""

    def __init__(
        self,
        in_channels: int = 3,
        plain_stage_channels: Sequence[int] = (64, 128, 256),
        plain_stage_blocks: Sequence[int] = (2, 2, 2),
        plain_stage_strides: Sequence[int] = (1, 2, 2),
        bottleneck_n_taxonomy_layers: int = 8,
        bottleneck_n_blocks: int = 2,
        bottleneck_stride: int = 2,
        target_l0: float = 8.0,
        bandwidth: float = 0.001,
        theta_init: float = 0.05,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            plain_stage_channels=plain_stage_channels,
            plain_stage_blocks=plain_stage_blocks,
            plain_stage_strides=plain_stage_strides,
            bottleneck_n_taxonomy_layers=bottleneck_n_taxonomy_layers,
            bottleneck_n_blocks=bottleneck_n_blocks,
            bottleneck_stride=bottleneck_stride,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
            temperature=temperature,
            hard=hard,
        )
        self.bottleneck_stage = JumpReLUTaxonResNetStage(
            in_channels=self._final_pre_bottleneck_channels,
            n_taxonomy_layers=self.bottleneck_n_taxonomy_layers,
            n_blocks=self.bottleneck_n_blocks,
            stride=self.bottleneck_stride,
            kernel_size=kernel_size,
            target_l0=target_l0,
            bandwidth=bandwidth,
            theta_init=theta_init,
            temperature=self.temperature,
            hard=self.default_hard,
            depth_decay=depth_decay,
        )
        self.final_channels = self.bottleneck_stage.total_out_channels

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if hard is None:
            hard = self.default_hard
        details: Dict[str, object] = {
            "input_shape": tuple(x.shape),
            "shape_trace": [], "stages": [],
        }
        x = self._run_stem_and_plain(x, details, return_details)
        x, stage_logp, regs = self.bottleneck_stage(x, hard=hard)
        if return_details:
            details["shape_trace"].append(("bottleneck", tuple(x.shape)))
            details["stages"].append({
                "name": "bottleneck", "output": x, "logp": stage_logp,
                "output_shape": tuple(x.shape),
                "dead_frac": regs["dead_frac"],
                "entropy": regs["entropy"], "dkl": regs["dkl"],
                "l0_hat": regs["l0_hat"], "sparsity": regs["sparsity"],
            })
        details["latent_shape"] = tuple(x.shape)
        details["dead_frac"] = regs["dead_frac"]
        details["entropy"] = regs["entropy"]
        details["dkl"] = regs["dkl"]
        details["l0_hat"] = regs["l0_hat"]
        details["sparsity"] = regs["sparsity"]
        return x, details


class BottleneckJumpReLUTaxonAutoencoder(nn.Module):
    """ResNet AE with a single JumpReLU taxon stage at the bottleneck."""

    def __init__(
        self,
        in_channels: int = 3,
        plain_stage_channels: Sequence[int] = (64, 128, 256),
        plain_stage_blocks: Sequence[int] = (2, 2, 2),
        plain_stage_strides: Sequence[int] = (1, 2, 2),
        bottleneck_n_taxonomy_layers: int = 8,
        bottleneck_n_blocks: int = 2,
        bottleneck_stride: int = 2,
        target_l0: float = 8.0,
        bandwidth: float = 0.001,
        theta_init: float = 0.05,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()
        self.output_activation = output_activation.lower()
        self.default_hard = bool(hard)

        self.encoder = BottleneckJumpReLUTaxonResNetEncoder(
            in_channels=in_channels,
            plain_stage_channels=plain_stage_channels,
            plain_stage_blocks=plain_stage_blocks,
            plain_stage_strides=plain_stage_strides,
            bottleneck_n_taxonomy_layers=bottleneck_n_taxonomy_layers,
            bottleneck_n_blocks=bottleneck_n_blocks,
            bottleneck_stride=bottleneck_stride,
            target_l0=target_l0,
            bandwidth=bandwidth,
            theta_init=theta_init,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
            temperature=temperature,
            hard=hard,
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
        raise ValueError(f"Unsupported output_activation={self.output_activation}.")

    def forward(self, x, hard=None, return_details=False):
        if hard is None:
            hard = self.default_hard
        z, enc_details = self.encode(x, hard=hard, return_details=True)
        recon, dec_details = self.decode(z, output_size=x.shape[-2:], return_details=return_details)
        recon = self._apply_output_activation(recon)
        info = {
            "dead_frac": enc_details["dead_frac"],
            "sparsity": enc_details["sparsity"],
            "l0_hat": enc_details["l0_hat"],
            "entropy": enc_details["entropy"],
            "dkl": enc_details["dkl"],
        }
        if not return_details:
            return recon, info
        details = {
            "input_shape": tuple(x.shape),
            "latent_shape": tuple(z.shape),
            "recon_shape": tuple(recon.shape),
            "encoder": enc_details,
            "decoder": dec_details,
        }
        return recon, info, details


__all__ = [
    "BottleneckJumpReLUTaxonResNetEncoder",
    "BottleneckJumpReLUTaxonAutoencoder",
]
