"""TopK + AuxK taxon autoencoder wrapper.

Composes :class:`~.encoder.TopKTaxonResNetEncoder` with
:class:`~.decoder.TaxonResNetDecoder`.

Uses the same hierarchical pairwise-softmax routing as the vanilla
:class:`~.taxon_ae.TaxonAutoencoder`.  Instead of DKL regularisation, diversity
is enforced by tracking dead nodes and an auxiliary loss that revives them.
Forward output:

* ``reconstruction``
* ``dead_frac``  — fraction of nodes flagged as dead (across all stages)
* optional ``details`` dictionary

Reference: Gao et al., "Scaling and Evaluating Sparse Autoencoders",
arXiv:2406.04093.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .decoder import TaxonResNetDecoder
from .encoder import TopKTaxonResNetEncoder


class TopKTaxonAutoencoder(nn.Module):
    """ResNet-style taxonomic autoencoder with hierarchical routing + AuxK loss."""

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_taxonomy_layers: Sequence[int] = (5, 6, 7, 8),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
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
        skip_rank: int = 0,
        k_leaves: int = 0,
    ) -> None:
        super().__init__()

        self.output_activation = output_activation.lower()
        self.default_hard = bool(hard)
        self.skip_rank = int(skip_rank)

        self.encoder = TopKTaxonResNetEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_taxonomy_layers=stage_taxonomy_layers,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
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

        # Low-rank skip connection (bypasses encoder + decoder).
        if self.skip_rank > 0:
            self.skip_down = nn.Conv2d(in_channels, skip_rank, kernel_size=1, bias=False)
            self.skip_up = nn.Conv2d(skip_rank, in_channels, kernel_size=1, bias=True)
            nn.init.kaiming_normal_(self.skip_down.weight, mode="fan_in")
            self.skip_down.weight.data.mul_(0.01)
            nn.init.kaiming_normal_(self.skip_up.weight, mode="fan_in")
            self.skip_up.weight.data.mul_(0.01)
            nn.init.zeros_(self.skip_up.bias)

    def encode(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        return self.encoder(x, hard=hard, return_details=return_details)

    def decode(
        self,
        z: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
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

    def compute_auxk_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate AuxK loss across all encoder stages.

        Replays the encoder forward pass to obtain the correct intermediate
        input for each stage (stages beyond 0 do not receive the raw image).
        """
        total = x.new_zeros(())
        h = self.encoder.stem(x)
        for stage in self.encoder.taxon_stages:
            total = total + stage.compute_auxk_loss(h, x, recon)
            # Replay the forward pass to advance h to the next stage's input.
            with torch.no_grad():
                h, _, _ = stage(h)
        return total

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ):
        """Run full autoencoder pass.

        Returns:
            - ``recon``     — reconstructed image ``[B, C, H, W]``
            - ``dead_frac`` — fraction of dead nodes (scalar)
            - optional details dict when ``return_details=True``
        """
        if hard is None:
            hard = self.default_hard
        z, enc_details = self.encode(x, hard=hard, return_details=return_details)
        recon, dec_details = self.decode(z, output_size=x.shape[-2:], return_details=return_details)
        if self.skip_rank > 0:
            recon = recon + self.skip_up(self.skip_down(x))
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
        """Forward pass returning one reconstruction per depth prefix.

        Returns:
            - ``prefix_recons`` — list of ``n_layers`` reconstructions
            - ``enc_details``   — encoder details dict with dead_frac etc.
        """
        if hard is None:
            hard = self.default_hard
        z, enc_details = self.encode(x, hard=hard)

        last_stage = self.encoder.taxon_stages[-1]
        n_layers = last_stage.n_taxonomy_layers
        layer_ch = last_stage.layer_channels

        skip_out = self.skip_up(self.skip_down(x)) if self.skip_rank > 0 else None

        prefix_recons: List[torch.Tensor] = []
        for d in range(n_layers):
            prefix_ch = sum(layer_ch[: d + 1])
            mask = torch.zeros(1, z.size(1), 1, 1, device=z.device)
            mask[:, :prefix_ch] = 1.0
            z_prefix = z * mask
            recon_d, _ = self.decode(z_prefix, output_size=x.shape[-2:])
            if skip_out is not None:
                recon_d = recon_d + skip_out
            prefix_recons.append(self._apply_output_activation(recon_d))

        return prefix_recons, enc_details


__all__ = ["TopKTaxonAutoencoder"]
