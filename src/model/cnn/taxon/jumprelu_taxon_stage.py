"""Bottleneck JumpReLU taxon stage.

Same hierarchical pairwise-softmax routing as
:class:`TopKTaxonResNetStage` (with ``depth_scales`` and an optional
``coverage_kl`` regulariser) but the **sparsity mechanism is JumpReLU**
instead of batch-TopK + AuxK revival.

Each output channel has its own learnable threshold ``θ_c = exp(log_theta_c)``:

    latent_c = relu(z_c) · H(z_c − θ_c)

with a Straight-Through Estimator for the gradient through the Heaviside
step (Rajamanoharan et al. 2024 — "Jumping Ahead").  Sparsity is encouraged
by penalising ``(L̂₀ − target_l0)²``.

The pre-activation ``z`` is the routing-gated taxon output
``logits * prob * depth_scales`` (same as in the TopK variant).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .encoder import ResidualConvBlock
from ..baseline.jumprelu_sae_encoder import _JumpReLUActivation, _L0SurrogateFunction


class JumpReLUTaxonResNetStage(nn.Module):
    """One ResNet stage with hierarchical pairwise softmax + JumpReLU sparsity.

    Identical routing to :class:`TopKTaxonResNetStage`:
      * pairwise (sibling) softmax at every binary split
      * cumulative log-probability across depths
      * per-depth learnable scale ``depth_scales``
      * ``depth_decay``-weighted local entropy / coverage_kl regularisers

    The TopK + AuxK + dead-node tracking are replaced by a per-channel
    learned threshold ``θ_c`` (JumpReLU).
    """

    def __init__(
        self,
        in_channels: int,
        n_taxonomy_layers: int,
        n_blocks: int,
        stride: int = 1,
        kernel_size: int = 3,
        target_l0: float = 8.0,
        bandwidth: float = 0.001,
        theta_init: float = 0.05,
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()
        if n_taxonomy_layers < 1:
            raise ValueError(f"n_taxonomy_layers must be >= 1, got {n_taxonomy_layers}")

        self.in_channels = in_channels
        self.n_taxonomy_layers = n_taxonomy_layers
        self.n_blocks = n_blocks
        self.stride = stride
        self.temperature = float(temperature)
        self.default_hard = bool(hard)
        self.depth_decay = float(depth_decay)
        self.target_l0 = float(target_l0)
        self.bandwidth = float(bandwidth)
        self.theta_init = float(theta_init)

        self.layer_channels: List[int] = [1 << (i + 1) for i in range(self.n_taxonomy_layers)]
        self.total_out_channels = self.output_channels(self.n_taxonomy_layers)

        # Per-depth learnable magnitude scale (init 1 = no-op).
        self.depth_scales = nn.Parameter(torch.ones(n_taxonomy_layers))

        blocks: List[nn.Module] = []
        for idx in range(self.n_blocks):
            block_in = self.in_channels if idx == 0 else self.total_out_channels
            block_stride = self.stride if idx == 0 else 1
            blocks.append(
                ResidualConvBlock(
                    in_channels=block_in,
                    out_channels=self.total_out_channels,
                    kernel_size=kernel_size,
                    stride=block_stride,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Per-channel learnable threshold (stored as log θ so θ > 0).
        init_log = float(torch.tensor(self.theta_init).log().item())
        self.log_theta = nn.Parameter(
            torch.full((self.total_out_channels,), init_log)
        )

    @staticmethod
    def output_channels(n_taxonomy_layers: int) -> int:
        return (1 << (n_taxonomy_layers + 1)) - 2

    # ------------------------------------------------------------------ helpers
    def _stage_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        for block in self.blocks:
            feat = block(feat)
        return feat

    def _taxon_logits_per_depth(self, x: torch.Tensor) -> List[torch.Tensor]:
        stage_logits = self._stage_features(x)
        if stage_logits.shape[1] != self.total_out_channels:
            raise ValueError(
                f"Stage output channel mismatch: got {stage_logits.shape[1]}, "
                f"expected {self.total_out_channels}."
            )
        return list(torch.split(stage_logits, self.layer_channels, dim=1))

    def _pairwise_log_softmax(
        self,
        logits: torch.Tensor,
        tau: Optional[float] = None,
        hard: Optional[bool] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if tau is None:
            tau = self.temperature
        if hard is None:
            hard = self.default_hard

        if logits.shape[1] % 2 != 0:
            raise ValueError(
                f"Expected even channel count for pairwise softmax, got {logits.shape[1]}"
            )

        bsz, channels, h, w = logits.shape
        pair_logits = logits.view(bsz, channels // 2, 2, h, w)
        y_soft = torch.softmax(pair_logits / tau, dim=2)

        if hard:
            argmax = y_soft.argmax(dim=2, keepdim=True)
            y_hard = torch.zeros_like(pair_logits).scatter_(2, argmax, 1.0)
            probs = y_hard - y_soft.detach() + y_soft
        else:
            probs = y_soft

        return probs.clamp_min(eps).log().view(bsz, channels, h, w)

    def _regularization_terms(
        self,
        prob: torch.Tensor,
        logp: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        local_entropy = -(prob * logp).sum(dim=1).mean()
        marginal = prob.mean(dim=(0, 2, 3))
        marginal = marginal / marginal.sum().clamp_min(eps)
        uniform_logp = -math.log(prob.shape[1])
        coverage_kl = (marginal * (marginal.clamp_min(eps).log() - uniform_logp)).sum()
        return local_entropy, coverage_kl

    # -------------------------------------------------------------------- main
    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        outputs: List[torch.Tensor] = []
        logps: List[torch.Tensor] = []
        prev: Optional[torch.Tensor] = None

        total_entropy = x.new_zeros(())
        total_dkl = x.new_zeros(())

        routing_per_depth = self._taxon_logits_per_depth(x)
        for depth_idx, logits in enumerate(routing_per_depth):
            log_cond = self._pairwise_log_softmax(logits, tau=self.temperature, hard=hard)
            logp = log_cond if prev is None else log_cond + prev.repeat_interleave(2, dim=1)
            prob = logp.exp()
            out = logits * prob * self.depth_scales[depth_idx]

            entropy_i, dkl_i = self._regularization_terms(prob, logp)
            depth_weight = self.depth_decay ** depth_idx
            total_entropy = total_entropy + depth_weight * entropy_i
            total_dkl = total_dkl + depth_weight * dkl_i

            outputs.append(out)
            logps.append(logp)
            prev = logp

        cat_output = torch.cat(outputs, dim=1)

        # ── JumpReLU sparsity ─────────────────────────────────────────────
        latent = _JumpReLUActivation.apply(cat_output, self.log_theta, self.bandwidth)
        l0_hat = _L0SurrogateFunction.apply(cat_output, self.log_theta, self.bandwidth)
        sparsity = (l0_hat - self.target_l0) ** 2

        # Dead-frac = fraction of channels that never fire over the batch.
        with torch.no_grad():
            any_active = latent.amax(dim=(0, 2, 3)) > 0
            dead_frac = (~any_active).float().mean()

        return (
            latent,
            torch.cat(logps, dim=1),
            {
                "entropy": total_entropy,
                "dkl": total_dkl,
                "dead_frac": dead_frac,
                "l0_hat": l0_hat,
                "sparsity": sparsity,
            },
        )


__all__ = ["JumpReLUTaxonResNetStage"]
