"""Bottleneck JumpReLU taxon stage.

Same hierarchical pairwise-softmax routing as
:class:`TopKTaxonResNetStage` (with ``depth_scales`` and an optional
``coverage_kl`` regulariser) but the **sparsity mechanism is JumpReLU**
instead of batch-TopK + AuxK revival.

Each output channel has its own learnable threshold ``θ_c = exp(log_theta_c)``:

    latent_c = relu(z_c) · H(z_c − θ_c)

The forward path uses the sharp Heaviside step (preserves true L0 sparsity
at inference) via :class:`_JumpReLUActivation` (Rajamanoharan et al. 2024
"Jumping Ahead" — STE rectangle-kernel gradient through θ for the *activation*
path).

For the **L0 sparsity penalty** we deliberately do *not* use the rectangle-
kernel STE surrogate from the SAE encoder.  That surrogate has zero gradient
on ``log_theta`` for any channel whose pre-activation is more than
``bandwidth/2`` away from the current threshold, which leads to threshold
freezing: live channels (z ≫ θ) cannot have their threshold raised, and dead
channels (z ≪ θ) cannot have it lowered.  Instead we use a temperature-
scaled **sigmoid surrogate** that is differentiable everywhere::

    L̂₀_soft = mean_{b,h,w} Σ_c σ((z_c − θ_c) / τ)

This gradient pushes ``θ`` up wherever ``z`` is large (when above target) and
down wherever ``z`` is small (when below target), so the L0 budget is
actually enforced.

The pre-activation ``z`` is the routing-gated taxon output
``logits * prob * depth_scales`` (same as in the TopK variant).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .encoder import ResidualConvBlock
from ..baseline.jumprelu_sae_encoder import _JumpReLUActivation


class JumpReLUTaxonResNetStage(nn.Module):
    """One ResNet stage with hierarchical pairwise softmax + JumpReLU sparsity.

    Identical routing to :class:`TopKTaxonResNetStage`:
      * pairwise (sibling) softmax at every binary split
      * cumulative log-probability across depths
      * per-depth learnable scale ``depth_scales``
      * ``depth_decay``-weighted local entropy / coverage_kl regularisers

    The TopK + AuxK + dead-node tracking are replaced by a per-channel
    learned threshold ``θ_c`` (JumpReLU) with a sigmoid L0 surrogate.
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
        l0_surrogate_temp: float = 1.0,
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
        gumbel: bool = False,
        resample_min_count: int = 0,
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
        self.l0_surrogate_temp = float(l0_surrogate_temp)
        self.gumbel = bool(gumbel)
        self.resample_min_count = int(resample_min_count)

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

        # Per-leaf activation count (rolling, reset by maybe_resample_dead_leaves).
        n_leaves = 1 << self.n_taxonomy_layers
        self.register_buffer(
            "_leaf_count", torch.zeros(n_leaves, dtype=torch.long)
        )

    # ------------------------------------------------------------------ tau
    def set_tau(self, tau: float) -> None:
        """External temperature override (used by trainer for tau annealing)."""
        self.temperature = float(tau)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (log_cond_probs [B,C,H,W], soft_pair [B,n_pairs,2,H,W], hard_pair).

        Soft and hard probabilities are returned in pair-shaped form for use in
        the Switch-Transformer balance loss.  ``log_cond_probs`` is what the
        rest of the forward pass consumes (with optional STE if ``hard``).
        """
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
        if self.gumbel and self.training:
            u = torch.rand_like(pair_logits).clamp_(1e-9, 1.0 - 1e-9)
            g = -torch.log(-torch.log(u))
            y_soft = torch.softmax((pair_logits + g) / tau, dim=2)
        else:
            y_soft = torch.softmax(pair_logits / tau, dim=2)

        argmax = y_soft.argmax(dim=2, keepdim=True)
        y_hard = torch.zeros_like(pair_logits).scatter_(2, argmax, 1.0)

        if hard:
            probs = y_hard - y_soft.detach() + y_soft
        else:
            probs = y_soft

        log_cond = probs.clamp_min(eps).log().view(bsz, channels, h, w)
        return log_cond, y_soft, y_hard

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
        balance_terms: List[torch.Tensor] = []

        routing_per_depth = self._taxon_logits_per_depth(x)
        for depth_idx, logits in enumerate(routing_per_depth):
            log_cond, soft_pair, hard_pair = self._pairwise_log_softmax(
                logits, tau=self.temperature, hard=hard
            )
            logp = log_cond if prev is None else log_cond + prev.repeat_interleave(2, dim=1)
            prob = logp.exp()

            # Switch-style balance loss restricted to ancestor-active pairs.
            # Ancestor weight per child = parent's cumulative prob (shape
            # [B, C_d, H, W] obtained by repeat_interleave on prev.exp()).
            if prev is None:
                anc_pair = torch.ones_like(soft_pair)
            else:
                anc = prev.exp().repeat_interleave(2, dim=1)
                anc_pair = anc.view_as(soft_pair)
            denom = anc_pair.sum(dim=(0, 3, 4)).clamp_min(1.0)  # [n_pairs, 2]
            p_mean = (soft_pair * anc_pair).sum(dim=(0, 3, 4)) / denom
            f_mean = (hard_pair * anc_pair).sum(dim=(0, 3, 4)) / denom
            balance_terms.append((2.0 * (f_mean * p_mean).sum(dim=-1)).mean())
            # Same prob floor as TopKTaxonResNetStage (BUGFIX 5): without this,
            # deep-node pre-activations are attenuated by ~2^-(d+1), so JumpReLU
            # thresholds settle above them and all deep channels die permanently.
            floor_d = 0.5 ** (depth_idx + 1)
            prob_gated = prob.clamp(min=floor_d)
            out = logits * prob_gated * self.depth_scales[depth_idx]

            entropy_i, dkl_i = self._regularization_terms(prob, logp)
            depth_weight = self.depth_decay ** depth_idx
            total_entropy = total_entropy + depth_weight * entropy_i
            total_dkl = total_dkl + depth_weight * dkl_i

            outputs.append(out)
            logps.append(logp)
            prev = logp

        cat_output = torch.cat(outputs, dim=1)

        # Deepest-depth leaf assignment for dead-leaf bookkeeping.
        if logps:
            with torch.no_grad():
                leaf_idx = logps[-1].argmax(dim=1)  # [B, H, W]
                if self.training:
                    n_leaves = 1 << self.n_taxonomy_layers
                    counts = torch.bincount(
                        leaf_idx.reshape(-1), minlength=n_leaves
                    )
                    self._leaf_count += counts

        balance_loss = (
            torch.stack(balance_terms).mean()
            if balance_terms
            else cat_output.new_zeros(())
        )

        # ── JumpReLU sparsity ─────────────────────────────────────────────
        # Clamp log_theta to a finite range to prevent runaway thresholds
        # (causes exp() overflow / val-loss explosion when the sparsity term
        # transiently dominates).  Clamp is differentiable in the interior;
        # outside it the gradient is zero, which is fine because we *want*
        # to keep θ inside this band.
        log_theta_c = self.log_theta.clamp(-10.0, 5.0)
        theta = log_theta_c.exp().view(1, -1, 1, 1)

        # Forward path: sharp Heaviside step, true L0 sparsity at inference.
        # Note: _JumpReLUActivation reads its own log_theta param; pass the
        # unclamped one (clamp range is wide enough that this is consistent).
        latent = _JumpReLUActivation.apply(cat_output, self.log_theta, self.bandwidth)

        # Diagnostics: true (non-differentiable) L0 count for logging.
        with torch.no_grad():
            l0_true = (cat_output > theta).float().sum(dim=1).mean()

        # L0 surrogate for the sparsity penalty.  Two design choices critical
        # for stability:
        #   1. **Wide sigmoid (τ ≳ 1):** σ' is non-negligible far from θ, so
        #      gradients flow to *dead* channels (z ≪ θ) and they can be
        #      revived by lowering θ.  Narrow τ (e.g. 0.1) recreates the
        #      rectangle-STE dead-zone problem in disguise.
        #   2. **Linear hinge above target:** penalize only when over-budget.
        #      A quadratic ``(l0_soft − target)²`` term has loss ~250 000 at
        #      init (l0_soft ≈ N_channels) and dominates recon by 10³, so
        #      the optimizer slams every threshold up and kills the network.
        #      With a hinge, once l0_soft ≤ target the sparsity gradient is
        #      zero and the recon loss alone pulls thresholds back down,
        #      automatically reviving over-suppressed channels.
        l0_soft = torch.sigmoid(
            (cat_output - theta) / self.l0_surrogate_temp
        ).sum(dim=1).mean()
        # Asymmetric hinge: strong push down when over-budget, weak push up
        # when under-budget.  The under-budget term is necessary because
        # _JumpReLUActivation's rectangle STE provides ZERO recon-gradient on
        # θ for dead channels (|z−θ| ≫ bandwidth/2), so the recon loss alone
        # cannot revive them.  The wide sigmoid (τ=1) does have gradient on
        # dead channels, so a small revival weight is enough to gradually
        # lower θ for any channel that has been killed.
        over = torch.relu(l0_soft - self.target_l0)
        under = torch.relu(self.target_l0 - l0_soft)
        sparsity = over + 0.05 * under

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
                "l0_hat": l0_true,
                "l0_soft": l0_soft.detach(),
                "sparsity": sparsity,
                "balance_loss": balance_loss,
            },
        )

    # ----------------------------------------------------------- resampling
    @torch.no_grad()
    def maybe_resample_dead_leaves(self) -> int:
        """Re-initialise final-block conv rows for paths to dead leaves.

        A leaf is "dead" if it was selected fewer than ``self.resample_min_count``
        times since the last reset.  For each dead leaf, we reset the rows of
        the final ResidualConvBlock's last Conv2d that correspond to every
        ancestor channel along the leaf's path; this breaks the symmetry that
        let the leaf collapse, without disturbing live leaves.  The leaf-count
        buffer is zeroed afterwards.  Returns the number of resampled leaves.
        """
        if self.resample_min_count <= 0:
            return 0
        counts = self._leaf_count
        dead = (counts < self.resample_min_count).nonzero(as_tuple=False).flatten()
        n_dead = int(dead.numel())
        if n_dead == 0:
            self._leaf_count.zero_()
            return 0

        # Locate the final conv with full output_channels = total_out_channels.
        final_block = self.blocks[-1]
        final_conv: Optional[nn.Conv2d] = None
        for m in reversed(list(final_block.main)):
            if isinstance(m, nn.Conv2d) and m.out_channels == self.total_out_channels:
                final_conv = m
                break
        if final_conv is None:
            self._leaf_count.zero_()
            return 0

        L = self.n_taxonomy_layers
        offset = 0
        # Map (depth, leaf_idx) -> absolute channel index in the catted output.
        for d in range(L):
            n_d = 1 << (d + 1)
            shift = (L - 1 - d)
            ancestors = (dead >> shift).unique()  # nodes at depth d under any dead leaf
            for node in ancestors.tolist():
                ch = offset + int(node)
                # Kaiming-fan-in re-init for this output channel only.
                fan_in = final_conv.weight.shape[1] * final_conv.weight.shape[2] * final_conv.weight.shape[3]
                std = (2.0 / fan_in) ** 0.5
                final_conv.weight.data[ch].normal_(0.0, std)
            offset += n_d

        # Also lower θ for the dead-leaf path channels so JumpReLU can fire again.
        offset = 0
        for d in range(L):
            n_d = 1 << (d + 1)
            shift = (L - 1 - d)
            ancestors = (dead >> shift).unique()
            for node in ancestors.tolist():
                ch = offset + int(node)
                # Reset to initial threshold (typically very small).
                self.log_theta.data[ch] = float(
                    torch.tensor(self.theta_init).log().item()
                )
            offset += n_d

        self._leaf_count.zero_()
        return n_dead


__all__ = ["JumpReLUTaxonResNetStage"]
