"""SAEBench-compatible Taxon SAE.

Combines a linear taxonomy encoder (binary-tree pairwise-softmax routing)
with the SAEBench BaseSAE interface.

The latent width (d_sae) is determined entirely by the taxonomy tree:
  d_sae = 2^(n_taxonomy_layers + 1) - 2
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



# ── Taxonomy routing stage ───────────────────────────────────────────────

class LinearTaxonStage(nn.Module):
    """One taxonomy tree over a linear projection.

    Mirrors :class:`TaxonResNetStage` (CNN) but replaces ``ResidualConvBlock``
    with ``nn.Linear`` and operates on 2-D ``[B, d_model]`` tensors.

    Raw (unconstrained) logits drive the pairwise-softmax routing decisions,
    while ``ReLU(logits)`` provides non-negative feature magnitudes.  This
    decouples routing expressiveness from the sparsity mechanism.

    Args:
        d_input:  Dimensionality of the input activation vector.
        n_taxonomy_layers:  Depth of the binary tree (L).
        temperature:  Softmax temperature for sibling competition.
        hard:  Use straight-through hard routing.
        depth_decay:  Exponential weight decay per depth for regularisers.
    """

    def __init__(
        self,
        d_input: int,
        n_taxonomy_layers: int,
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()

        if n_taxonomy_layers < 1:
            raise ValueError(f"n_taxonomy_layers must be >= 1, got {n_taxonomy_layers}")

        self.d_input = d_input
        self.n_taxonomy_layers = n_taxonomy_layers
        self.temperature = float(temperature)
        self.default_hard = bool(hard)
        self.depth_decay = float(depth_decay)

        self.layer_channels: List[int] = [1 << (i + 1) for i in range(n_taxonomy_layers)]
        self.total_out_channels = self.output_channels(n_taxonomy_layers)

        self.linear = nn.Linear(d_input, self.total_out_channels)

    @staticmethod
    def output_channels(n_taxonomy_layers: int) -> int:
        """Total output features for depths [2, 4, ..., 2^L]."""
        return (1 << (n_taxonomy_layers + 1)) - 2

    def _pairwise_log_softmax(
        self,
        logits: torch.Tensor,
        tau: float,
        hard: bool,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Pairwise softmax over sibling pairs — *log-space* for regularisation.

        Returns log-conditional probabilities [B, C].
        """
        B, C = logits.shape
        pair_logits = logits.view(B, C // 2, 2)
        y_soft = torch.softmax(pair_logits / tau, dim=2)

        if hard:
            argmax = y_soft.argmax(dim=2, keepdim=True)
            y_hard = torch.zeros_like(pair_logits).scatter_(2, argmax, 1.0)
            probs = y_hard - y_soft.detach() + y_soft
        else:
            probs = y_soft

        return probs.clamp_min(eps).log().view(B, C)

    def _pairwise_softmax(
        self,
        logits: torch.Tensor,
        tau: float,
        hard: bool,
    ) -> torch.Tensor:
        """Pairwise softmax — *raw probability* space for feature gating.

        With hard routing the losing sibling gets exactly 0 (not eps), so
        ``relu(logit) * gate_prob`` produces true zeros on off-path features.
        """
        B, C = logits.shape
        pair_logits = logits.view(B, C // 2, 2)
        y_soft = torch.softmax(pair_logits / tau, dim=2)

        if hard:
            argmax = y_soft.argmax(dim=2, keepdim=True)
            y_hard = torch.zeros_like(pair_logits).scatter_(2, argmax, 1.0)
            probs = y_hard - y_soft.detach() + y_soft
        else:
            probs = y_soft

        return probs.view(B, C)

    def _regularization_terms(
        self,
        prob: torch.Tensor,
        logp: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Entropy and coverage-DKL for one depth level."""
        local_entropy = -(prob * logp).sum(dim=1).mean()

        marginal = prob.mean(dim=0)
        marginal = marginal / marginal.sum().clamp_min(eps)
        uniform_logp = -math.log(prob.shape[1])
        coverage_kl = (marginal * (marginal.clamp_min(eps).log() - uniform_logp)).sum()

        return local_entropy, coverage_kl

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run one taxonomy stage.

        Args:
            x: [B, d_input] activation vectors.
            hard: Override for hard routing.

        Returns:
            gated_output: [B, total_out_channels] — taxonomy-gated features.
            info: Dict with ``dkl`` and ``entropy`` scalars.
        """
        if hard is None:
            hard = self.default_hard

        # Raw logits (unconstrained) for routing; ReLU'd for feature values
        all_logits = self.linear(x)                         # [B, total_out_channels]
        depth_logits = torch.split(all_logits, self.layer_channels, dim=1)

        outputs: List[torch.Tensor] = []
        prev_logp: Optional[torch.Tensor] = None
        prev_gate: Optional[torch.Tensor] = None
        total_entropy = x.new_zeros(())
        total_dkl = x.new_zeros(())

        for depth_idx, logits in enumerate(depth_logits):
            # Log-space routing for regularisation (entropy / DKL)
            log_cond = self._pairwise_log_softmax(logits, self.temperature, hard)
            logp = log_cond if prev_logp is None else log_cond + prev_logp.repeat_interleave(2, dim=1)
            prob = logp.exp()

            # Raw-probability routing for feature gating (exact zeros with hard)
            cond = self._pairwise_softmax(logits, self.temperature, hard)
            gate_prob = cond if prev_gate is None else cond * prev_gate.repeat_interleave(2, dim=1)

            # Feature values: non-negative magnitude × routing probability
            out = F.relu(logits) * gate_prob

            entropy_i, dkl_i = self._regularization_terms(prob, logp)
            w = self.depth_decay ** depth_idx
            total_entropy = total_entropy + w * entropy_i
            total_dkl = total_dkl + w * dkl_i

            outputs.append(out)
            prev_logp = logp
            prev_gate = gate_prob

        return torch.cat(outputs, dim=1), {"dkl": total_dkl, "entropy": total_entropy}


# ── SAEBench wrapper ─────────────────────────────────────────────────────

class TaxonSAE(nn.Module):
    """SAEBench-compatible linear taxonomy SAE.

    Internally uses a LinearTaxonStage encoder (binary-tree pairwise softmax
    routing) and a linear decoder, but exposes the standard BaseSAE interface.
    """

    def __init__(
        self,
        d_in: int,
        n_taxonomy_layers: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        temperature: float = 0.5,
        hard: bool = False,
        depth_decay: float = 0.5,
        hook_name: str | None = None,
    ):
        d_sae = LinearTaxonStage.output_channels(n_taxonomy_layers)

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__()

        self.n_taxonomy_layers = n_taxonomy_layers
        self.temperature = temperature
        self.default_hard = hard
        self.depth_decay = depth_decay

        # Taxonomy-structured encoder
        self.encoder = LinearTaxonStage(
            d_input=d_in,
            n_taxonomy_layers=n_taxonomy_layers,
            temperature=temperature,
            hard=hard,
            depth_decay=depth_decay,
        )

        # W_dec: [d_sae, d_in]  — decoder directions (unit-normed rows)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)

        # W_enc: [d_in, d_sae]  — kept for BaseSAE compatibility but NOT used
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae, device=device, dtype=dtype))

        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype))

        self.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # SAEBench interface
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        flat = flat - self.b_dec
        z, _ = self.encoder(flat)              # uses self.default_hard from config
        return z.reshape(*shape[:-1], -1)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------
    # Training helpers (not part of SAEBench interface)
    # ------------------------------------------------------------------

    def encode_with_info(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat = x.reshape(-1, x.shape[-1])
        flat = flat - self.b_dec
        z, info = self.encoder(flat)
        return z.reshape(*x.shape[:-1], -1), info

    def forward_with_loss(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z, info = self.encode_with_info(x)
        x_hat = self.decode(z)
        info["x_hat"] = x_hat
        return x_hat, info

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.W_dec.data.norm(dim=1, keepdim=True)
        self.W_dec.data /= norms.clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def get_checkpoint_args(self) -> dict:
        return {
            "model_variant": "taxon_sae",
            "d_in": self.cfg.d_in,
            "d_sae": self.cfg.d_sae,
            "n_taxonomy_layers": self.n_taxonomy_layers,
            "temperature": self.temperature,
            "hard": self.default_hard,
            "depth_decay": self.depth_decay,
            "model_name": self.cfg.model_name,
            "hook_layer": self.cfg.hook_layer,
            "hook_name": self.cfg.hook_name,
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "TaxonSAE":
        state = torch.load(path, map_location="cpu", weights_only=False)
        args = state["args"]
        sae = cls(
            d_in=args["d_in"],
            n_taxonomy_layers=args["n_taxonomy_layers"],
            model_name=args["model_name"],
            hook_layer=args["hook_layer"],
            device=device,
            dtype=dtype,
            temperature=args.get("temperature", 0.5),
            hard=args.get("hard", False),
            depth_decay=args.get("depth_decay", 0.5),
            hook_name=args.get("hook_name"),
        )
        sae.load_state_dict(state["model_state"])
        sae.to(device=device, dtype=dtype)
        return sae
