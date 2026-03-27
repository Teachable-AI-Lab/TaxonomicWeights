"""SAEBench-compatible Multi-Taxon SAE.

K independent taxonomy hierarchies with an inter-hierarchy gate,
conforming to the SAEBench BaseSAE interface.

The latent width is determined entirely by the taxonomy trees:
  d_sae = n_hierarchies * (2^(n_taxonomy_layers + 1) - 2)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sae_bench.custom_saes.base_sae import BaseSAE

from .taxon_sae import LinearTaxonStage


# ── Multi-hierarchy routing stage ────────────────────────────────────────

class LinearMultiTaxonStage(nn.Module):
    """K independent taxonomy trees + inter-hierarchy gate (linear version).

    Args:
        d_input:  Dimensionality of the input activation vector.
        n_taxonomy_layers:  Depth of each binary tree.
        n_hierarchies:  Number of independent trees (K).
        temperature:  Softmax temperature for routing.
        hard:  Straight-through hard routing.
        depth_decay:  Exponential weight for depth-level regs.
    """

    def __init__(
        self,
        d_input: int,
        n_taxonomy_layers: int,
        n_hierarchies: int = 4,
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()
        if n_hierarchies < 1:
            raise ValueError(f"n_hierarchies must be >= 1, got {n_hierarchies}")

        self.d_input = d_input
        self.n_taxonomy_layers = n_taxonomy_layers
        self.n_hierarchies = n_hierarchies
        self.temperature = float(temperature)
        self.default_hard = bool(hard)
        self.depth_decay = float(depth_decay)

        self.hierarchies = nn.ModuleList([
            LinearTaxonStage(
                d_input=d_input,
                n_taxonomy_layers=n_taxonomy_layers,
                temperature=temperature,
                hard=hard,
                depth_decay=depth_decay,
            )
            for _ in range(n_hierarchies)
        ])

        self.gate_linear = nn.Linear(d_input, n_hierarchies)

        self.hierarchy_out_channels = LinearTaxonStage.output_channels(n_taxonomy_layers)
        self.total_out_channels = n_hierarchies * self.hierarchy_out_channels

    def _compute_gate(
        self,
        x: torch.Tensor,
        hard: bool,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Return gate probabilities [B, K]."""
        logits = self.gate_linear(x)
        gate_soft = torch.softmax(logits / self.temperature, dim=1)

        if hard:
            argmax = gate_soft.argmax(dim=1, keepdim=True)
            gate_hard = torch.zeros_like(gate_soft).scatter_(1, argmax, 1.0)
            gate = gate_hard - gate_soft.detach() + gate_soft
        else:
            gate = gate_soft

        return gate.clamp_min(eps)

    def _gate_regularization(
        self,
        gate: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gate entropy and coverage KL from [B, K]."""
        log_gate = gate.clamp_min(eps).log()
        gate_entropy = -(gate * log_gate).sum(dim=1).mean()

        marginal = gate.mean(dim=0)
        marginal = marginal / marginal.sum().clamp_min(eps)
        uniform_logp = -math.log(self.n_hierarchies)
        gate_dkl = (marginal * (marginal.clamp_min(eps).log() - uniform_logp)).sum()

        return gate_entropy, gate_dkl

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run multi-hierarchy stage.

        Returns:
            gated_output: [B, K * C_taxon]
            info: Dict with dkl, entropy, gate_dkl, gate_entropy, gate_probs.
        """
        if hard is None:
            hard = self.default_hard

        gate = self._compute_gate(x, hard=hard)
        gate_entropy, gate_dkl = self._gate_regularization(gate)

        total_entropy = x.new_zeros(())
        total_dkl = x.new_zeros(())
        all_outputs: List[torch.Tensor] = []

        for k, hierarchy in enumerate(self.hierarchies):
            out_k, regs_k = hierarchy(x, hard=hard)
            gate_k = gate[:, k : k + 1]
            all_outputs.append(out_k * gate_k)
            total_entropy = total_entropy + regs_k["entropy"]
            total_dkl = total_dkl + regs_k["dkl"]

        return torch.cat(all_outputs, dim=1), {
            "dkl": total_dkl,
            "entropy": total_entropy,
            "gate_dkl": gate_dkl,
            "gate_entropy": gate_entropy,
            "gate_probs": gate,
        }


# ── SAEBench wrapper ─────────────────────────────────────────────────────

class MultiTaxonSAE(BaseSAE):
    """SAEBench-compatible multi-taxonomy SAE.

    K independent LinearTaxonStage trees with a softmax inter-hierarchy gate.
    """

    def __init__(
        self,
        d_in: int,
        n_taxonomy_layers: int,
        n_hierarchies: int,
        model_name: str,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        temperature: float = 0.5,
        hard: bool = False,
        depth_decay: float = 0.5,
        hook_name: str | None = None,
    ):
        per_hierarchy = LinearTaxonStage.output_channels(n_taxonomy_layers)
        d_sae = n_hierarchies * per_hierarchy

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(d_in, d_sae, model_name, hook_layer, device, dtype, hook_name)

        self.n_taxonomy_layers = n_taxonomy_layers
        self.n_hierarchies = n_hierarchies
        self.temperature = temperature
        self.default_hard = hard
        self.depth_decay = depth_decay

        # Multi-taxon encoder
        self.encoder = LinearMultiTaxonStage(
            d_input=d_in,
            n_taxonomy_layers=n_taxonomy_layers,
            n_hierarchies=n_hierarchies,
            temperature=temperature,
            hard=hard,
            depth_decay=depth_decay,
        )

        # W_dec: [d_sae, d_in]
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)

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
        z, _ = self.encoder(flat)
        return z.reshape(*shape[:-1], -1)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------
    # Training helpers
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
            "model_variant": "multi_taxon_sae",
            "d_in": self.cfg.d_in,
            "d_sae": self.cfg.d_sae,
            "n_taxonomy_layers": self.n_taxonomy_layers,
            "n_hierarchies": self.n_hierarchies,
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
    ) -> "MultiTaxonSAE":
        state = torch.load(path, map_location="cpu", weights_only=False)
        args = state["args"]
        sae = cls(
            d_in=args["d_in"],
            n_taxonomy_layers=args["n_taxonomy_layers"],
            n_hierarchies=args["n_hierarchies"],
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
