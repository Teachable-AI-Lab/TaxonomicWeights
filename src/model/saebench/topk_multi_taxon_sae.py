"""SAEBench-compatible TopK Multi-Taxon SAE.

K independent TopK taxonomy hierarchies with a TopK inter-hierarchy gate,
conforming to the SAEBench BaseSAE interface.

Replaces DKL regularisation with dead-node tracking + AuxK auxiliary loss
per hierarchy.  The gate uses TopK selection (gate_k) to choose active
hierarchies, mirroring the CNN :class:`TopKMultiTaxonAutoencoder`.

Reference: Gao et al., "Scaling and Evaluating Sparse Autoencoders",
arXiv:2406.04093.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from .topk_taxon_sae import TopKLinearTaxonStage


# ── Multi-hierarchy routing stage with TopK gate ─────────────────────────

class TopKLinearMultiTaxonStage(nn.Module):
    """K independent TopK taxonomy trees + TopK inter-hierarchy gate.

    Args:
        d_input:  Dimensionality of the input activation vector.
        n_taxonomy_layers:  Depth of each binary tree.
        n_hierarchies:  Number of independent trees (K).
        temperature:  Softmax temperature for routing.
        hard:  Straight-through hard routing.
        depth_decay:  Exponential weight for depth levels.
        k_aux:  Dead features per hierarchy used in aux loss.
        dead_steps:  Steps without winning before a node is flagged dead.
        gate_k:  Number of hierarchies active per token.
    """

    def __init__(
        self,
        d_input: int,
        n_taxonomy_layers: int,
        n_hierarchies: int = 4,
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
        k_aux: Optional[int] = None,
        dead_steps: int = 2000,
        gate_k: int = 1,
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
        self.gate_k = min(gate_k, n_hierarchies)

        self.hierarchies = nn.ModuleList([
            TopKLinearTaxonStage(
                d_input=d_input,
                n_taxonomy_layers=n_taxonomy_layers,
                temperature=temperature,
                hard=hard,
                depth_decay=depth_decay,
                k_aux=k_aux,
                dead_steps=dead_steps,
            )
            for _ in range(n_hierarchies)
        ])

        self.gate_linear = nn.Linear(d_input, n_hierarchies)

        self.hierarchy_out_channels = TopKLinearTaxonStage.output_channels(n_taxonomy_layers)
        self.total_out_channels = n_hierarchies * self.hierarchy_out_channels

    def _compute_gate(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """TopK gate: select gate_k hierarchies per token [B, K]."""
        logits = self.gate_linear(x)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        _, idx = logits.topk(self.gate_k, dim=-1)
        mask = torch.zeros_like(weights)
        mask.scatter_(1, idx, 1.0)
        # Straight-through: gate = mask * softmax
        gate = mask * weights
        return gate

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if hard is None:
            hard = self.default_hard

        gate = self._compute_gate(x)  # [B, K]

        total_dead = x.new_zeros(())
        all_outputs: List[torch.Tensor] = []

        for k, hierarchy in enumerate(self.hierarchies):
            out_k, info_k = hierarchy(x, hard=hard)
            gate_k = gate[:, k : k + 1]
            all_outputs.append(out_k * gate_k)
            total_dead = total_dead + info_k["dead_frac"]

        return torch.cat(all_outputs, dim=1), {
            "dead_frac": total_dead,
            "gate_probs": gate,
        }


# ── SAEBench wrapper ─────────────────────────────────────────────────────

class TopKMultiTaxonSAE(nn.Module):
    """SAEBench-compatible multi-taxonomy SAE with TopK sparsity + AuxK loss."""

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
        k_aux: Optional[int] = None,
        dead_steps: int = 2000,
        gate_k: int = 1,
        hook_name: str | None = None,
    ):
        per_hierarchy = TopKLinearTaxonStage.output_channels(n_taxonomy_layers)
        d_sae = n_hierarchies * per_hierarchy

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__()

        self.n_taxonomy_layers = n_taxonomy_layers
        self.n_hierarchies = n_hierarchies
        self.temperature = temperature
        self.default_hard = hard
        self.depth_decay = depth_decay
        self.k_aux = k_aux
        self.dead_steps = dead_steps
        self.gate_k = gate_k

        self.encoder = TopKLinearMultiTaxonStage(
            d_input=d_in,
            n_taxonomy_layers=n_taxonomy_layers,
            n_hierarchies=n_hierarchies,
            temperature=temperature,
            hard=hard,
            depth_decay=depth_decay,
            k_aux=k_aux,
            dead_steps=dead_steps,
            gate_k=gate_k,
        )

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

    def compute_auxk_loss(
        self, x: torch.Tensor, x_recon: torch.Tensor
    ) -> torch.Tensor:
        flat_x = x.reshape(-1, x.shape[-1]) - self.b_dec
        flat_recon = x_recon.reshape(-1, x_recon.shape[-1]) - self.b_dec
        total = flat_x.new_zeros(())
        for hierarchy in self.encoder.hierarchies:
            total = total + hierarchy.compute_auxk_loss(flat_x, flat_recon)
        return total

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.W_dec.data.norm(dim=1, keepdim=True)
        self.W_dec.data /= norms.clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def get_checkpoint_args(self) -> dict:
        return {
            "model_variant": "topk_multi_taxon_sae",
            "d_in": self.cfg.d_in,
            "d_sae": self.cfg.d_sae,
            "n_taxonomy_layers": self.n_taxonomy_layers,
            "n_hierarchies": self.n_hierarchies,
            "temperature": self.temperature,
            "hard": self.default_hard,
            "depth_decay": self.depth_decay,
            "k_aux": self.k_aux,
            "dead_steps": self.dead_steps,
            "gate_k": self.gate_k,
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
    ) -> "TopKMultiTaxonSAE":
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
            k_aux=args.get("k_aux"),
            dead_steps=args.get("dead_steps", 2000),
            gate_k=args.get("gate_k", 1),
            hook_name=args.get("hook_name"),
        )
        sae.load_state_dict(state["model_state"])
        sae.to(device=device, dtype=dtype)
        return sae
