"""SAEBench-compatible TopK Taxon SAE.

Replaces DKL regularisation with dead-node tracking and an AuxK auxiliary loss
that revives underused features.  Routing uses the same binary-tree
pairwise-softmax mechanism as :class:`TaxonSAE`, but sparsity is enforced
by keeping only one active path per hierarchy (k = n_taxonomy_layers).

Reference: Gao et al., "Scaling and Evaluating Sparse Autoencoders",
arXiv:2406.04093.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



# ── TopK taxonomy routing stage ──────────────────────────────────────────

class TopKLinearTaxonStage(nn.Module):
    """Linear taxonomy tree with dead-node tracking + AuxK revival.

    Same routing as :class:`LinearTaxonStage`, but instead of DKL / entropy
    regularisation, diversity is encouraged by tracking nodes that consistently
    lose their sibling competition.

    Args:
        d_input:  Dimensionality of the input activation vector.
        n_taxonomy_layers:  Depth of the binary tree (L).
        temperature:  Softmax temperature for sibling competition.
        hard:  Use straight-through hard routing.
        depth_decay:  Exponential weight decay per depth.
        k_aux:  Number of dead features used in the aux loss (default: half).
        dead_steps:  Steps without winning before a node is flagged dead.
    """

    def __init__(
        self,
        d_input: int,
        n_taxonomy_layers: int,
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
        k_aux: Optional[int] = None,
        dead_steps: int = 2000,
    ) -> None:
        super().__init__()
        if n_taxonomy_layers < 1:
            raise ValueError(f"n_taxonomy_layers must be >= 1, got {n_taxonomy_layers}")

        self.d_input = d_input
        self.n_taxonomy_layers = n_taxonomy_layers
        self.temperature = float(temperature)
        self.default_hard = bool(hard)
        self.depth_decay = float(depth_decay)
        self.dead_steps = int(dead_steps)

        self.layer_channels: List[int] = [1 << (i + 1) for i in range(n_taxonomy_layers)]
        self.total_out_channels = self.output_channels(n_taxonomy_layers)
        # Default: revive one dead path = one node per depth level
        self.k_aux = k_aux if k_aux is not None else n_taxonomy_layers

        self.linear = nn.Linear(d_input, self.total_out_channels)

        self.register_buffer(
            "_steps_since_active",
            torch.zeros(self.total_out_channels, dtype=torch.long),
        )

    @staticmethod
    def output_channels(n_taxonomy_layers: int) -> int:
        return (1 << (n_taxonomy_layers + 1)) - 2

    def _pairwise_softmax(
        self,
        logits: torch.Tensor,
        tau: float,
        hard: bool,
    ) -> torch.Tensor:
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

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if hard is None:
            hard = self.default_hard

        all_logits = self.linear(x)
        depth_logits = torch.split(all_logits, self.layer_channels, dim=1)

        outputs: List[torch.Tensor] = []
        all_probs: List[torch.Tensor] = []
        prev_gate: Optional[torch.Tensor] = None

        for depth_idx, logits in enumerate(depth_logits):
            cond = self._pairwise_softmax(logits, self.temperature, hard)
            gate_prob = cond if prev_gate is None else cond * prev_gate.repeat_interleave(2, dim=1)

            out = F.relu(logits) * gate_prob
            outputs.append(out)
            all_probs.append(gate_prob)
            prev_gate = gate_prob

        # Dead-node tracking
        if self.training:
            all_prob = torch.cat(all_probs, dim=1)
            any_active = all_prob.amax(dim=0) > 0.5
            self._steps_since_active[any_active] = 0
            self._steps_since_active[~any_active] += 1

        dead_frac = (self._steps_since_active >= self.dead_steps).float().mean()

        return torch.cat(outputs, dim=1), {"dead_frac": dead_frac}

    def compute_auxk_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
    ) -> torch.Tensor:
        """AuxK loss: train dead features to model reconstruction error."""
        dead_mask = self._steps_since_active >= self.dead_steps
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return x.new_zeros(())

        all_logits = self.linear(x - x_recon.detach() + x)
        dead_float = dead_mask.to(dtype=x.dtype).unsqueeze(0)
        dead_vals = F.relu(all_logits) * dead_float

        k_aux_eff = min(self.k_aux, n_dead)
        _, aux_idx = dead_vals.topk(k_aux_eff, dim=-1)
        aux_mask = torch.zeros_like(dead_vals)
        aux_mask.scatter_(1, aux_idx, 1.0)

        error = (x - x_recon).detach()
        dead_signal = (dead_vals * aux_mask).sum(dim=1, keepdim=True)
        error_norm = error.norm(dim=1, keepdim=True)
        return F.mse_loss(dead_signal, error_norm)


# ── SAEBench wrapper ─────────────────────────────────────────────────────

class TopKTaxonSAE(nn.Module):
    """SAEBench-compatible linear taxonomy SAE with TopK sparsity + AuxK loss."""

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
        k_aux: Optional[int] = None,
        dead_steps: int = 2000,
        hook_name: str | None = None,
    ):
        d_sae = TopKLinearTaxonStage.output_channels(n_taxonomy_layers)
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__()

        self.n_taxonomy_layers = n_taxonomy_layers
        self.temperature = temperature
        self.default_hard = hard
        self.depth_decay = depth_decay
        self.k_aux = k_aux
        self.dead_steps = dead_steps

        self.encoder = TopKLinearTaxonStage(
            d_input=d_in,
            n_taxonomy_layers=n_taxonomy_layers,
            temperature=temperature,
            hard=hard,
            depth_decay=depth_decay,
            k_aux=k_aux,
            dead_steps=dead_steps,
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
        return self.encoder.compute_auxk_loss(flat_x, flat_recon)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        norms = self.W_dec.data.norm(dim=1, keepdim=True)
        self.W_dec.data /= norms.clamp(min=1e-8)

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def get_checkpoint_args(self) -> dict:
        return {
            "model_variant": "topk_taxon_sae",
            "d_in": self.cfg.d_in,
            "d_sae": self.cfg.d_sae,
            "n_taxonomy_layers": self.n_taxonomy_layers,
            "temperature": self.temperature,
            "hard": self.default_hard,
            "depth_decay": self.depth_decay,
            "k_aux": self.k_aux,
            "dead_steps": self.dead_steps,
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
    ) -> "TopKTaxonSAE":
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
            k_aux=args.get("k_aux"),
            dead_steps=args.get("dead_steps", 2000),
            hook_name=args.get("hook_name"),
        )
        sae.load_state_dict(state["model_state"])
        sae.to(device=device, dtype=dtype)
        return sae
