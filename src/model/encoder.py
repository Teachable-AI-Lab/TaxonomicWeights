"""Taxonomy-constrained ResNet-style encoder modules.

This module contains the encoder-side building blocks used by the taxon
autoencoder:
- A pre-activation residual block (ResNet basic-block style).
- A taxonomic stage that converts stage activations into hierarchical path
  distributions through pairwise softmax.
- A ResNet-like multi-stage encoder that applies the taxonomic stage at each
  residual stage.

Regularization terms produced by each taxonomic stage:
- ``entropy``: mean per-spatial entropy of path probabilities (lower => spikier).
- ``dkl``: KL(path-marginal || uniform), where the path marginal is aggregated
  across batch and spatial locations (coverage/diversity term).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def resolve_resnet_stage_blocks(
    resnet_variant: str | int | None = "18",
    stage_blocks: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Resolve stage block counts from a ResNet variant.

    Args:
        resnet_variant: ResNet variant identifier (e.g., ``"18"``, ``"34"``).
        stage_blocks: Explicit block counts. If provided, it is returned directly.

    Returns:
        Tuple of block counts per stage.
    """
    if stage_blocks is not None:
        return tuple(int(v) for v in stage_blocks)

    presets = {
        "18": (2, 2, 2, 2),
        "34": (3, 4, 6, 3),
        "50": (3, 4, 6, 3),
        "101": (3, 4, 23, 3),
    }

    key = "18" if resnet_variant is None else str(resnet_variant).lower().replace("resnet", "").strip()
    if key not in presets:
        raise ValueError(f"Unsupported resnet_variant={resnet_variant}. Supported: {sorted(presets)}")
    return presets[key]


class ResidualConvBlock(nn.Module):
    """Pre-activation residual block used inside taxonomic encoder stages.

    Structure: ``BN -> Conv -> ReLU -> BN -> Conv + skip``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if (in_channels != out_channels) or (stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class TaxonResNetStage(nn.Module):
    """One ResNet stage with taxonomy-constrained hierarchical routing.

    The stage first computes residual features, then interprets them as logits for
    a binary-tree taxonomy per depth, with channel counts ``[2, 4, 8, ...]``.
    """

    def __init__(
        self,
        in_channels: int,
        n_taxonomy_layers: int,
        n_blocks: int,
        stride: int = 1,
        kernel_size: int = 3,
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

        self.layer_channels: List[int] = [1 << (i + 1) for i in range(self.n_taxonomy_layers)]
        self.total_out_channels = self.output_channels(self.n_taxonomy_layers)

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

    @staticmethod
    def output_channels(n_taxonomy_layers: int) -> int:
        """Return total output channels for depths ``[2, 4, ..., 2^L]``."""
        return (1 << (n_taxonomy_layers + 1)) - 2

    def _stage_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        for block in self.blocks:
            feat = block(feat)
        return feat

    def _taxon_logits_per_depth(self, x: torch.Tensor) -> List[torch.Tensor]:
        stage_logits = self._stage_features(x)
        if stage_logits.shape[1] != self.total_out_channels:
            raise ValueError(
                f"Stage output channel mismatch: got {stage_logits.shape[1]}, expected {self.total_out_channels}."
            )
        return list(torch.split(stage_logits, self.layer_channels, dim=1))

    def _pairwise_log_softmax(
        self,
        logits: torch.Tensor,
        tau: Optional[float] = None,
        hard: Optional[bool] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Apply pairwise (sibling) softmax and return log-conditionals.

        When ``hard=True``, a straight-through one-hot estimator is used.
        """
        if tau is None:
            tau = self.temperature
        if hard is None:
            hard = self.default_hard

        if logits.shape[1] % 2 != 0:
            raise ValueError(f"Expected even channel count for pairwise softmax, got {logits.shape[1]}")

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
        """Compute local entropy and aggregated coverage KL for one depth."""
        local_entropy = -(prob * logp).sum(dim=1).mean()

        marginal = prob.mean(dim=(0, 2, 3))
        marginal = marginal / marginal.sum().clamp_min(eps)
        uniform_logp = -math.log(prob.shape[1])
        coverage_kl = (marginal * (marginal.clamp_min(eps).log() - uniform_logp)).sum()

        return local_entropy, coverage_kl

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run stage forward pass.

        Returns:
            - Concatenated taxonomy-gated stage activations.
            - Concatenated log path probabilities.
            - Dict with ``{"entropy", "dkl"}``.
        """
        outputs: List[torch.Tensor] = []
        logps: List[torch.Tensor] = []
        prev: Optional[torch.Tensor] = None

        total_entropy = x.new_zeros(())
        total_dkl = x.new_zeros(())

        for depth_idx, logits in enumerate(self._taxon_logits_per_depth(x)):
            log_cond = self._pairwise_log_softmax(logits, tau=self.temperature, hard=hard)
            logp = log_cond if prev is None else log_cond + prev.repeat_interleave(2, dim=1)
            prob = logp.exp()
            out = logits * prob

            entropy_i, dkl_i = self._regularization_terms(prob, logp)
            depth_weight = self.depth_decay ** depth_idx
            total_entropy = total_entropy + depth_weight * entropy_i
            total_dkl = total_dkl + depth_weight * dkl_i

            outputs.append(out)
            logps.append(logp)
            prev = logp

        return (
            torch.cat(outputs, dim=1),
            torch.cat(logps, dim=1),
            {"entropy": total_entropy, "dkl": total_dkl},
        )

    @torch.no_grad()
    def evaluate_path_encoding(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> Dict[str, object]:
        """Validate path-mass constraints for this stage on input ``x``."""
        out_all, logp_all, regs = self.forward(x, hard=hard)
        out_splits = torch.split(out_all, self.layer_channels, dim=1)
        logp_splits = torch.split(logp_all, self.layer_channels, dim=1)

        depth_reports: List[Dict[str, object]] = []
        prev_logp: Optional[torch.Tensor] = None
        all_ok = True

        for depth_idx, (out, logp) in enumerate(zip(out_splits, logp_splits)):
            probs = logp.exp()
            mass_err = (probs.sum(dim=1) - 1.0).abs().max().item()
            mass_ok = mass_err <= (atol + rtol)

            if prev_logp is None:
                parent_err = 0.0
                parent_ok = True
            else:
                pair_mass = probs.view(probs.shape[0], -1, 2, probs.shape[2], probs.shape[3]).sum(dim=2)
                parent = prev_logp.exp()
                parent_err = (pair_mass - parent).abs().max().item()
                parent_ok = parent_err <= (atol + rtol)

            finite_ok = bool(torch.isfinite(out).all().item() and torch.isfinite(logp).all().item())
            depth_ok = bool(mass_ok and parent_ok and finite_ok)
            all_ok = all_ok and depth_ok

            depth_reports.append(
                {
                    "depth": depth_idx + 1,
                    "channels": int(out.shape[1]),
                    "mass_error": float(mass_err),
                    "parent_error": float(parent_err),
                    "finite_ok": finite_ok,
                    "ok": depth_ok,
                }
            )
            prev_logp = logp

        return {
            "ok": all_ok,
            "n_depths": self.n_taxonomy_layers,
            "layer_channels": list(self.layer_channels),
            "dkl": float(regs["dkl"].detach().cpu()),
            "entropy": float(regs["entropy"].detach().cpu()),
            "depth_reports": depth_reports,
        }


class TaxonResNetEncoder(nn.Module):
    """ResNet-like encoder whose stage activations are taxonomy-constrained.

    The structure mirrors standard ResNet staging:
    ``stem -> stage1 -> stage2 -> stage3 -> stage4``
    with stage block counts from a ResNet preset (e.g., 18 => ``[2,2,2,2]``).

    Each stage output is not a plain feature map; it is transformed into
    taxonomy-gated activations via :class:`TaxonResNetStage`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_taxonomy_layers: Sequence[int] = (5, 6, 7, 8),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        temperature: float = 1.0,
        hard: bool = False,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()

        resolved_stage_blocks = resolve_resnet_stage_blocks(
            resnet_variant=resnet_variant,
            stage_blocks=stage_blocks,
        )

        if not (len(resolved_stage_blocks) == len(stage_taxonomy_layers) == len(stage_strides)):
            raise ValueError(
                "stage_blocks, stage_taxonomy_layers, and stage_strides must have equal length. "
                f"Got {len(resolved_stage_blocks)}, {len(stage_taxonomy_layers)}, {len(stage_strides)}"
            )

        self.in_channels = in_channels
        self.resnet_variant = str(resnet_variant)
        self.stage_blocks = tuple(int(v) for v in resolved_stage_blocks)
        self.stage_taxonomy_layers = tuple(int(v) for v in stage_taxonomy_layers)
        self.stage_strides = tuple(int(v) for v in stage_strides)
        self.temperature = float(temperature)
        self.default_hard = bool(hard)
        self.use_stem = bool(use_stem)
        self.stem_stride = int(stem_stride)
        self.use_stem_maxpool = bool(use_stem_maxpool)

        if self.use_stem:
            stem_ops: List[nn.Module] = [
                nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=self.stem_stride, padding=3, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            ]
            if self.use_stem_maxpool:
                stem_ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.stem = nn.Sequential(*stem_ops)
            current_channels = stem_channels
            self.stem_total_stride = self.stem_stride * (2 if self.use_stem_maxpool else 1)
        else:
            self.stem = nn.Identity()
            current_channels = in_channels
            self.stem_total_stride = 1

        self.stage_input_channels: List[int] = []
        self.taxon_stages = nn.ModuleList()

        for blocks, n_layers, stride in zip(self.stage_blocks, self.stage_taxonomy_layers, self.stage_strides):
            self.stage_input_channels.append(current_channels)
            stage = TaxonResNetStage(
                in_channels=current_channels,
                n_taxonomy_layers=n_layers,
                n_blocks=blocks,
                stride=stride,
                kernel_size=kernel_size,
                temperature=self.temperature,
                hard=self.default_hard,
                depth_decay=depth_decay,
            )
            self.taxon_stages.append(stage)
            current_channels = stage.total_out_channels

        self.final_channels = current_channels

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode input image into taxonomy-aware latent features."""
        if hard is None:
            hard = self.default_hard

        details: Dict[str, object] = {
            "input_shape": tuple(x.shape),
            "shape_trace": [],
            "stages": [],
            "resnet_variant": self.resnet_variant,
            "stage_blocks": self.stage_blocks,
            "stage_taxonomy_layers": self.stage_taxonomy_layers,
            "stage_strides": self.stage_strides,
        }

        x = self.stem(x)
        if return_details:
            details["shape_trace"].append(("stem", tuple(x.shape)))

        total_dkl = x.new_zeros(())
        total_entropy = x.new_zeros(())

        for stage_idx, stage in enumerate(self.taxon_stages, start=1):
            x, stage_logp, regs = stage(x, hard=hard)
            total_dkl = total_dkl + regs["dkl"]
            total_entropy = total_entropy + regs["entropy"]

            if return_details:
                details["shape_trace"].append((f"stage{stage_idx}", tuple(x.shape)))
                details["stages"].append(
                    {
                        "name": f"stage{stage_idx}",
                        "output": x,
                        "logp": stage_logp,
                        "output_shape": tuple(x.shape),
                        "logp_shape": tuple(stage_logp.shape),
                        "layer_channels": list(stage.layer_channels),
                        "taxonomy_depth": stage.n_taxonomy_layers,
                        "n_blocks": stage.n_blocks,
                        "stride": stage.stride,
                        "dkl": regs["dkl"],
                        "entropy": regs["entropy"],
                    }
                )

        details["latent_shape"] = tuple(x.shape)
        details["dkl"] = total_dkl
        details["entropy"] = total_entropy

        return x, details


class MultiTaxonResNetStage(nn.Module):
    """One ResNet stage with K independent taxonomy hierarchies and an inter-hierarchy gate.

    Each hierarchy independently routes input features through a binary-tree taxonomy
    of depth ``n_taxonomy_layers`` (exactly like :class:`TaxonResNetStage`).  A
    lightweight inter-hierarchy gate — a K-way per-spatial-location softmax — then
    scales each hierarchy's output, encouraging only one hierarchy to dominate at any
    given location::

        gate  ∈ R^{B × K × H_out × W_out}   (K-way softmax over dim=1)
        out_k ∈ R^{B × C_taxon × H_out × W_out}
        output = concat(gate_k * out_k  for k = 1..K)
               ∈ R^{B × (K · C_taxon) × H_out × W_out}

    Sparse routing is enforced at two levels:

    * **Within-hierarchy**: one active path per spatial location via the pairwise
      softmax tree (identical to :class:`TaxonResNetStage`).
    * **Across-hierarchies**: the inter-hierarchy gate pushes one hierarchy to
      dominate per spatial location.

    Regularization terms returned by :meth:`forward`:

    * ``entropy``     — mean per-spatial path entropy, summed over K hierarchies.
    * ``dkl``         — path-marginal KL, summed over K hierarchies.
    * ``gate_entropy``— per-spatial entropy of the inter-hierarchy gate (lower → spikier).
    * ``gate_dkl``    — KL(gate-marginal || uniform over K).
    * ``gate_probs``  — the raw gate tensor ``[B, K, H, W]`` for analysis.
    """

    def __init__(
        self,
        in_channels: int,
        n_taxonomy_layers: int,
        n_blocks: int,
        n_hierarchies: int = 3,
        stride: int = 1,
        kernel_size: int = 3,
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()
        if n_hierarchies < 1:
            raise ValueError(f"n_hierarchies must be >= 1, got {n_hierarchies}")
        if n_taxonomy_layers < 1:
            raise ValueError(f"n_taxonomy_layers must be >= 1, got {n_taxonomy_layers}")

        self.in_channels = in_channels
        self.n_taxonomy_layers = n_taxonomy_layers
        self.n_hierarchies = n_hierarchies
        self.n_blocks = n_blocks
        self.stride = stride
        self.temperature = float(temperature)
        self.default_hard = bool(hard)
        self.depth_decay = float(depth_decay)

        # K independent taxonomy stages — each has its own residual blocks and routing.
        self.hierarchies = nn.ModuleList([
            TaxonResNetStage(
                in_channels=in_channels,
                n_taxonomy_layers=n_taxonomy_layers,
                n_blocks=n_blocks,
                stride=stride,
                kernel_size=kernel_size,
                temperature=temperature,
                hard=hard,
                depth_decay=depth_decay,
            )
            for _ in range(n_hierarchies)
        ])

        # Inter-hierarchy gate: lightweight strided 1×1 conv → BN → K-way softmax.
        # stride matches the encoder-stage downsampling so gate and hierarchy outputs
        # share the same spatial resolution H_out × W_out.
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, n_hierarchies, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(n_hierarchies),
        )

        self.hierarchy_out_channels: int = TaxonResNetStage.output_channels(n_taxonomy_layers)
        self.layer_channels: List[int] = self.hierarchies[0].layer_channels  # alias for compat
        self.total_out_channels: int = n_hierarchies * self.hierarchy_out_channels

    def _compute_gate(
        self,
        x: torch.Tensor,
        hard: bool,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Return inter-hierarchy gate probabilities ``[B, K, H_out, W_out]``."""
        logits = self.gate_conv(x)                                        # [B, K, H, W]
        gate_soft = torch.softmax(logits / self.temperature, dim=1)       # [B, K, H, W]
        if hard:
            argmax = gate_soft.argmax(dim=1, keepdim=True)
            gate_hard = torch.zeros_like(logits).scatter_(1, argmax, 1.0)
            gate = gate_hard - gate_soft.detach() + gate_soft
        else:
            gate = gate_soft
        return gate.clamp_min(eps)

    def _gate_regularization(
        self,
        gate: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gate entropy and coverage KL from ``gate [B, K, H, W]``.

        Returns ``(gate_entropy, gate_dkl)`` — lower gate_entropy means one
        hierarchy dominates (spiky); positive gate_dkl means the marginal
        deviates from uniform.
        """
        log_gate = gate.log()
        gate_entropy = -(gate * log_gate).sum(dim=1).mean()

        marginal = gate.mean(dim=(0, 2, 3))                               # [K]
        marginal = marginal / marginal.sum().clamp_min(eps)
        uniform_logp = -math.log(self.n_hierarchies)
        gate_dkl = (marginal * (marginal.clamp_min(eps).log() - uniform_logp)).sum()

        return gate_entropy, gate_dkl

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run multi-hierarchy stage forward pass.

        Returns:
            - Concatenated gate-scaled outputs ``[B, K·C_taxon, H, W]``.
            - Concatenated log path probabilities ``[B, K·C_logp, H, W]``.
            - Regs dict: ``entropy``, ``dkl``, ``gate_entropy``, ``gate_dkl``,
              ``gate_probs``.
        """
        if hard is None:
            hard = self.default_hard

        gate = self._compute_gate(x, hard=hard)          # [B, K, H_out, W_out]
        gate_entropy, gate_dkl = self._gate_regularization(gate)

        total_entropy = x.new_zeros(())
        total_dkl     = x.new_zeros(())
        all_outputs:   List[torch.Tensor] = []
        all_logps:     List[torch.Tensor] = []

        for k, hierarchy in enumerate(self.hierarchies):
            out_k, logp_k, regs_k = hierarchy(x, hard=hard)     # [B, C, H, W]
            gate_k = gate[:, k : k + 1, :, :]                   # [B, 1, H, W]
            all_outputs.append(out_k * gate_k)
            all_logps.append(logp_k)
            total_entropy = total_entropy + regs_k["entropy"]
            total_dkl     = total_dkl     + regs_k["dkl"]

        return (
            torch.cat(all_outputs, dim=1),
            torch.cat(all_logps,   dim=1),
            {
                "entropy":      total_entropy,
                "dkl":          total_dkl,
                "gate_entropy": gate_entropy,
                "gate_dkl":     gate_dkl,
                "gate_probs":   gate,   # [B, K, H, W] — kept for analysis
            },
        )


class MultiTaxonResNetEncoder(nn.Module):
    """ResNet-like encoder with K independent taxonomy hierarchies per stage.

    Each stage uses :class:`MultiTaxonResNetStage`.  Because each stage outputs
    ``n_hierarchies × C_taxon`` channels, intermediate feature sizes grow by that
    factor::

        stem(stem_ch) → stage1(K·C1) → stage2(K·C2) → stage3(K·C3) → stage4(K·C4)

    The decoder receives ``stage_input_channels`` that already account for this
    inflation, so :class:`~.decoder.TaxonResNetDecoder` can be used unchanged.

    Parameters
    ----------
    n_hierarchies:
        Number of independent taxonomy trees per stage.
    stage_taxonomy_layers:
        Depth of the binary-tree taxonomy per stage (shared across all hierarchies
        within that stage).
    All other parameters mirror :class:`TaxonResNetEncoder`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_taxonomy_layers: Sequence[int] = (5, 6, 7, 8),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        n_hierarchies: int = 3,
        temperature: float = 1.0,
        hard: bool = False,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()

        resolved_stage_blocks = resolve_resnet_stage_blocks(
            resnet_variant=resnet_variant,
            stage_blocks=stage_blocks,
        )

        if not (len(resolved_stage_blocks) == len(stage_taxonomy_layers) == len(stage_strides)):
            raise ValueError(
                "stage_blocks, stage_taxonomy_layers, and stage_strides must have equal length. "
                f"Got {len(resolved_stage_blocks)}, {len(stage_taxonomy_layers)}, {len(stage_strides)}"
            )

        self.in_channels = in_channels
        self.resnet_variant = str(resnet_variant)
        self.stage_blocks = tuple(int(v) for v in resolved_stage_blocks)
        self.stage_taxonomy_layers = tuple(int(v) for v in stage_taxonomy_layers)
        self.stage_strides = tuple(int(v) for v in stage_strides)
        self.n_hierarchies = int(n_hierarchies)
        self.temperature = float(temperature)
        self.default_hard = bool(hard)
        self.use_stem = bool(use_stem)
        self.stem_stride = int(stem_stride)
        self.use_stem_maxpool = bool(use_stem_maxpool)

        # ── stem ──────────────────────────────────────────────────────────────
        if self.use_stem:
            stem_ops: List[nn.Module] = [
                nn.Conv2d(in_channels, stem_channels, kernel_size=7,
                          stride=self.stem_stride, padding=3, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            ]
            if self.use_stem_maxpool:
                stem_ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.stem = nn.Sequential(*stem_ops)
            current_channels = stem_channels
            self.stem_total_stride = self.stem_stride * (2 if self.use_stem_maxpool else 1)
        else:
            self.stem = nn.Identity()
            current_channels = in_channels
            self.stem_total_stride = 1

        # ── multi-hierarchy stages ─────────────────────────────────────────────
        self.stage_input_channels: List[int] = []
        self.multi_taxon_stages = nn.ModuleList()

        for blocks, n_layers, stride in zip(
            self.stage_blocks, self.stage_taxonomy_layers, self.stage_strides
        ):
            self.stage_input_channels.append(current_channels)
            stage = MultiTaxonResNetStage(
                in_channels=current_channels,
                n_taxonomy_layers=n_layers,
                n_blocks=blocks,
                n_hierarchies=self.n_hierarchies,
                stride=stride,
                kernel_size=kernel_size,
                temperature=self.temperature,
                hard=self.default_hard,
                depth_decay=depth_decay,
            )
            self.multi_taxon_stages.append(stage)
            current_channels = stage.total_out_channels   # K * hierarchy_out_ch

        self.final_channels = current_channels

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode input image into multi-hierarchy taxonomy-aware latent features."""
        if hard is None:
            hard = self.default_hard

        details: Dict[str, object] = {
            "input_shape": tuple(x.shape),
            "shape_trace": [],
            "stages": [],
            "resnet_variant": self.resnet_variant,
            "stage_blocks": self.stage_blocks,
            "stage_taxonomy_layers": self.stage_taxonomy_layers,
            "stage_strides": self.stage_strides,
            "n_hierarchies": self.n_hierarchies,
        }

        x = self.stem(x)
        if return_details:
            details["shape_trace"].append(("stem", tuple(x.shape)))

        total_dkl          = x.new_zeros(())
        total_entropy      = x.new_zeros(())
        total_gate_dkl     = x.new_zeros(())
        total_gate_entropy = x.new_zeros(())

        for stage_idx, stage in enumerate(self.multi_taxon_stages, start=1):
            x, stage_logp, regs = stage(x, hard=hard)
            total_dkl          = total_dkl          + regs["dkl"]
            total_entropy      = total_entropy      + regs["entropy"]
            total_gate_dkl     = total_gate_dkl     + regs["gate_dkl"]
            total_gate_entropy = total_gate_entropy + regs["gate_entropy"]

            if return_details:
                details["shape_trace"].append((f"stage{stage_idx}", tuple(x.shape)))
                details["stages"].append({
                    "name": f"stage{stage_idx}",
                    "output": x,
                    "logp": stage_logp,
                    "output_shape": tuple(x.shape),
                    "n_hierarchies": self.n_hierarchies,
                    "hierarchy_out_channels": stage.hierarchy_out_channels,
                    "gate_probs": regs["gate_probs"],   # [B, K, H, W]
                    "dkl":          regs["dkl"],
                    "entropy":      regs["entropy"],
                    "gate_dkl":     regs["gate_dkl"],
                    "gate_entropy": regs["gate_entropy"],
                })

        details["latent_shape"]   = tuple(x.shape)
        details["dkl"]            = total_dkl
        details["entropy"]        = total_entropy
        details["gate_dkl"]       = total_gate_dkl
        details["gate_entropy"]   = total_gate_entropy

        return x, details


__all__ = [
    "resolve_resnet_stage_blocks",
    "ResidualConvBlock",
    "TaxonResNetStage",
    "TaxonResNetEncoder",
    "MultiTaxonResNetStage",
    "MultiTaxonResNetEncoder",
]
