#!/usr/bin/env python3
"""Hierarchy exploration analysis for taxonomy autoencoders.

Measures how uniformly each taxonomy tree's branches are activated across
the validation set.  Ideal behaviour: all 2^(d+1) branches at depth d are
equally likely → maximal entropy.

Supports all variants:
  taxon, topk_taxon, bias_taxon
  multi_taxon, topk_multi_taxon, bias_multi_taxon

Per-depth statistics (for each encoder stage / hierarchy × depth):
  mean_probs   : array of length 2^(d+1), mean activation probability of each branch
  entropy_bits : H = −Σ p·log2(p)
  relative_H   : entropy_bits / (d+1)  — 1.0 = perfectly uniform
  effective_n  : 2^entropy_bits   (effective number of active branches)
  effective_frac: effective_n / 2^(d+1)
  dead_frac    : fraction of branches with mean prob < 0.1/2^(d+1)
  cv           : std(mean_probs) / mean(mean_probs)  (coefficient of variation)
  dominant_ratio: max(mean_probs) × 2^(d+1)  (>1 = one branch over-used)

For multi-taxon additionally: gate utilisation per hierarchy per stage.

Outputs (under <analysis_save_dir>/hierarchy_exploration/):
  hierarchy_exploration.csv
  hierarchy_exploration.png         — summary heatmap / bar charts
  hierarchy_exploration_detail.png  — sorted branch-prob curves (deepest level)

Usage (from TaxonomicWeights/):
  python -m src.analyze.hierarchy_exploration --config <path/to/config.json>
  python -m src.analyze.hierarchy_exploration --config <path> \\
      --checkpoint <path/to/best.pt> --num-batches 50
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.model.cnn.taxon.taxon_ae import TaxonAutoencoder
from src.model.cnn.taxon.topk_taxon_ae import TopKTaxonAutoencoder
from src.model.cnn.taxon.bias_taxon_ae import BiasTaxonAutoencoder
from src.model.cnn.taxon.multi_taxon_ae import MultiTaxonAutoencoder
from src.model.cnn.taxon.topk_multi_taxon_ae import TopKMultiTaxonAutoencoder
from src.model.cnn.taxon.bias_multi_taxon_ae import BiasMultiTaxonAutoencoder
from src.utils.dataloader import CelebAHQLoader


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hierarchy exploration analysis for taxonomy autoencoders"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path (otherwise from config)")
    p.add_argument("--num-batches", type=int, default=50,
                   help="Validation batches to evaluate (default: 50)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


# ── Config / path helpers ──────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_checkpoint(config: dict, override: Optional[str]) -> str:
    if override:
        return override
    ckpt = config.get("analysis", {}).get("checkpoint_path")
    if ckpt:
        return ckpt
    output_dir = config.get("output", {}).get("output_dir", "")
    exp_name   = config.get("experiment_name", "")
    if output_dir and exp_name:
        return str(Path(output_dir).parent / exp_name / "checkpoints" / "best.pt")
    raise ValueError(
        "Cannot resolve checkpoint. Pass --checkpoint or set "
        "config['analysis']['checkpoint_path']."
    )


def resolve_save_dir(config: dict) -> Path:
    asd = config.get("output", {}).get("analysis_save_dir")
    if asd:
        return Path(asd) / "hierarchy_exploration"
    output_dir = config.get("output", {}).get("output_dir", "")
    exp_name   = config.get("experiment_name", "")
    if output_dir and exp_name:
        return Path(output_dir).parent / exp_name / "analysis" / "hierarchy_exploration"
    raise ValueError("Cannot resolve save dir from config.")


# ── Model loading ──────────────────────────────────────────────────────────────

def _detect_variant(mc: dict, ckpt_args: dict, state_dict: dict) -> str:
    """Return one of: taxon, topk_taxon, bias_taxon,
                       multi_taxon, topk_multi_taxon, bias_multi_taxon"""
    is_multi = "n_hierarchies" in mc or "n_hierarchies" in ckpt_args
    for src in (mc, ckpt_args):
        if "bias_update_rate" in src or "bias_ema_decay" in src:
            return "bias_multi_taxon" if is_multi else "bias_taxon"
        if "k_aux" in src or "dead_steps" in src:
            return "topk_multi_taxon" if is_multi else "topk_taxon"
    if is_multi:
        return "multi_taxon"
    # Check state dict for topk / bias hints
    for k in state_dict:
        if "_steps_since_active" in k:
            return "topk_taxon"
        if "_bias" in k and "taxon_stages" in k:
            return "bias_taxon"
    return "taxon"


def _common_single_kw(merged: dict) -> dict:
    return dict(
        in_channels           = merged.get("in_channels", 3),
        resnet_variant        = merged.get("resnet_variant", "18"),
        stage_taxonomy_layers = tuple(merged.get("stage_taxonomy_layers", [5, 6, 7, 8])),
        stage_strides         = tuple(merged.get("stage_strides", [1, 2, 2, 2])),
        stage_blocks          = merged.get("stage_blocks", None),
        kernel_size           = merged.get("kernel_size", 3),
        use_stem              = merged.get("use_stem", True),
        stem_channels         = merged.get("stem_channels", 64),
        stem_stride           = merged.get("stem_stride", 2),
        use_stem_maxpool      = merged.get("use_stem_maxpool", True),
        output_activation     = merged.get("output_activation", "none"),
        depth_decay           = merged.get("depth_decay", 0.5),
        temperature           = merged.get("temperature", 1.0),
        hard                  = merged.get("hard", False),
    )


def load_model(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
) -> Tuple[nn.Module, str]:
    """Load any taxonomy model variant; return (model, variant_name)."""
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state  = ckpt.get("model_state", ckpt)
    ca     = ckpt.get("args", {})
    mc     = config.get("model", {})
    merged = {**ca, **mc}
    variant = _detect_variant(mc, ca, state)

    kw = _common_single_kw(merged)
    if variant == "taxon":
        model = TaxonAutoencoder(**kw)
    elif variant == "topk_taxon":
        model = TopKTaxonAutoencoder(
            **kw,
            k_aux      = merged.get("k_aux", None),
            dead_steps = merged.get("dead_steps", 2000),
        )
    elif variant == "bias_taxon":
        model = BiasTaxonAutoencoder(
            **kw,
            bias_update_rate = merged.get("bias_update_rate", 0.001),
            bias_ema_decay   = merged.get("bias_ema_decay", 0.99),
        )
    else:
        kw_m = {**kw, "n_hierarchies": merged.get("n_hierarchies", 3)}
        if variant == "multi_taxon":
            model = MultiTaxonAutoencoder(**kw_m)
        elif variant == "topk_multi_taxon":
            model = TopKMultiTaxonAutoencoder(
                **kw_m,
                k_aux      = merged.get("k_aux", None),
                dead_steps = merged.get("dead_steps", 2000),
                gate_k     = merged.get("gate_k", 1),
            )
        elif variant == "bias_multi_taxon":
            model = BiasMultiTaxonAutoencoder(
                **kw_m,
                bias_update_rate = merged.get("bias_update_rate", 0.001),
                bias_ema_decay   = merged.get("bias_ema_decay", 0.99),
                gate_k           = merged.get("gate_k", 1),
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, variant


# ── Statistics helpers ─────────────────────────────────────────────────────────

_EPS = 1e-12


def branch_stats(mean_probs: np.ndarray) -> dict:
    """Compute all exploration metrics from an array of mean branch probabilities."""
    n = len(mean_probs)
    assert abs(mean_probs.sum() - 1.0) < 1e-3, f"probs don't sum to 1: {mean_probs.sum()}"
    p = mean_probs.clip(_EPS, None)
    h_bits         = float(-np.sum(p * np.log2(p)))
    max_h          = math.log2(n)
    rel_h          = h_bits / max_h if max_h > 0 else 1.0
    effective_n    = 2 ** h_bits
    effective_frac = effective_n / n
    uniform_p      = 1.0 / n
    dead_frac      = float(np.mean(mean_probs < 0.1 * uniform_p))
    cv             = float(np.std(mean_probs) / (np.mean(mean_probs) + _EPS))
    dominant_ratio = float(np.max(mean_probs) * n)  # 1.0 = at uniform level
    return dict(
        n_branches     = n,
        entropy_bits   = h_bits,
        max_entropy_bits = max_h,
        relative_H     = rel_h,
        effective_n    = effective_n,
        effective_frac = effective_frac,
        dead_frac      = dead_frac,
        cv             = cv,
        dominant_ratio = dominant_ratio,
    )


# ── Data collection (forward hooks) ───────────────────────────────────────────

def collect_stats(
    model: nn.Module,
    variant: str,
    val_loader,
    device: torch.device,
    num_batches: int,
) -> dict:
    """Run inference, accumulate per-branch probabilities, return stats dict.

    Returned structure:
      Single-taxon:
        result["stages"][s]["depths"][d] = {"mean_probs": np.ndarray, **branch_stats}
      Multi-taxon:
        result["stages"][s]["hierarchies"][k]["depths"][d] = {...}
        result["stages"][s]["gate"][k] = {"mean_gate": float, "dominant_frac": float}
    """
    is_multi = variant in ("multi_taxon", "topk_multi_taxon", "bias_multi_taxon")

    if is_multi:
        return _collect_multi(model, val_loader, device, num_batches)
    else:
        return _collect_single(model, val_loader, device, num_batches)


# ─── single-taxon ─────────────────────────────────────────────────────────────

def _collect_single(model, val_loader, device, num_batches):
    taxon_stages = model.encoder.taxon_stages
    n_stages     = len(taxon_stages)

    # accum[s][d] = {"sum": tensor[2^(d+1)], "n": int}
    accum: List[Dict] = [{} for _ in range(n_stages)]

    def _make_hook(si):
        def hook(module, input, output):
            z, logps, regs = output          # logps: [B, total_ch, H, W]
            B, C, H, W = logps.shape
            n = B * H * W
            offset = 0
            for d in range(module.n_taxonomy_layers):
                width = 1 << (d + 1)
                lp = logps[:, offset : offset + width, :, :]
                p  = lp.detach().exp().sum(dim=(0, 2, 3)).cpu()  # [width]
                if d not in accum[si]:
                    accum[si][d] = {"sum": p, "n": n}
                else:
                    accum[si][d]["sum"] += p
                    accum[si][d]["n"]   += n
                offset += width
        return hook

    handles = [
        stage.register_forward_hook(_make_hook(si))
        for si, stage in enumerate(taxon_stages)
    ]

    _run_batches(model, val_loader, device, num_batches)

    for h in handles:
        h.remove()

    stages_out = []
    for si in range(n_stages):
        depths_out = {}
        for d, buf in accum[si].items():
            mean_p = (buf["sum"] / buf["n"]).numpy()
            depths_out[d] = {"mean_probs": mean_p, **branch_stats(mean_p)}
        stages_out.append({"depths": depths_out})
    return {"stages": stages_out}


# ─── multi-taxon ──────────────────────────────────────────────────────────────

def _collect_multi(model, val_loader, device, num_batches):
    taxon_stages = model.encoder.taxon_stages
    n_stages     = len(taxon_stages)

    # hier_accum[s][k][d] = {"sum": tensor, "n": int}
    hier_accum: List[List[Dict]] = []
    # gate_accum[s] = {"sum": tensor[K], "dom": tensor[K], "n": int}
    gate_accum: List[Dict] = []

    for si, stage in enumerate(taxon_stages):
        K = stage.n_hierarchies
        hier_accum.append([{} for _ in range(K)])
        gate_accum.append({"sum": torch.zeros(K), "dom": torch.zeros(K), "n": 0})

    def _make_hier_hook(si, ki):
        def hook(module, input, output):
            z, logps, regs = output
            B, C, H, W = logps.shape
            n = B * H * W
            offset = 0
            for d in range(module.n_taxonomy_layers):
                width = 1 << (d + 1)
                lp = logps[:, offset : offset + width, :, :]
                p  = lp.detach().exp().sum(dim=(0, 2, 3)).cpu()
                if d not in hier_accum[si][ki]:
                    hier_accum[si][ki][d] = {"sum": p, "n": n}
                else:
                    hier_accum[si][ki][d]["sum"] += p
                    hier_accum[si][ki][d]["n"]   += n
                offset += width
        return hook

    def _make_stage_hook(si):
        def hook(module, input, output):
            z, logp, regs = output
            gate = regs["gate_probs"]          # [B, K, H, W]
            B, K, H, W = gate.shape
            n   = B * H * W
            gp  = gate.detach().cpu()
            gsum = gp.sum(dim=(0, 2, 3))       # [K]
            dom  = (gp.argmax(dim=1)           # [B, H, W]
                       .view(-1)
                       .bincount(minlength=K)
                       .float())
            gate_accum[si]["sum"] += gsum
            gate_accum[si]["dom"] += dom
            gate_accum[si]["n"]   += n
        return hook

    handles = []
    for si, stage in enumerate(taxon_stages):
        for ki, hier in enumerate(stage.hierarchies):
            handles.append(hier.register_forward_hook(_make_hier_hook(si, ki)))
        handles.append(stage.register_forward_hook(_make_stage_hook(si)))

    _run_batches(model, val_loader, device, num_batches)

    for h in handles:
        h.remove()

    stages_out = []
    for si, stage in enumerate(taxon_stages):
        K = stage.n_hierarchies
        hierarchies_out = []
        for ki in range(K):
            depths_out = {}
            for d, buf in hier_accum[si][ki].items():
                mean_p = (buf["sum"] / buf["n"]).numpy()
                depths_out[d] = {"mean_probs": mean_p, **branch_stats(mean_p)}
            hierarchies_out.append({"depths": depths_out})

        ga = gate_accum[si]
        n  = ga["n"]
        mean_gate     = (ga["sum"] / n).numpy()        # [K], sums to ~1
        dominant_frac = (ga["dom"] / n).numpy()         # [K] fraction of (B,H,W) where k is dominant
        gate_info = {k: {"mean_gate": float(mean_gate[k]),
                         "dominant_frac": float(dominant_frac[k])}
                     for k in range(K)}

        stages_out.append({"hierarchies": hierarchies_out, "gate": gate_info})

    return {"stages": stages_out}


def _run_batches(model, val_loader, device, num_batches):
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            if i >= num_batches:
                break
            model(images.to(device))


# ── CSV output ─────────────────────────────────────────────────────────────────

def save_csv(result: dict, variant: str, save_dir: Path, exp_name: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    is_multi = "multi" in variant

    rows = []
    for si, stage in enumerate(result["stages"]):
        if is_multi:
            for ki, hier in enumerate(stage["hierarchies"]):
                for d, ds in hier["depths"].items():
                    rows.append({
                        "experiment": exp_name,
                        "variant":    variant,
                        "stage":      si,
                        "hierarchy":  ki,
                        "depth":      d,
                        "n_branches": ds["n_branches"],
                        "entropy_bits": round(ds["entropy_bits"], 5),
                        "max_entropy_bits": round(ds["max_entropy_bits"], 5),
                        "relative_H":  round(ds["relative_H"], 5),
                        "effective_n": round(ds["effective_n"], 3),
                        "effective_frac": round(ds["effective_frac"], 5),
                        "dead_frac":   round(ds["dead_frac"], 5),
                        "cv":          round(ds["cv"], 5),
                        "dominant_ratio": round(ds["dominant_ratio"], 5),
                        "gate_mean":      "",
                        "gate_dom_frac":  "",
                    })
            for k, gi in stage["gate"].items():
                rows.append({
                    "experiment": exp_name,
                    "variant":    variant,
                    "stage":      si,
                    "hierarchy":  f"gate_{k}",
                    "depth":      "",
                    "n_branches": "",
                    "entropy_bits": "",
                    "max_entropy_bits": "",
                    "relative_H":  "",
                    "effective_n": "",
                    "effective_frac": "",
                    "dead_frac":   "",
                    "cv":          "",
                    "dominant_ratio": "",
                    "gate_mean":     round(gi["mean_gate"], 5),
                    "gate_dom_frac": round(gi["dominant_frac"], 5),
                })
        else:
            for d, ds in stage["depths"].items():
                rows.append({
                    "experiment": exp_name,
                    "variant":    variant,
                    "stage":      si,
                    "hierarchy":  "",
                    "depth":      d,
                    "n_branches": ds["n_branches"],
                    "entropy_bits": round(ds["entropy_bits"], 5),
                    "max_entropy_bits": round(ds["max_entropy_bits"], 5),
                    "relative_H":  round(ds["relative_H"], 5),
                    "effective_n": round(ds["effective_n"], 3),
                    "effective_frac": round(ds["effective_frac"], 5),
                    "dead_frac":   round(ds["dead_frac"], 5),
                    "cv":          round(ds["cv"], 5),
                    "dominant_ratio": round(ds["dominant_ratio"], 5),
                    "gate_mean":      "",
                    "gate_dom_frac":  "",
                })

    csv_path = save_dir / "hierarchy_exploration.csv"
    fields = [
        "experiment", "variant", "stage", "hierarchy", "depth",
        "n_branches", "entropy_bits", "max_entropy_bits", "relative_H",
        "effective_n", "effective_frac", "dead_frac", "cv", "dominant_ratio",
        "gate_mean", "gate_dom_frac",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV:  {csv_path}")


# ── Visualisation ──────────────────────────────────────────────────────────────

_CMAP_RH  = "RdYlGn"   # relative entropy: red=bad, green=good
_CMAP_DEAD = "YlOrRd"  # dead frac: yellow=few dead, red=many dead


def save_plots(
    result: dict,
    variant: str,
    save_dir: Path,
    exp_name: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    is_multi = "multi" in variant

    if is_multi:
        _plot_multi(result, variant, save_dir, exp_name)
    else:
        _plot_single(result, variant, save_dir, exp_name)


def _plot_single(result, variant, save_dir, exp_name):
    stages  = result["stages"]
    n_s     = len(stages)
    max_d   = max(max(s["depths"].keys()) for s in stages)

    # ── Figure 1: summary heatmaps ────────────────────────────────────────────
    fig, axes = plt.subplots(2, n_s, figsize=(4 * n_s + 1, 8), squeeze=False)

    for si, stage in enumerate(stages):
        depths = stage["depths"]
        max_depth = max(depths.keys())
        n_depths  = max_depth + 1

        # Row 0: relative_H bar chart
        ax = axes[0, si]
        xs = list(range(n_depths))
        hs = [depths[d]["relative_H"] if d in depths else 0.0 for d in xs]
        colours = [plt.cm.RdYlGn(h) for h in hs]
        bars = ax.bar(xs, hs, color=colours, edgecolor="black", linewidth=0.7)
        ax.axhline(1.0, color="green", linestyle="--", linewidth=1.2, label="Uniform")
        ax.set_ylim(0, 1.15)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"d={d}\n({1<<(d+1)} br)" for d in xs], fontsize=8)
        ax.set_ylabel("Relative entropy  H/H_max", fontsize=9)
        ax.set_title(f"Stage {si}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        # Annotate bars with value
        for b, v in zip(bars, hs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        # Row 1: auxiliary metrics — dead_frac + cv + dominant_ratio
        ax2 = axes[1, si]
        xs2 = list(range(n_depths))
        df  = [depths[d]["dead_frac"]      if d in depths else 0.0 for d in xs2]
        cv  = [depths[d]["cv"]             if d in depths else 0.0 for d in xs2]
        dr  = [min(depths[d]["dominant_ratio"], 5.0) if d in depths else 1.0 for d in xs2]

        w = 0.25
        ax2.bar([x - w for x in xs2], df,  width=w, label="Dead frac",        color="#e63946", alpha=0.85, edgecolor="black", lw=0.5)
        ax2.bar(xs2,                   cv,  width=w, label="CoV (std/mean)",   color="#457b9d", alpha=0.85, edgecolor="black", lw=0.5)
        ax2.bar([x + w for x in xs2], dr,  width=w, label="Dominant ratio\n(1=uniform, >1=concentrated)", color="#f4a261", alpha=0.85, edgecolor="black", lw=0.5)
        ax2.axhline(1.0, color="#f4a261", linestyle=":", linewidth=0.8)
        ax2.set_xticks(xs2)
        ax2.set_xticklabels([f"d={d}" for d in xs2], fontsize=8)
        ax2.set_ylabel("Value", fontsize=9)
        ax2.set_title(f"Stage {si} — concentration metrics", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=7, loc="upper left")
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Hierarchy Exploration — {exp_name}  [{variant}]\n"
        "Relative entropy = 1.0 → perfectly uniform branch activation",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    p = save_dir / "hierarchy_exploration.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {p}")

    # ── Figure 2: sorted branch-probability detail ────────────────────────────
    fig2, axes2 = plt.subplots(1, n_s, figsize=(5 * n_s, 5), squeeze=False)
    for si, stage in enumerate(stages):
        ax = axes2[0, si]
        depths = stage["depths"]
        max_d  = max(depths.keys())
        for d in sorted(depths.keys()):
            ds = depths[d]
            sp = np.sort(ds["mean_probs"])[::-1]
            n  = len(sp)
            unif = 1.0 / n
            ax.semilogy(
                np.arange(n) / (n - 1) * 100,
                sp,
                label=f"d={d} ({n} br)",
                linewidth=1.5,
                alpha=0.8,
            )
        # uniform baseline for deepest depth
        max_n = 1 << (max_d + 1)
        ax.axhline(1.0 / max_n, color="grey", linestyle="--", linewidth=1.0,
                   label=f"Uniform (d={max_d})")
        ax.set_xlabel("Branch rank (% of total)", fontsize=9)
        ax.set_ylabel("Mean activation probability  (log scale)", fontsize=9)
        ax.set_title(f"Stage {si} — sorted branch probs", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig2.suptitle(
        f"Branch Probability Distribution (sorted) — {exp_name}  [{variant}]\n"
        "Flatter curve = more uniform exploration",
        fontsize=11, fontweight="bold",
    )
    fig2.tight_layout()
    p2 = save_dir / "hierarchy_exploration_detail.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot: {p2}")


def _plot_multi(result, variant, save_dir, exp_name):
    stages   = result["stages"]
    n_s      = len(stages)
    K        = len(stages[0]["hierarchies"])
    max_d    = max(
        max(hier["depths"].keys())
        for stage in stages
        for hier in stage["hierarchies"]
        if hier["depths"]
    )

    # ── Figure 1: relative entropy per hierarchy per stage + gate ─────────────
    n_rows = K + 1   # K hierarchy rows + 1 gate row
    fig, axes = plt.subplots(n_rows, n_s, figsize=(4 * n_s + 1, 4 * n_rows), squeeze=False)

    for si, stage in enumerate(stages):
        # hierarchy rows
        for ki, hier in enumerate(stage["hierarchies"]):
            ax = axes[ki, si]
            depths = hier["depths"]
            n_d    = max(depths.keys()) + 1 if depths else 1
            xs     = list(range(n_d))
            hs     = [depths[d]["relative_H"] if d in depths else 0.0 for d in xs]
            colours = [plt.cm.RdYlGn(h) for h in hs]
            bars = ax.bar(xs, hs, color=colours, edgecolor="black", linewidth=0.7)
            ax.axhline(1.0, color="green", linestyle="--", linewidth=1.0)
            ax.set_ylim(0, 1.15)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"d={d}" for d in xs], fontsize=8)
            ax.set_title(
                f"Stage {si}  ·  Hier {ki}", fontsize=9, fontweight="bold"
            )
            ax.set_ylabel("Relative H" if si == 0 else "", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
            for b, v in zip(bars, hs):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            if ki == 0:
                ax.set_title(f"Stage {si}\nHier {ki}", fontsize=9, fontweight="bold")

        # gate row
        ax_g = axes[K, si]
        gate = stage["gate"]
        gate_means = [gate[k]["mean_gate"]     for k in range(K)]
        gate_doms  = [gate[k]["dominant_frac"] for k in range(K)]
        ks = list(range(K))
        w  = 0.35
        ax_g.bar([k - w / 2 for k in ks], gate_means, width=w,
                 label="Mean gate weight",    color="#457b9d", edgecolor="black", lw=0.6)
        ax_g.bar([k + w / 2 for k in ks], gate_doms, width=w,
                 label="Dominant frac",       color="#f4a261", edgecolor="black", lw=0.6)
        ax_g.axhline(1.0 / K, color="grey", linestyle="--", linewidth=1.0,
                     label="Uniform (1/K)")
        ax_g.set_xticks(ks)
        ax_g.set_xticklabels([f"H{k}" for k in ks], fontsize=9)
        ax_g.set_ylim(0, 1.05)
        ax_g.set_ylabel("Fraction" if si == 0 else "", fontsize=8)
        ax_g.set_title(f"Stage {si}  ·  Gate", fontsize=9, fontweight="bold")
        if si == 0:
            ax_g.legend(fontsize=7)
        ax_g.grid(axis="y", alpha=0.3)

    row_labels = [f"Hierarchy {k} — relative entropy per depth" for k in range(K)]
    row_labels.append("Gate utilisation (mean weight + dominant fraction)")
    for row_i, lbl in enumerate(row_labels):
        axes[row_i, 0].set_ylabel(lbl, fontsize=8, labelpad=6)

    fig.suptitle(
        f"Hierarchy Exploration — {exp_name}  [{variant}]\n"
        "Relative entropy = 1.0 → perfectly uniform branch activation  |  "
        "Gate uniform = 1/K",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    p = save_dir / "hierarchy_exploration.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {p}")

    # ── Figure 2: sorted branch probs at deepest level, all hierarchies ───────
    fig2, axes2 = plt.subplots(1, n_s, figsize=(5 * n_s, 5), squeeze=False)
    colours_k = plt.cm.tab10(np.linspace(0, 1, K))
    for si, stage in enumerate(stages):
        ax = axes2[0, si]
        for ki, hier in enumerate(stage["hierarchies"]):
            depths = hier["depths"]
            if not depths:
                continue
            max_d_hier = max(depths.keys())
            ds = depths[max_d_hier]
            sp = np.sort(ds["mean_probs"])[::-1]
            n  = len(sp)
            ax.semilogy(
                np.arange(n) / (n - 1) * 100, sp,
                color=colours_k[ki], linewidth=1.6,
                label=f"Hier {ki} (d={max_d_hier}, {n} br)",
            )
        n_unif = 1 << (max_d + 1)
        ax.axhline(1.0 / n_unif, color="grey", linestyle="--", linewidth=1.0,
                   label="Uniform")
        ax.set_xlabel("Branch rank (%)", fontsize=9)
        ax.set_ylabel("Mean activation prob (log scale)" if si == 0 else "", fontsize=9)
        ax.set_title(f"Stage {si} — deepest-level branch probs", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig2.suptitle(
        f"Branch Probability Distribution (deepest level, sorted) — {exp_name}  [{variant}]\n"
        "Flatter = more uniform exploration across hierarchies",
        fontsize=11, fontweight="bold",
    )
    fig2.tight_layout()
    p2 = save_dir / "hierarchy_exploration_detail.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot: {p2}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args     = parse_args()
    config   = load_json(args.config)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = config.get("experiment_name", "unknown")
    print(f"Device: {device}  |  Experiment: {exp_name}")

    ckpt_path = resolve_checkpoint(config, args.checkpoint)
    save_dir  = resolve_save_dir(config)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Save dir:   {save_dir}")

    print("Loading model...")
    model, variant = load_model(ckpt_path, config, device)
    print(f"  Variant: {variant}")

    # data
    dc = config.get("data", {})
    loader_obj = CelebAHQLoader(
        data_root  = args.data_root or dc.get("data_root", "./data/celeba_hq"),
        image_size = dc.get("image_size", 256),
        batch_size = args.batch_size or dc.get("batch_size", 32),
        num_workers= args.num_workers,
        val_split  = dc.get("val_split", 0.05),
    )
    _, val_loader = loader_obj.get_loaders()
    if val_loader is None:
        raise RuntimeError("No validation split found.")

    print(f"\nCollecting branch statistics ({args.num_batches} batches)...")
    result = collect_stats(model, variant, val_loader, device, args.num_batches)

    # Print a compact summary to stdout
    is_multi = "multi" in variant
    for si, stage in enumerate(result["stages"]):
        if is_multi:
            for ki, hier in enumerate(stage["hierarchies"]):
                depths = hier["depths"]
                avg_rh = np.mean([ds["relative_H"] for ds in depths.values()])
                avg_dead = np.mean([ds["dead_frac"] for ds in depths.values()])
                print(f"  Stage {si}  Hier {ki}:  "
                      f"mean_rel_H={avg_rh:.3f}  mean_dead={avg_dead:.3f}")
            for k, gi in stage["gate"].items():
                print(f"  Stage {si}  Gate H{k}:  "
                      f"mean_weight={gi['mean_gate']:.3f}  dominant_frac={gi['dominant_frac']:.3f}")
        else:
            depths = stage["depths"]
            avg_rh = np.mean([ds["relative_H"] for ds in depths.values()])
            avg_dead = np.mean([ds["dead_frac"] for ds in depths.values()])
            print(f"  Stage {si}:  mean_rel_H={avg_rh:.3f}  mean_dead={avg_dead:.3f}")

    print(f"\nSaving to {save_dir} ...")
    save_csv(result, variant, save_dir, exp_name)
    save_plots(result, variant, save_dir, exp_name)
    print("Done.")


if __name__ == "__main__":
    main()
