#!/usr/bin/env python3
"""Multi-hierarchy taxon ablation: hierarchy muting + depth truncation.

Two analyses are run sequentially on the **last encoder stage** of any
multi-taxon model (MultiTaxon, TopKMultiTaxon, BiasmultiTaxon):

Analysis 1 — Hierarchy muting:
    Zero out one or more hierarchy's channel block at inference time.
    For n_hierarchies=K, sweeps all single-hierarchy and pair-hierarchy
    muting combinations plus the full baseline (none muted).

Analysis 2 — Depth truncation:
    Zero out the deep depth levels within EVERY hierarchy's block.
    Sweeps keep_depths = 1 … L where L = n_taxonomy_layers of last stage.

Channel layout of the last stage output  [B, K·C, H, W]:
    hierarchy k occupies channels [k·C : (k+1)·C]
    within each block: [2 ch, 4 ch, 8 ch, …, 2^L ch]  (depth-major)

Results saved under:
    <analysis_save_dir>/hierarchy_muting/
        muting_metrics.csv
        muting_ablation.png
    <analysis_save_dir>/truncation_ablation/
        truncation_metrics.csv
        truncation_ablation.png

Usage (from TaxonomicWeights/):
    python -m src.analyze.multi_taxon_truncation_ablation \\
        --config configs/celeba_hq/ablations/multi_taxon/<name>.json
    python -m src.analyze.multi_taxon_truncation_ablation \\
        --config <path> --checkpoint <path/to/best.pt> --num-batches 50
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.model.cnn.taxon.multi_taxon_ae import MultiTaxonAutoencoder
from src.model.cnn.taxon.topk_multi_taxon_ae import TopKMultiTaxonAutoencoder
from src.model.cnn.taxon.bias_multi_taxon_ae import BiasMultiTaxonAutoencoder
from src.utils.dataloader import CelebAHQLoader


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hierarchy muting + depth truncation for multi-taxon models"
    )
    p.add_argument("--config", required=True, help="Path to JSON config file")
    p.add_argument("--checkpoint", default=None,
                   help="Override checkpoint path (otherwise inferred from config)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-batches", type=int, default=50,
                   help="Val batches to evaluate per condition (default: 50)")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_checkpoint_path(config: dict, override: Optional[str]) -> str:
    if override:
        return override
    ckpt = config.get("analysis", {}).get("checkpoint_path")
    if ckpt:
        return ckpt
    # Derive: <output_dir parent> / <experiment_name> / checkpoints / best.pt
    output_dir = config.get("output", {}).get("output_dir", "")
    exp_name   = config.get("experiment_name", "")
    if output_dir and exp_name:
        return str(Path(output_dir).parent / exp_name / "checkpoints" / "best.pt")
    raise ValueError(
        "Cannot resolve checkpoint. Pass --checkpoint or set "
        "config['analysis']['checkpoint_path']."
    )


def resolve_analysis_save_dir(config: dict) -> Path:
    asd = config.get("output", {}).get("analysis_save_dir")
    if asd:
        return Path(asd)
    output_dir = config.get("output", {}).get("output_dir", "")
    exp_name   = config.get("experiment_name", "")
    if output_dir and exp_name:
        return Path(output_dir).parent / exp_name / "analysis"
    raise ValueError("Cannot resolve analysis_save_dir from config.")


# ── Model loading ──────────────────────────────────────────────────────────────

def _detect_variant(mc: dict, ckpt_args: dict) -> str:
    """Detect multi-taxon variant from config model section then ckpt args."""
    for src in (mc, ckpt_args):
        if "bias_update_rate" in src or "bias_ema_decay" in src:
            return "bias_multi_taxon"
        if "k_aux" in src or "dead_steps" in src:
            return "topk_multi_taxon"
    return "multi_taxon"


def _common_kwargs(a: dict) -> dict:
    return dict(
        in_channels        = a.get("in_channels", 3),
        resnet_variant     = a.get("resnet_variant", "18"),
        stage_taxonomy_layers = tuple(a.get("stage_taxonomy_layers", [5, 6, 7, 8])),
        stage_strides      = tuple(a.get("stage_strides", [1, 2, 2, 2])),
        stage_blocks       = a.get("stage_blocks", None),
        n_hierarchies      = a.get("n_hierarchies", 3),
        kernel_size        = a.get("kernel_size", 3),
        use_stem           = a.get("use_stem", True),
        stem_channels      = a.get("stem_channels", 64),
        stem_stride        = a.get("stem_stride", 2),
        use_stem_maxpool   = a.get("use_stem_maxpool", True),
        output_activation  = a.get("output_activation", "none"),
        depth_decay        = a.get("depth_decay", 0.5),
        temperature        = a.get("temperature", 1.0),
        hard               = a.get("hard", False),
    )


def load_model(
    checkpoint_path: str,
    device: torch.device,
    config: dict,
) -> Tuple[nn.Module, dict]:
    """Load a multi-taxon model, auto-detecting the variant."""
    ckpt     = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_a   = ckpt.get("args", {})
    mc       = config.get("model", {})

    # Merge: config values take priority for architecture (they're the ground truth),
    # but ckpt_args may have extras (k_aux, dead_steps, bias params) not in config.
    merged = {**ckpt_a, **mc}  # config wins on overlap
    variant = _detect_variant(mc, ckpt_a)

    kw = _common_kwargs(merged)
    if variant == "topk_multi_taxon":
        model = TopKMultiTaxonAutoencoder(
            **kw,
            k_aux      = merged.get("k_aux", None),
            dead_steps = merged.get("dead_steps", 2000),
            gate_k     = merged.get("gate_k", 1),
        )
    elif variant == "bias_multi_taxon":
        model = BiasMultiTaxonAutoencoder(
            **kw,
            bias_update_rate = merged.get("bias_update_rate", 0.001),
            bias_ema_decay   = merged.get("bias_ema_decay", 0.99),
            gate_k           = merged.get("gate_k", 1),
        )
    else:
        model = MultiTaxonAutoencoder(**kw)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    print(f"  Loaded {variant} model from {checkpoint_path}")
    return model, ckpt


# ── Stage inspection ───────────────────────────────────────────────────────────

def inspect_last_stage(model: nn.Module) -> Tuple[object, int, int, int]:
    """Return (last_stage, n_hierarchies, n_taxonomy_layers, per_hier_ch)."""
    last_stage = model.encoder.taxon_stages[-1]
    if not hasattr(last_stage, "hierarchies"):
        raise RuntimeError(
            "Last encoder stage has no 'hierarchies' attribute. "
            "This script only supports multi_taxon / topk_multi_taxon / bias_multi_taxon."
        )
    n_hierarchies    = len(last_stage.hierarchies)
    n_taxonomy_layers = last_stage.hierarchies[0].n_taxonomy_layers
    per_hier_ch      = last_stage.hierarchies[0].total_out_channels
    return last_stage, n_hierarchies, n_taxonomy_layers, per_hier_ch


# ── Channel helpers ────────────────────────────────────────────────────────────

def channels_up_to_depth(keep_depths: int) -> int:
    """Channels occupied by depth levels 0 … keep_depths-1 within one hierarchy block."""
    return (1 << (keep_depths + 1)) - 2


# ── Core evaluation ────────────────────────────────────────────────────────────

def _run_batches(
    model: nn.Module,
    data_loader,
    device: torch.device,
    num_batches: int,
    last_stage: object,
    hook_fn=None,
) -> Tuple[List[float], List[float]]:
    """Run inference with an optional forward hook on last_stage."""
    handle = None
    if hook_fn is not None:
        handle = last_stage.register_forward_hook(hook_fn)
    mse_list: List[float] = []
    mae_list:  List[float] = []
    try:
        with torch.no_grad():
            for i, (images, _) in enumerate(data_loader):
                if i >= num_batches:
                    break
                images = images.to(device)
                recon, *_ = model(images)
                mse_list.extend(
                    ((images - recon) ** 2).mean(dim=(1, 2, 3)).cpu().tolist()
                )
                mae_list.extend(
                    torch.abs(images - recon).mean(dim=(1, 2, 3)).cpu().tolist()
                )
    finally:
        if handle is not None:
            handle.remove()
    return mse_list, mae_list


def _stats(vals: List[float]) -> Tuple[float, float]:
    a = np.array(vals)
    return float(np.mean(a)), float(np.std(a))


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 1 — Hierarchy muting
# ══════════════════════════════════════════════════════════════════════════════

def _mute_hook(mute_set: FrozenSet[int], n_hierarchies: int, per_hier_ch: int):
    def hook(module, input, output):
        z, logps, sd = output
        z = z.clone()
        for k in mute_set:
            z[:, k * per_hier_ch : (k + 1) * per_hier_ch, :, :] = 0.0
        return (z, logps, sd)
    return hook


def run_hierarchy_muting(
    model: nn.Module,
    data_loader,
    device: torch.device,
    last_stage,
    n_hierarchies: int,
    per_hier_ch: int,
    num_batches: int,
) -> List[Dict]:
    """Sweep all single and pair muting combinations plus no-muting baseline."""
    # Build ordered list of conditions: (frozenset of muted indices, label)
    conditions: List[Tuple[FrozenSet[int], str]] = []

    # Baseline: nothing muted
    conditions.append((frozenset(), "none (full)"))

    # Single-hierarchy muting
    for k in range(n_hierarchies):
        conditions.append((frozenset([k]), f"mute h{k}"))

    # Pair-hierarchy muting (only meaningful when K >= 3)
    if n_hierarchies >= 3:
        for k1, k2 in itertools.combinations(range(n_hierarchies), 2):
            conditions.append((frozenset([k1, k2]), f"mute h{k1}+h{k2}"))

    # All-but-one muting (keep only one hierarchy alive)
    if n_hierarchies >= 3:
        for k_keep in range(n_hierarchies):
            mute_set = frozenset(range(n_hierarchies)) - {k_keep}
            label = f"only h{k_keep}"
            conditions.append((mute_set, label))

    results = []
    for mute_set, label in conditions:
        n_muted = len(mute_set)
        print(f"  [{label}] ...", end=" ", flush=True)
        hook_fn = None if not mute_set else _mute_hook(mute_set, n_hierarchies, per_hier_ch)
        mse_list, mae_list = _run_batches(
            model, data_loader, device, num_batches, last_stage, hook_fn
        )
        mean_mse, std_mse = _stats(mse_list)
        mean_mae, std_mae = _stats(mae_list)
        print(f"MSE={mean_mse:.5f}  MAE={mean_mae:.5f}")
        results.append({
            "label":     label,
            "n_muted":   n_muted,
            "muted_ids": ",".join(str(k) for k in sorted(mute_set)) or "none",
            "mean_mse":  mean_mse,
            "std_mse":   std_mse,
            "mean_mae":  mean_mae,
            "std_mae":   std_mae,
            "n_samples": len(mse_list),
        })
    return results


def save_muting_results(
    results: List[Dict],
    save_dir: Path,
    n_hierarchies: int,
    experiment_name: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = save_dir / "muting_metrics.csv"
    fields = ["label", "n_muted", "muted_ids", "mean_mse", "std_mse",
              "mean_mae", "std_mae", "n_samples"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV:  {csv_path}")

    # Plot
    labels   = [r["label"]    for r in results]
    mses     = [r["mean_mse"] for r in results]
    mse_errs = [r["std_mse"]  for r in results]
    maes     = [r["mean_mae"] for r in results]
    mae_errs = [r["std_mae"]  for r in results]
    n_muted  = [r["n_muted"]  for r in results]

    # Colour by number of hierarchies muted
    cmap = plt.cm.get_cmap("RdYlGn_r", n_hierarchies + 1)
    colours = [cmap(nm) for nm in n_muted]

    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(results) * 1.1), 6))

    for ax, vals, errs, ylabel, title in [
        (axes[0], mses, mse_errs, "Mean MSE",      "MSE vs. hierarchy muting"),
        (axes[1], maes, mae_errs, "Mean MAE",       "MAE vs. hierarchy muting"),
    ]:
        bars = ax.bar(range(len(results)), vals, yerr=errs, capsize=4,
                      color=colours, edgecolor="black", linewidth=0.8)
        baseline = next(r["mean_mse"] if ylabel == "Mean MSE" else r["mean_mae"]
                        for r in results if r["n_muted"] == 0)
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=1.0,
                   label="Full model (none muted)")
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Legend for colour coding
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=cmap(nm), edgecolor="black",
                        label=f"{nm} hierarch{'y' if nm==1 else 'ies'} muted")
                  for nm in range(n_hierarchies + 1)]
    fig.legend(handles=legend_els, loc="lower center",
               ncol=n_hierarchies + 1, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
    fig.suptitle(
        f"Hierarchy Muting Ablation — {experiment_name}\n"
        f"n_hierarchies={n_hierarchies}  |  "
        "deeper channels zeroed at inference, no retraining",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    plot_path = save_dir / "muting_ablation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {plot_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Analysis 2 — Depth truncation (applied across all hierarchies uniformly)
# ══════════════════════════════════════════════════════════════════════════════

def _trunc_hook(keep_depths: int, n_hierarchies: int, per_hier_ch: int):
    trunc_start = channels_up_to_depth(keep_depths)

    def hook(module, input, output):
        z, logps, sd = output
        z = z.clone()
        for k in range(n_hierarchies):
            base = k * per_hier_ch
            z[:, base + trunc_start : base + per_hier_ch, :, :] = 0.0
        return (z, logps, sd)
    return hook


def run_depth_truncation(
    model: nn.Module,
    data_loader,
    device: torch.device,
    last_stage,
    n_hierarchies: int,
    n_taxonomy_layers: int,
    per_hier_ch: int,
    num_batches: int,
) -> List[Dict]:
    """Sweep keep_depths from 1 up to n_taxonomy_layers (full)."""
    results = []
    for keep in range(1, n_taxonomy_layers + 1):
        is_full = keep == n_taxonomy_layers
        label = f"all {n_taxonomy_layers} (full)" if is_full else f"depth ≤ {keep}"
        print(f"  keep={keep}/{n_taxonomy_layers}  [{label}] ...", end=" ", flush=True)
        hook_fn = None if is_full else _trunc_hook(keep, n_hierarchies, per_hier_ch)
        mse_list, mae_list = _run_batches(
            model, data_loader, device, num_batches, last_stage, hook_fn
        )
        mean_mse, std_mse = _stats(mse_list)
        mean_mae, std_mae = _stats(mae_list)
        print(f"MSE={mean_mse:.5f}  MAE={mean_mae:.5f}")
        results.append({
            "keep_depths": keep,
            "label":       label,
            "mean_mse":    mean_mse,
            "std_mse":     std_mse,
            "mean_mae":    mean_mae,
            "std_mae":     std_mae,
            "n_samples":   len(mse_list),
        })
    return results


def save_truncation_results(
    results: List[Dict],
    save_dir: Path,
    n_taxonomy_layers: int,
    n_hierarchies: int,
    experiment_name: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "truncation_metrics.csv"
    fields = ["keep_depths", "label", "mean_mse", "std_mse",
              "mean_mae", "std_mae", "n_samples"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV:  {csv_path}")

    xs       = [r["keep_depths"] for r in results]
    mses     = [r["mean_mse"]    for r in results]
    mse_errs = [r["std_mse"]     for r in results]
    maes     = [r["mean_mae"]    for r in results]
    mae_errs = [r["std_mae"]     for r in results]

    full_mse = next(r["mean_mse"] for r in results if r["keep_depths"] == n_taxonomy_layers)
    full_mae = next(r["mean_mae"] for r in results if r["keep_depths"] == n_taxonomy_layers)
    pct_mse  = [min(full_mse / m, 1.0) * 100 if m > 0 else 0.0 for m in mses]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.errorbar(xs, mses, yerr=mse_errs, marker="o", color="#1f77b4",
                capsize=4, linewidth=2, markersize=7)
    ax.axhline(full_mse, color="grey", linestyle="--", linewidth=1.0, label="Full model")
    ax.set_xlabel(f"Depth levels kept per hierarchy  (last stage, L={n_taxonomy_layers})", fontsize=11)
    ax.set_ylabel("Mean MSE", fontsize=11)
    ax.set_title("MSE  vs.  Depth kept", fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.errorbar(xs, maes, yerr=mae_errs, marker="s", color="#d62728",
                capsize=4, linewidth=2, markersize=7)
    ax.axhline(full_mae, color="grey", linestyle="--", linewidth=1.0, label="Full model")
    ax.set_xlabel(f"Depth levels kept per hierarchy  (last stage, L={n_taxonomy_layers})", fontsize=11)
    ax.set_ylabel("Mean MAE", fontsize=11)
    ax.set_title("MAE  vs.  Depth kept", fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(xs, pct_mse, marker="D", color="#2ca02c", linewidth=2, markersize=7)
    ax.axhline(100.0, color="grey", linestyle="--", linewidth=1.0, label="Full model (100%)")
    ax.fill_between(xs, pct_mse, 100.0, alpha=0.12, color="#2ca02c")
    ax.set_xlabel(f"Depth levels kept per hierarchy  (last stage, L={n_taxonomy_layers})", fontsize=11)
    ax.set_ylabel("Recon quality retained  (%)", fontsize=11)
    ax.set_title(
        "% Quality Retained\n(full_MSE / trunc_MSE × 100, capped at 100%)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(xs)
    ax.set_ylim(0.0, 110.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Depth Truncation Ablation — {experiment_name}\n"
        f"n_hierarchies={n_hierarchies}, last-stage L={n_taxonomy_layers}  |  "
        "truncation applied uniformly across all hierarchies",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    plot_path = save_dir / "truncation_ablation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {plot_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args       = parse_args()
    config     = load_json(args.config)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name   = config.get("experiment_name", "unknown")
    print(f"Device: {device}  |  Experiment: {exp_name}")

    checkpoint_path  = resolve_checkpoint_path(config, args.checkpoint)
    analysis_save_dir = resolve_analysis_save_dir(config)

    # ── Load model ────────────────────────────────────────────────────────────
    model, _ = load_model(checkpoint_path, device, config)

    # ── Inspect last multi-taxon stage ────────────────────────────────────────
    last_stage, n_hierarchies, n_taxonomy_layers, per_hier_ch = inspect_last_stage(model)
    print(
        f"Last stage: {n_hierarchies} hierarchies × {n_taxonomy_layers} depth levels "
        f"= {per_hier_ch} ch/hierarchy  ({n_hierarchies * per_hier_ch} total channels)"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    dc          = config.get("data", {})
    data_root   = args.data_root or dc.get("data_root", "./data/celeba_hq")
    batch_size  = args.batch_size or dc.get("batch_size", 32)
    loader_obj  = CelebAHQLoader(
        data_root=data_root,
        image_size=dc.get("image_size", 256),
        batch_size=batch_size,
        num_workers=args.num_workers,
        val_split=dc.get("val_split", 0.05),
    )
    _, val_loader = loader_obj.get_loaders()
    if val_loader is None:
        raise RuntimeError("No validation split found — check data_root or val_split.")

    # ══════════════════════════════════════════════════════════════════════════
    # Analysis 1: Hierarchy muting
    # ══════════════════════════════════════════════════════════════════════════
    muting_save = analysis_save_dir / "hierarchy_muting"
    print(f"\n{'='*60}")
    print(f"Analysis 1: Hierarchy muting  →  {muting_save}")
    print(f"{'='*60}")
    muting_results = run_hierarchy_muting(
        model, val_loader, device, last_stage,
        n_hierarchies, per_hier_ch, args.num_batches,
    )
    save_muting_results(muting_results, muting_save, n_hierarchies, exp_name)

    # ══════════════════════════════════════════════════════════════════════════
    # Analysis 2: Depth truncation
    # ══════════════════════════════════════════════════════════════════════════
    trunc_save = analysis_save_dir / "truncation_ablation"
    print(f"\n{'='*60}")
    print(f"Analysis 2: Depth truncation  →  {trunc_save}")
    print(f"{'='*60}")
    trunc_results = run_depth_truncation(
        model, val_loader, device, last_stage,
        n_hierarchies, n_taxonomy_layers, per_hier_ch, args.num_batches,
    )
    save_truncation_results(
        trunc_results, trunc_save, n_taxonomy_layers, n_hierarchies, exp_name
    )

    print("\nAll analyses complete.")


if __name__ == "__main__":
    main()
