#!/usr/bin/env python3
"""Comparative analysis: all OpenWebText / LLM linear SAE runs.

Discovers every run directory under ``outputs/openwebtext/`` that has a
``checkpoints/best.pt``, loads pre-computed analysis artifacts (npz files
from ``analyze_linear_sae.py``), and produces a multi-page summary comparing:

  Page 1 — Reconstruction Quality
    • Training curves (val-recon per epoch) for every run with history.json
    • Bar chart: best-val MSE, MAE, and cosine similarity per model

  Page 2 — Sparsity Analysis
    • Bar charts: mean L0 norm, dead/selective/dense feature fractions
    • Lifetime-sparsity histogram comparison across models

Output: ``outputs/comparison_openwebtext/comparison_openwebtext.png``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─── colour palettes ──────────────────────────────────────────────────────────

_TAXON_PALETTE = ["#1f77b4", "#4e9fc7", "#1a5a8a", "#6baed6"]
_MULTI_TAXON_PALETTE = ["#9467bd", "#7b4f9e", "#c5a0d8", "#6a3d9a"]
_SAE_PALETTE = ["#ff7f0e", "#e55e00", "#ffa64d", "#cc4a00"]
_TOPK_SAE_PALETTE = ["#d62728", "#c03030", "#e85555", "#a82020"]


def _model_type(run_dir: str) -> str:
    name = run_dir.lower()
    if "multi_taxon" in name:
        return "multi_taxon"
    if "taxon" in name:
        return "taxon"
    if "topk" in name:
        return "topk_sae"
    return "sae"


def _model_colour(model_type: str, idx: int) -> str:
    palettes = {
        "taxon": _TAXON_PALETTE,
        "multi_taxon": _MULTI_TAXON_PALETTE,
        "sae": _SAE_PALETTE,
        "topk_sae": _TOPK_SAE_PALETTE,
    }
    pal = palettes.get(model_type, _SAE_PALETTE)
    return pal[idx % len(pal)]


def _short_name(run_dir: str) -> str:
    n = run_dir
    for prefix in ("linear_multi_taxon_sae_gpt2", "linear_taxon_sae_gpt2",
                   "linear_topk_sae_gpt2", "linear_sae_gpt2",
                   "linear_multi_taxon_sae", "linear_taxon_sae",
                   "linear_topk_sae", "linear_sae"):
        n = n.replace(prefix, "").strip("_")
    mtype = _model_type(run_dir)
    n = n.replace("_", " ").strip()
    suffix = f" ({n})" if n else ""
    return f"{mtype}{suffix}"


# ─── discovery ────────────────────────────────────────────────────────────────

def discover_runs(outputs_dir: Path) -> List[Dict]:
    runs = []
    if not outputs_dir.exists():
        return runs

    for name in sorted(os.listdir(outputs_dir)):
        run_path = outputs_dir / name
        if not run_path.is_dir():
            continue
        best_ckpt = run_path / "checkpoints" / "best.pt"
        if not best_ckpt.exists():
            continue
        runs.append({
            "name": name,
            "short": _short_name(name),
            "type": _model_type(name),
            "path": run_path,
            "best_ckpt": best_ckpt,
            "history": run_path / "training_history.json",
            "analysis": run_path / "analysis",
        })

    _order = {"taxon": 0, "multi_taxon": 1, "sae": 2, "topk_sae": 3}
    runs.sort(key=lambda r: (_order.get(r["type"], 99), r["name"]))

    type_counters: Dict[str, int] = {}
    for r in runs:
        idx = type_counters.get(r["type"], 0)
        r["colour"] = _model_colour(r["type"], idx)
        type_counters[r["type"]] = idx + 1

    return runs


# ─── plotting ─────────────────────────────────────────────────────────────────

def _load_history(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_npz(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def plot_page1_reconstruction(runs: List[Dict], out_dir: Path) -> None:
    """Page 1: training curves + bar charts for MSE, MAE, cosine similarity."""
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # (0, 0:2) — training curves
    ax_tc = fig.add_subplot(gs[0, :])
    has_curves = False
    for r in runs:
        h = _load_history(r["history"])
        if h is None:
            continue
        epochs = h.get("epochs", [])
        val_recon = h.get("val_recon", [])
        if not epochs or not val_recon:
            continue
        ax_tc.plot(epochs, val_recon, label=r["short"], color=r["colour"], linewidth=1.5)
        has_curves = True
    if has_curves:
        ax_tc.set_title("Validation Reconstruction Loss per Epoch", fontsize=12)
        ax_tc.set_xlabel("Epoch")
        ax_tc.set_ylabel("MSE")
        ax_tc.legend(fontsize=7, ncol=2)
        ax_tc.grid(True, alpha=0.3)

    # Gather npz stats
    names, mse_vals, mae_vals, cos_vals, colours = [], [], [], [], []
    for r in runs:
        recon = _load_npz(r["analysis"] / "reconstruction_metrics.npz")
        if recon is None:
            continue
        names.append(r["short"])
        mse_vals.append(float(recon.get("mean_mse", np.array(recon["mse"])).mean()))
        mae_vals.append(float(recon.get("mean_mae", np.array(recon["mae"])).mean()))
        cos_vals.append(float(recon.get("mean_cosine_similarity",
                                        np.array(recon.get("cosine_similarity", [0]))).mean()))
        colours.append(r["colour"])

    if names:
        for col_idx, (title, vals) in enumerate([
            ("Mean MSE", mse_vals),
            ("Mean MAE", mae_vals),
            ("Mean Cosine Similarity", cos_vals),
        ]):
            ax = fig.add_subplot(gs[1, col_idx])
            bars = ax.bar(range(len(names)), vals, color=colours, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_title(title, fontsize=11)
            ax.grid(axis="y", alpha=0.3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=6)

    fig.suptitle("Page 1 — Reconstruction Quality", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "page1_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> page1_reconstruction.png")


def plot_page2_sparsity(runs: List[Dict], out_dir: Path) -> None:
    """Page 2: sparsity analysis bar charts + lifetime sparsity histograms."""
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    names, l0_vals, dead_vals, sel_vals, dense_vals = [], [], [], [], []
    lts_data = []
    colours = []
    for r in runs:
        lat = _load_npz(r["analysis"] / "latent_statistics.npz")
        if lat is None:
            continue
        names.append(r["short"])
        l0_vals.append(float(lat["mean_l0"]))
        dead_vals.append(float(lat["dead_frac"]) * 100)
        sel_vals.append(float(lat["selective_frac"]) * 100)
        dense_vals.append(float(lat["dense_frac"]) * 100)
        lts_data.append(np.array(lat["lifetime_sparsity"]))
        colours.append(r["colour"])

    if not names:
        plt.close(fig)
        return

    # (0, 0) — Mean L0
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(range(len(names)), l0_vals, color=colours, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_title("Mean L0 Norm", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, l0_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    # (0, 1) — Dead feature fraction
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(range(len(names)), dead_vals, color=colours, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_title("Dead Features (<1% activation)", fontsize=11)
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, dead_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    # (0, 2) — Feature category stacked bar
    ax = fig.add_subplot(gs[0, 2])
    x_pos = range(len(names))
    mod_vals = [100 - d - s - dn for d, s, dn in zip(dead_vals, sel_vals, dense_vals)]
    ax.bar(x_pos, dead_vals, color="#d62728", label="Dead (<1%)")
    ax.bar(x_pos, sel_vals, bottom=dead_vals, color="#2ca02c", label="Selective (1-20%)")
    ax.bar(x_pos, mod_vals, bottom=[d + s for d, s in zip(dead_vals, sel_vals)],
           color="#ff7f0e", label="Moderate (20-50%)")
    ax.bar(x_pos, dense_vals, bottom=[d + s + m for d, s, m in zip(dead_vals, sel_vals, mod_vals)],
           color="#1f77b4", label="Dense (>50%)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_title("Feature Activity Categories", fontsize=11)
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # (1, 0:3) — Lifetime sparsity histograms
    ax = fig.add_subplot(gs[1, :])
    for i, (name, lts) in enumerate(zip(names, lts_data)):
        ax.hist(lts, bins=50, alpha=0.5, label=name, color=colours[i], edgecolor="none")
    ax.set_title("Lifetime Sparsity Distribution (fraction near-zero per dim)", fontsize=12)
    ax.set_xlabel("Lifetime sparsity")
    ax.set_ylabel("# Dimensions")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Page 2 — Sparsity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "page2_sparsity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> page2_sparsity.png")


def save_summary_csv(runs: List[Dict], out_dir: Path) -> None:
    """Save a CSV summary of all runs."""
    import csv
    rows = []
    for r in runs:
        recon = _load_npz(r["analysis"] / "reconstruction_metrics.npz")
        lat = _load_npz(r["analysis"] / "latent_statistics.npz")
        sp = _load_npz(r["analysis"] / "sparsity_metrics.npz")
        row = {"name": r["name"], "type": r["type"]}
        if recon is not None:
            row["mean_mse"] = float(recon.get("mean_mse", np.array(recon["mse"]).mean()))
            row["mean_mae"] = float(recon.get("mean_mae", np.array(recon["mae"]).mean()))
            row["mean_cos"] = float(recon.get("mean_cosine_similarity", 0))
        if lat is not None:
            row["mean_l0"] = float(lat["mean_l0"])
            row["mean_sparsity"] = float(lat["mean_sparsity"])
            row["dead_frac"] = float(lat["dead_frac"])
            row["latent_dim"] = int(lat["latent_dim"])
        if sp is not None:
            row["mean_reg_loss"] = float(sp["mean_sparsity_loss"])
        rows.append(row)

    if not rows:
        return

    keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in keys:
                keys.append(k)

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> summary.csv  ({len(rows)} runs)")


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare OpenWebText linear SAE runs")
    parser.add_argument("--outputs-dir", type=str,
                        default="./outputs/openwebtext")
    parser.add_argument("--save-dir", type=str,
                        default="./outputs/comparison_openwebtext")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(outputs_dir)
    if not runs:
        print(f"No runs found under {outputs_dir}")
        return

    print(f"Discovered {len(runs)} runs:")
    for r in runs:
        print(f"  [{r['type']:>15s}]  {r['name']}")

    print("\nGenerating comparison plots ...")
    plot_page1_reconstruction(runs, out_dir)
    plot_page2_sparsity(runs, out_dir)
    save_summary_csv(runs, out_dir)

    print(f"\nAll comparison artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
