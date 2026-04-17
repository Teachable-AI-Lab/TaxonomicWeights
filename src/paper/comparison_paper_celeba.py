#!/usr/bin/env python3
"""Comparison plots across all CelebA-HQ per-model paper analysis results.

Scans ``outputs/paper/celeba_hq/`` for every subdirectory containing a
``metrics.json`` and produces comparison figures plus a summary CSV.

Saved to ``outputs/paper/celeba_hq/comparison/``
    comparison_overview.png  — single large figure with all panels
    page1_reconstruction.png — MSE / MAE / PSNR bar charts + training curves
    page2_sparsity.png       — L0 and dead-neuron bars
    page3_downstream.png     — linear-probe AUC and KNN accuracy bars
    page4_tsne.png           — t-SNE side-by-side grid
    comparison_summary.csv   — all scalar metrics in one table

Usage
-----
python src/paper/comparison_paper_celeba.py \\
    --paper-dir outputs/paper/celeba_hq \\
    --save-dir  outputs/paper/celeba_hq/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Colour palette (mirrors compare_celeba_hq.py)
# ---------------------------------------------------------------------------

_PALETTE = {
    "taxon":                    "#1f77b4",
    "multi_taxon":              "#9467bd",
    "topk_taxon":               "#0077b6",
    "topk_multi_taxon":         "#7209b7",
    "bias_taxon":               "#06d6a0",
    "bias_multi_taxon":         "#e63946",
    "sae":                      "#ff7f0e",
    "topk_sae":                 "#d62728",
    "gated_sae":                "#17becf",
    "jumprelu_sae":             "#8c564b",
    "matryoshka_batch_topk_sae": "#1a9850",
    "softmax_sae":              "#f4a261",
    "baseline":                 "#2ca02c",
}


def _model_type_from_name(name: str) -> str:
    if name.startswith("topk_multi_taxon_"):
        return "topk_multi_taxon"
    if name.startswith("topk_taxon_"):
        return "topk_taxon"
    if name.startswith("bias_multi_taxon_"):
        return "bias_multi_taxon"
    if name.startswith("bias_taxon_"):
        return "bias_taxon"
    if name.startswith("multi_taxon_"):
        return "multi_taxon"
    if name.startswith("sae_matryoshka_batch_topk_"):
        return "matryoshka_batch_topk_sae"
    if name.startswith("sae_softmax_") or "softmax_sae" in name:
        return "softmax_sae"
    if name.startswith("sae_jumprelu_"):
        return "jumprelu_sae"
    if name.startswith("sae_topk_"):
        return "topk_sae"
    if name.startswith("sae_gated_"):
        return "gated_sae"
    if name.startswith("sae_"):
        return "sae"
    if name.startswith("baseline_"):
        return "baseline"
    return "taxon"


def _colour(name: str) -> str:
    return _PALETTE.get(_model_type_from_name(name), "#999999")


def _short_label(name: str) -> str:
    mtype = _model_type_from_name(name)   # resolve type from original name
    for prefix in (
        "topk_multi_taxon_ae_celeba_hq_r18_",
        "topk_taxon_ae_celeba_hq_r18_",
        "bias_multi_taxon_ae_celeba_hq_r18_",
        "bias_taxon_ae_celeba_hq_r18_",
        "multi_taxon_ae_celeba_hq_r18_",
        "taxon_ae_celeba_hq_r18_",
        "sae_matryoshka_batch_topk_celeba_hq_r18_",
        "sae_softmax_celeba_hq_r18_",
        "sae_jumprelu_celeba_hq_r18_",
        "sae_topk_celeba_hq_r18_",
        "sae_gated_celeba_hq_r18_",
        "sae_celeba_hq_r18",
        "baseline_ae_celeba_hq_r18",
    ):
        name = name.replace(prefix, "").strip("_")
    cleaned = name.replace("_", " ").strip()
    return f"{mtype} ({cleaned})" if cleaned else mtype


_TYPE_ORDER = {
    "taxon": 0, "topk_taxon": 1, "bias_taxon": 2,
    "multi_taxon": 3, "topk_multi_taxon": 4, "bias_multi_taxon": 5,
    "sae": 6, "topk_sae": 7, "gated_sae": 8, "jumprelu_sae": 9,
    "matryoshka_batch_topk_sae": 10, "softmax_sae": 11, "baseline": 12,
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_results(paper_dir: Path) -> List[Dict]:
    """Find all per-model result directories with a metrics.json."""
    runs = []
    for name in sorted(os.listdir(paper_dir)):
        d = paper_dir / name
        if not d.is_dir() or name == "comparison":
            continue
        mj = d / "metrics.json"
        if not mj.exists():
            continue
        with open(mj) as f:
            m = json.load(f)
        tsne_path = d / "tsne_embeddings.npy"
        training_history = None
        # Try to find training history in the outputs dir
        out_run = ROOT / "outputs" / "celeba_hq" / name
        if (out_run / "training_history.json").exists():
            training_history = out_run / "training_history.json"

        runs.append({
            "name": name,
            "short": _short_label(name),
            "type": _model_type_from_name(name),
            "colour": _colour(name),
            "dir": d,
            "metrics": m,
            "tsne": tsne_path if tsne_path.exists() else None,
            "history": training_history,
        })

    runs.sort(key=lambda r: (_TYPE_ORDER.get(r["type"], 99), r["name"]))
    return runs


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _bar_chart(ax, runs, metric_key, title, ylabel, higher_is_better=False,
               pct=False):
    names = [r["short"] for r in runs]
    vals = [r["metrics"].get(metric_key, float("nan")) for r in runs]
    colours = [r["colour"] for r in runs]
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colours, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    if pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    # Annotate bars
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            label = f"{v*100:.1f}%" if pct else f"{v:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=5, rotation=90)
    ax.grid(axis="y", alpha=0.3)


def _training_curves(ax, runs):
    for r in runs:
        if r["history"] is None:
            continue
        try:
            with open(r["history"]) as f:
                h = json.load(f)
            recon_key = "val_recon" if "val_recon" in h else "val_recon_k0"
            if recon_key not in h:
                continue
            ys = np.array(h[recon_key], dtype=float)
            xs = np.array(h["epochs"])
            ax.plot(xs, ys, label=r["short"], color=r["colour"], linewidth=1.2)
        except Exception:
            pass
    ax.set_title("Val reconstruction loss (per epoch)", fontsize=9)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Val recon", fontsize=8)
    ax.legend(fontsize=5, ncol=2)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def build_page1_reconstruction(runs, save_path):
    """MSE / MAE / PSNR bars + training curves."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4)

    _bar_chart(fig.add_subplot(gs[0, 0]), runs, "mse_mean",
               "Mean MSE (↓)", "MSE")
    _bar_chart(fig.add_subplot(gs[0, 1]), runs, "mae_mean",
               "Mean MAE (↓)", "MAE")
    _bar_chart(fig.add_subplot(gs[0, 2]), runs, "psnr_mean",
               "Mean PSNR (↑)", "PSNR (dB)", higher_is_better=True)
    _training_curves(fig.add_subplot(gs[1, :]))

    fig.suptitle("Reconstruction Quality — CelebA-HQ", fontsize=13, y=1.01)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def build_page2_sparsity(runs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _bar_chart(axes[0], runs, "l0_mean", "Mean L0 (active features per image, ↓)", "L0")
    _bar_chart(axes[1], runs, "dead_neuron_frac",
               "Dead neuron fraction (↑ = worse)", "Fraction", pct=True)
    fig.suptitle("Sparsity Metrics — CelebA-HQ", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def build_page3_downstream(runs, save_path):
    has_probe = any(not np.isnan(r["metrics"].get("probe_mean_auc", float("nan")))
                    for r in runs)
    ncols = 3 if has_probe else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    _bar_chart(axes[0], runs, "probe_mean_auc",
               "Linear probe mean AUC-ROC (40 attrs, ↑)", "AUC-ROC",
               higher_is_better=True)
    if has_probe:
        _bar_chart(axes[1], runs, "knn_k1_mean_acc",
                   "1-NN accuracy (mean over attrs, ↑)", "Acc",
                   higher_is_better=True)
        _bar_chart(axes[2], runs, "knn_k5_mean_acc",
                   "5-NN accuracy (mean over attrs, ↑)", "Acc",
                   higher_is_better=True)

    fig.suptitle("Downstream Task (Linear Probe / KNN) — CelebA-HQ", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def build_page4_tsne(runs_with_tsne, save_path):
    n = len(runs_with_tsne)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, r in enumerate(runs_with_tsne):
        emb = np.load(r["tsne"])
        axes[i].scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.4, color=r["colour"])
        axes[i].set_title(r["short"], fontsize=7)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("t-SNE of latent codes — CelebA-HQ", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def build_overview(runs, save_path):
    """Compact single-page overview with the key metrics."""
    n = len(runs)
    if n == 0:
        print("No results found for overview.")
        return

    metric_keys = [
        ("mse_mean",         "MSE ↓",          False),
        ("psnr_mean",        "PSNR ↑",          True),
        ("l0_mean",          "L0 ↓",            False),
        ("dead_neuron_frac", "Dead% ↑=worse",   False),
        ("probe_mean_auc",   "Probe AUC ↑",     True),
        ("knn_k5_mean_acc",  "5-NN Acc ↑",      True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax, (key, label, hib) in zip(axes, metric_keys):
        _bar_chart(ax, runs, key, label, label, higher_is_better=hib,
                   pct=(key == "dead_neuron_frac"))
    fig.suptitle("Model Comparison Overview — CelebA-HQ", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comparison plots for CelebA-HQ paper results")
    parser.add_argument("--paper-dir", default=None,
                        help="Root of per-model paper results. Default: outputs/paper/celeba_hq")
    parser.add_argument("--save-dir", default=None,
                        help="Where to save comparison outputs. Default: <paper-dir>/comparison")
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir or (ROOT / "outputs" / "paper" / "celeba_hq"))
    save_dir = Path(args.save_dir or (paper_dir / "comparison"))
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {paper_dir} ...")
    runs = discover_results(paper_dir)
    if not runs:
        print("No per-model results found. Run paper_analysis_celeba.py for each model first.")
        return
    print(f"Found {len(runs)} results: {[r['name'] for r in runs]}")

    # ── Summary CSV ──────────────────────────────────────────────────────────
    all_keys = sorted({k for r in runs for k in r["metrics"].keys()
                       if not isinstance(r["metrics"][k], list)})
    csv_path = save_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run"] + all_keys)
        for r in runs:
            row = [r["name"]] + [r["metrics"].get(k, "") for k in all_keys]
            writer.writerow(row)
    print(f"CSV  → {csv_path}")

    # ── Figures ──────────────────────────────────────────────────────────────
    build_overview(runs, save_dir / "comparison_overview.png")
    print(f"Overview → {save_dir / 'comparison_overview.png'}")

    build_page1_reconstruction(runs, save_dir / "page1_reconstruction.png")
    print(f"Page 1  → {save_dir / 'page1_reconstruction.png'}")

    build_page2_sparsity(runs, save_dir / "page2_sparsity.png")
    print(f"Page 2  → {save_dir / 'page2_sparsity.png'}")

    build_page3_downstream(runs, save_dir / "page3_downstream.png")
    print(f"Page 3  → {save_dir / 'page3_downstream.png'}")

    runs_with_tsne = [r for r in runs if r["tsne"] is not None]
    if runs_with_tsne:
        build_page4_tsne(runs_with_tsne, save_dir / "page4_tsne.png")
        print(f"Page 4  → {save_dir / 'page4_tsne.png'}")
    else:
        print("Page 4: no t-SNE embeddings found.")

    print(f"\nDone. All comparison outputs saved to {save_dir}")


if __name__ == "__main__":
    main()
