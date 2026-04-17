#!/usr/bin/env python3
"""Comparison plots across all ImageNet per-model paper analysis results.

Scans ``outputs/paper/imagenet/`` for every subdirectory containing a
``metrics.json`` and produces comparison figures plus a summary CSV.

Saved to ``outputs/paper/imagenet/comparison/``
    comparison_overview.png
    page1_reconstruction.png
    page2_sparsity.png
    page3_downstream.png
    page4_tsne.png
    comparison_summary.csv

Usage
-----
python src/paper/comparison_paper_imagenet.py \\
    --paper-dir outputs/paper/imagenet \\
    --save-dir  outputs/paper/imagenet/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paper.comparison_paper_celeba import (
    _PALETTE,
    _TYPE_ORDER,
    _bar_chart,
)


# ---------------------------------------------------------------------------
# ImageNet-specific helpers (run names differ from CelebA-HQ)
# ---------------------------------------------------------------------------

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
    if name.startswith("sae_matryoshka_batch_topk_") or "matryoshka" in name:
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
        "topk_multi_taxon_ae_imagenet_r18_",
        "topk_taxon_ae_imagenet_r18_",
        "bias_multi_taxon_ae_imagenet_r18_",
        "bias_taxon_ae_imagenet_r18_",
        "multi_taxon_ae_imagenet_r18_",
        "taxon_ae_imagenet_r18_",
        "sae_matryoshka_batch_topk_imagenet_r18_",
        "sae_softmax_imagenet_r18_",
        "sae_jumprelu_imagenet_r18_",
        "sae_topk_imagenet_r18_",
        "sae_gated_imagenet_r18_",
        "sae_imagenet_r18",
        "baseline_ae_imagenet_r18",
    ):
        name = name.replace(prefix, "").strip("_")
    cleaned = name.replace("_", " ").strip()
    return f"{mtype} ({cleaned})" if cleaned else mtype


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_results(paper_dir: Path) -> List[Dict]:
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
        out_run = ROOT / "outputs" / "imagenet" / name
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
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val recon")
    ax.legend(fontsize=5, ncol=2)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def build_page1_reconstruction(runs, save_path):
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4)
    _bar_chart(fig.add_subplot(gs[0, 0]), runs, "mse_mean", "Mean MSE (↓)", "MSE")
    _bar_chart(fig.add_subplot(gs[0, 1]), runs, "mae_mean", "Mean MAE (↓)", "MAE")
    _bar_chart(fig.add_subplot(gs[0, 2]), runs, "psnr_mean", "Mean PSNR (↑)", "PSNR (dB)",
               higher_is_better=True)
    _training_curves(fig.add_subplot(gs[1, :]), runs)
    fig.suptitle("Reconstruction Quality — ImageNet", fontsize=13, y=1.01)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def build_page2_sparsity(runs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _bar_chart(axes[0], runs, "l0_mean", "Mean L0 (active features, ↓)", "L0")
    _bar_chart(axes[1], runs, "dead_neuron_frac",
               "Dead neuron fraction (↑ = worse)", "Fraction", pct=True)
    fig.suptitle("Sparsity Metrics — ImageNet", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def build_page3_downstream(runs, save_path):
    has_probe = any(not np.isnan(r["metrics"].get("probe_top1_acc", float("nan")))
                    for r in runs)
    ncols = 3 if has_probe else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))
    if ncols == 1:
        axes = [axes]
    _bar_chart(axes[0], runs, "probe_top1_acc",
               "Linear probe top-1 acc (1000 cls, ↑)", "Acc", higher_is_better=True)
    if has_probe:
        _bar_chart(axes[1], runs, "knn_k1_top1_acc",
                   "1-NN top-1 acc (↑)", "Acc", higher_is_better=True)
        _bar_chart(axes[2], runs, "knn_k5_top1_acc",
                   "5-NN top-1 acc (↑)", "Acc", higher_is_better=True)
    fig.suptitle("Downstream Classification — ImageNet", fontsize=12)
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
        labels_path = r["dir"] / "labels.npy"
        if labels_path.exists():
            labels = np.load(labels_path)
            # Use at most 20 distinct colours
            unique_cls = np.unique(labels)[:20]
            cmap = plt.cm.get_cmap("tab20", len(unique_cls))
            for j, c in enumerate(unique_cls):
                mask = labels == c
                if mask.any() and len(emb) == len(labels):
                    axes[i].scatter(emb[mask, 0], emb[mask, 1], s=2, alpha=0.5,
                                    color=cmap(j))
        else:
            axes[i].scatter(emb[:, 0], emb[:, 1], s=2, alpha=0.4, color=r["colour"])
        axes[i].set_title(r["short"], fontsize=7)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("t-SNE of latent codes — ImageNet", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


def build_overview(runs, save_path):
    metric_keys = [
        ("mse_mean",        "MSE ↓",        False),
        ("psnr_mean",       "PSNR ↑",       True),
        ("l0_mean",         "L0 ↓",         False),
        ("dead_neuron_frac","Dead% ↑=worse", False),
        ("probe_top1_acc",  "Probe acc ↑",  True),
        ("knn_k5_top1_acc", "5-NN acc ↑",   True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for ax, (key, label, hib) in zip(axes, metric_keys):
        _bar_chart(ax, runs, key, label, label, higher_is_better=hib,
                   pct=(key == "dead_neuron_frac"))
    fig.suptitle("Model Comparison Overview — ImageNet", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comparison plots for ImageNet paper results")
    parser.add_argument("--paper-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir or (ROOT / "outputs" / "paper" / "imagenet"))
    save_dir = Path(args.save_dir or (paper_dir / "comparison"))
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {paper_dir} ...")
    runs = discover_results(paper_dir)
    if not runs:
        print("No per-model results found. Run paper_analysis_imagenet.py for each model first.")
        return
    print(f"Found {len(runs)} results: {[r['name'] for r in runs]}")

    # Summary CSV
    all_keys = sorted({k for r in runs for k in r["metrics"].keys()
                       if not isinstance(r["metrics"][k], list)})
    csv_path = save_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run"] + all_keys)
        for r in runs:
            row = [r["name"]] + [r["metrics"].get(k, "") for k in all_keys]
            writer.writerow(row)
    print(f"CSV    → {csv_path}")

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
