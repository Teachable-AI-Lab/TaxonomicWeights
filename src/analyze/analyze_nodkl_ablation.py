#!/usr/bin/env python3
"""Analyze DKL=0 ablation results.

Reads the ablation logs produced by ``train_nodkl_ablation_taxon_ae.py``
and generates visualizations of:

1. Per-node gradient magnitude heatmaps over training epochs (per stage/depth).
2. Node usage bar charts (per stage/depth) at selected epochs.
3. Summary time-series: Gini coefficient, top-1 fraction, dead fraction.
4. Full-model gradient and usage snapshots at the final logged epoch.

Usage:
    python src/analyze/analyze_nodkl_ablation.py --output-dir <run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_ablation_history(ablation_dir: Path) -> dict:
    p = ablation_dir / "ablation_history.json"
    if not p.exists():
        raise FileNotFoundError(f"ablation_history.json not found in {ablation_dir}")
    with open(p) as f:
        return json.load(f)


def load_epoch_npz(ablation_dir: Path, prefix: str, epoch: int) -> Dict[str, np.ndarray]:
    p = ablation_dir / f"{prefix}_epoch{epoch:03d}.npz"
    if not p.exists():
        return {}
    data = np.load(p)
    return {k: data[k] for k in data.files}


def discover_logged_epochs(ablation_dir: Path) -> List[int]:
    epochs = set()
    for p in ablation_dir.glob("gradient_stats_epoch*.npz"):
        try:
            ep = int(p.stem.split("epoch")[1])
            epochs.add(ep)
        except (ValueError, IndexError):
            continue
    return sorted(epochs)


def _parse_key(key: str):
    """Parse 'stage0_depth3' -> (stage_idx, depth_idx)."""
    parts = key.split("_")
    s_idx = int(parts[0].replace("stage", ""))
    d_idx = int(parts[1].replace("depth", ""))
    return s_idx, d_idx


# ---------------------------------------------------------------------------
# Plot: time-series summary
# ---------------------------------------------------------------------------

def plot_summary_timeseries(history: dict, save_dir: Path) -> None:
    epochs = history["epochs"]
    if not epochs:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Taxonomy conv-layer weight gradient stats
    panels_grad = [
        ("Conv Weight Grad Gini", "grad_gini"),
        ("Conv Weight Grad Top-1 Fraction", "grad_top1_frac"),
        ("Conv Weight Grad Near-Zero Fraction", "grad_zero_frac"),
    ]
    for ax, (title, key) in zip(axes[0], panels_grad):
        ax.plot(epochs, history[key], marker="o", markersize=4, linewidth=1.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.split("_", 1)[1].replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    # Usage stats
    panels_usage = [
        ("Usage Gini", "usage_gini"),
        ("Dead Node Fraction", "usage_dead_frac"),
        ("Usage Top-1 Fraction", "usage_top1_frac"),
    ]
    for ax, (title, key) in zip(axes[1], panels_usage):
        ax.plot(epochs, history[key], marker="s", markersize=4, linewidth=1.5,
                color="tab:orange")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key.split("_", 1)[1].replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle("DKL=0 Ablation Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "summary_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_dir / 'summary_timeseries.png'}")


# ---------------------------------------------------------------------------
# Plot: gradient heatmaps over time
# ---------------------------------------------------------------------------

def plot_gradient_heatmaps(ablation_dir: Path, save_dir: Path) -> None:
    """For each stage/depth, create a heatmap [epochs x nodes] of gradient magnitude."""
    epochs = discover_logged_epochs(ablation_dir)
    if not epochs:
        return

    # Gather all keys from first epoch
    first = load_epoch_npz(ablation_dir, "gradient_stats", epochs[0])
    if not first:
        return

    keys = sorted(first.keys(), key=lambda k: _parse_key(k))

    for key in keys:
        n_channels = len(first[key])
        mat = np.zeros((len(epochs), n_channels))
        for i, ep in enumerate(epochs):
            data = load_epoch_npz(ablation_dir, "gradient_stats", ep)
            if key in data:
                mat[i] = data[key]

        fig, ax = plt.subplots(figsize=(max(6, n_channels * 0.25), max(4, len(epochs) * 0.3)))
        im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="hot")
        ax.set_xlabel("Node (channel)")
        ax.set_ylabel("Epoch")
        ax.set_yticks(range(len(epochs)))
        ax.set_yticklabels(epochs)
        ax.set_title(f"Mean |grad| — {key}", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="|grad|")
        plt.tight_layout()
        plt.savefig(save_dir / f"grad_heatmap_{key}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved {len(keys)} gradient heatmaps to {save_dir}")


# ---------------------------------------------------------------------------
# Plot: node usage bar charts at selected epochs
# ---------------------------------------------------------------------------

def plot_usage_bars(ablation_dir: Path, save_dir: Path, max_snapshots: int = 6) -> None:
    """Bar charts of node usage counts at evenly spaced epochs."""
    epochs = discover_logged_epochs(ablation_dir)
    if not epochs:
        return

    # Pick subset of epochs to show
    if len(epochs) > max_snapshots:
        indices = np.linspace(0, len(epochs) - 1, max_snapshots, dtype=int)
        selected_epochs = [epochs[i] for i in indices]
    else:
        selected_epochs = epochs

    first = load_epoch_npz(ablation_dir, "node_usage", selected_epochs[0])
    if not first:
        return

    keys = sorted(first.keys(), key=lambda k: _parse_key(k))

    for key in keys:
        n_channels = len(first[key])
        fig, axes = plt.subplots(1, len(selected_epochs),
                                  figsize=(4 * len(selected_epochs), 4),
                                  sharey=True)
        if len(selected_epochs) == 1:
            axes = [axes]

        for ax, ep in zip(axes, selected_epochs):
            data = load_epoch_npz(ablation_dir, "node_usage", ep)
            counts = data.get(key, np.zeros(n_channels))
            total = counts.sum()
            fracs = counts / max(1, total)

            ax.bar(range(n_channels), fracs, color="steelblue", edgecolor="none")
            ax.set_title(f"Epoch {ep}", fontsize=10)
            ax.set_xlabel("Node")
            if ax is axes[0]:
                ax.set_ylabel("Usage fraction")
            ax.set_ylim(0, max(0.1, fracs.max() * 1.2))

            # Annotate dead nodes
            n_dead = int((counts == 0).sum())
            ax.text(0.95, 0.95, f"dead: {n_dead}/{n_channels}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        plt.suptitle(f"Node usage — {key}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_dir / f"usage_bars_{key}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved {len(keys)} usage bar-chart strips to {save_dir}")


# ---------------------------------------------------------------------------
# Plot: per-epoch full snapshot (gradient + usage)
# ---------------------------------------------------------------------------

def plot_epoch_snapshot(ablation_dir: Path, save_dir: Path, epoch: int, label: str) -> None:
    """Combined gradient + usage bar charts for a given logged epoch."""
    grad_data = load_epoch_npz(ablation_dir, "gradient_stats", epoch)
    usage_data = load_epoch_npz(ablation_dir, "node_usage", epoch)

    if not grad_data or not usage_data:
        print(f"Warning: missing data for epoch {epoch}, skipping {label} snapshot.")
        return

    keys = sorted(set(grad_data.keys()) & set(usage_data.keys()),
                  key=lambda k: _parse_key(k))

    n_keys = len(keys)
    fig, axes = plt.subplots(n_keys, 2, figsize=(14, 3 * n_keys), squeeze=False)

    for row, key in enumerate(keys):
        grads = grad_data[key]
        counts = usage_data[key]
        total = counts.sum()
        fracs = counts / max(1, total)
        n_ch = len(grads)

        # Gradient magnitude
        ax0 = axes[row, 0]
        ax0.bar(range(n_ch), grads, color="salmon", edgecolor="none")
        ax0.set_title(f"{key} — |grad|", fontsize=10, fontweight="bold")
        ax0.set_ylabel("Mean |grad|")
        if row == n_keys - 1:
            ax0.set_xlabel("Node")

        # Usage
        ax1 = axes[row, 1]
        ax1.bar(range(n_ch), fracs, color="steelblue", edgecolor="none")
        ax1.set_title(f"{key} — usage frac", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Fraction")
        if row == n_keys - 1:
            ax1.set_xlabel("Node")

    plt.suptitle(f"{label} (epoch {epoch})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"{label.lower().replace(' ', '_')}_snapshot.png"
    plt.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname} for epoch {epoch}")


# ---------------------------------------------------------------------------
# Plot: final-epoch full snapshot (kept for backward compatibility)
# ---------------------------------------------------------------------------

def plot_final_snapshot(ablation_dir: Path, save_dir: Path) -> None:
    """Combined gradient + usage for the last logged epoch."""
    epochs = discover_logged_epochs(ablation_dir)
    if not epochs:
        return
    plot_epoch_snapshot(ablation_dir, save_dir, epochs[-1], "Final Snapshot")


def plot_first_snapshot(ablation_dir: Path, save_dir: Path) -> None:
    """Combined gradient + usage for the first logged epoch."""
    epochs = discover_logged_epochs(ablation_dir)
    if not epochs:
        return
    plot_epoch_snapshot(ablation_dir, save_dir, epochs[0], "First Epoch Snapshot")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze DKL=0 ablation results")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Run output directory (contains ablation_logs/)")
    parser.add_argument("--save-dir", type=str, default="",
                        help="Where to save plots (default: <output-dir>/ablation_analysis)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ablation_dir = output_dir / "ablation_logs"

    if not ablation_dir.exists():
        raise FileNotFoundError(f"No ablation_logs/ directory found in {output_dir}")

    save_dir = Path(args.save_dir) if args.save_dir else output_dir / "ablation_analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ablation dir: {ablation_dir}")
    print(f"Saving analysis to: {save_dir}")

    history = load_ablation_history(ablation_dir)
    plot_summary_timeseries(history, save_dir)
    plot_gradient_heatmaps(ablation_dir, save_dir)
    plot_usage_bars(ablation_dir, save_dir)
    plot_first_snapshot(ablation_dir, save_dir)
    plot_final_snapshot(ablation_dir, save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
