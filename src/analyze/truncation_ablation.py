#!/usr/bin/env python3
"""Hierarchy truncation ablation for taxonomy autoencoders.

For a *trained* taxonomy model, zero out the latent channels that correspond
to hierarchy depths > X in the last encoder stage, then measure how
reconstruction quality degrades as X decreases.  No retraining is performed.

Works for: taxon, topk_taxon, bias_taxon models (all share TaxonResNetEncoder).
Does NOT support multi_taxon / topk_multi_taxon / bias_multi_taxon (different
encoder architecture).

Results are saved under:
    <analysis_save_dir>/truncation_ablation/
        truncation_metrics.csv
        truncation_ablation.png

Usage (run from TaxonomicWeights/):
    python -m src.analyze.truncation_ablation --config <path/to/config.json>
    python -m src.analyze.truncation_ablation --config <path> \\
        --checkpoint <path/to/best.pt> --num-batches 50
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.analyze.analyze_celeba_hq_ae import load_model
from src.utils.dataloader import CelebAHQLoader


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hierarchy truncation ablation for taxonomy models"
    )
    p.add_argument("--config", required=True, help="Path to JSON config file")
    p.add_argument(
        "--checkpoint", default=None,
        help="Override checkpoint path (otherwise inferred from config)",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument(
        "--num-batches", type=int, default=50,
        help="Validation batches to evaluate per truncation level (default: 50)",
    )
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


# ── Config helpers ─────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_checkpoint_path(config: dict, override: Optional[str]) -> str:
    """Resolve checkpoint path with fallback chain:
    1. CLI --checkpoint
    2. config["analysis"]["checkpoint_path"]
    3. Derived: output_dir.parent / experiment_name / checkpoints / best.pt
    """
    if override:
        return override
    ckpt = config.get("analysis", {}).get("checkpoint_path")
    if ckpt:
        return ckpt
    output_dir = config.get("output", {}).get("output_dir")
    exp_name = config.get("experiment_name")
    if output_dir and exp_name:
        return str(Path(output_dir).parent / exp_name / "checkpoints" / "best.pt")
    raise ValueError(
        "Cannot resolve checkpoint path. Pass --checkpoint or set "
        "config['analysis']['checkpoint_path']."
    )


def resolve_save_dir(config: dict) -> Path:
    """Resolve where to save: config output.analysis_save_dir > derived."""
    analysis_save_dir = config.get("output", {}).get("analysis_save_dir")
    if analysis_save_dir:
        return Path(analysis_save_dir) / "truncation_ablation"
    output_dir = config.get("output", {}).get("output_dir")
    exp_name = config.get("experiment_name")
    if output_dir and exp_name:
        return Path(output_dir).parent / exp_name / "analysis" / "truncation_ablation"
    raise ValueError("Cannot resolve save dir from config.")


# ── Channel layout ─────────────────────────────────────────────────────────────

def channels_up_to_depth(keep_depths: int) -> int:
    """Return the number of channels occupied by the first `keep_depths` levels.

    Level layout (0-indexed): level d has 2^(d+1) channels.
    Cumulative: sum_{d=0}^{K-1} 2^(d+1) = 2^(K+1) - 2
    """
    return (1 << (keep_depths + 1)) - 2


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _run_batches(model, data_loader, device, num_batches, hook_fn=None):
    """Run inference for `num_batches` batches, optionally with a forward hook
    on the last encoder stage.  Returns (mse_list, mae_list)."""
    handle = None
    if hook_fn is not None:
        handle = model.encoder.taxon_stages[-1].register_forward_hook(hook_fn)

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


def evaluate_keep_depths(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    keep_depths: int,
    n_taxonomy_layers: int,
    num_batches: int,
) -> Dict[str, float]:
    """Evaluate reconstruction with only the first `keep_depths` levels active."""
    if keep_depths >= n_taxonomy_layers:
        # No truncation — run unmodified
        mse_list, mae_list = _run_batches(model, data_loader, device, num_batches)
    else:
        trunc_start = channels_up_to_depth(keep_depths)

        def _zero_deep(module, input, output):
            z, logps, stage_dict = output
            z = z.clone()
            z[:, trunc_start:, :, :] = 0.0
            return (z, logps, stage_dict)

        mse_list, mae_list = _run_batches(
            model, data_loader, device, num_batches, hook_fn=_zero_deep
        )

    return {
        "mean_mse": float(np.mean(mse_list)),
        "std_mse":  float(np.std(mse_list)),
        "mean_mae": float(np.mean(mae_list)),
        "std_mae":  float(np.std(mae_list)),
        "n_samples": len(mse_list),
    }


def run_ablation(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    n_taxonomy_layers: int,
    num_batches: int,
) -> List[Dict]:
    """Sweep truncation from keep_depths=1 to keep_depths=n_taxonomy_layers."""
    results = []
    for keep in range(1, n_taxonomy_layers + 1):
        is_full = keep == n_taxonomy_layers
        label = f"all {n_taxonomy_layers} (full)" if is_full else f"depth ≤ {keep}"
        print(f"  keep={keep}/{n_taxonomy_layers}  [{label}] ...", end=" ", flush=True)
        m = evaluate_keep_depths(
            model, data_loader, device, keep, n_taxonomy_layers, num_batches
        )
        m["keep_depths"] = keep
        m["label"] = label
        results.append(m)
        print(f"MSE={m['mean_mse']:.5f}  MAE={m['mean_mae']:.5f}")
    return results


# ── Saving ─────────────────────────────────────────────────────────────────────

def save_results(
    results: List[Dict],
    save_dir: Path,
    n_taxonomy_layers: int,
    experiment_name: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = save_dir / "truncation_metrics.csv"
    fieldnames = [
        "keep_depths", "label",
        "mean_mse", "std_mse",
        "mean_mae", "std_mae",
        "n_samples",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV:  {csv_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    xs       = [r["keep_depths"] for r in results]
    mses     = [r["mean_mse"]    for r in results]
    mse_errs = [r["std_mse"]     for r in results]
    maes     = [r["mean_mae"]    for r in results]
    mae_errs = [r["std_mae"]     for r in results]

    full_mse = next(r["mean_mse"] for r in results if r["keep_depths"] == n_taxonomy_layers)
    full_mae = next(r["mean_mae"] for r in results if r["keep_depths"] == n_taxonomy_layers)

    # % of full reconstruction quality retained (higher = better at that depth)
    pct_mse = [min(full_mse / m, 1.0) * 100 if m > 0 else 0.0 for m in mses]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1 — MSE
    ax = axes[0]
    ax.errorbar(xs, mses, yerr=mse_errs, marker="o", color="#1f77b4",
                capsize=4, linewidth=2, markersize=7)
    ax.axhline(full_mse, color="grey", linestyle="--", linewidth=1.0, label="Full model")
    ax.set_xlabel("Hierarchy levels kept  (last encoder stage)", fontsize=11)
    ax.set_ylabel("Mean MSE", fontsize=11)
    ax.set_title("MSE  vs.  Depth kept", fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2 — MAE
    ax = axes[1]
    ax.errorbar(xs, maes, yerr=mae_errs, marker="s", color="#d62728",
                capsize=4, linewidth=2, markersize=7)
    ax.axhline(full_mae, color="grey", linestyle="--", linewidth=1.0, label="Full model")
    ax.set_xlabel("Hierarchy levels kept  (last encoder stage)", fontsize=11)
    ax.set_ylabel("Mean MAE", fontsize=11)
    ax.set_title("MAE  vs.  Depth kept", fontsize=12, fontweight="bold")
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3 — % quality retained
    ax = axes[2]
    ax.plot(xs, pct_mse, marker="D", color="#2ca02c", linewidth=2, markersize=7)
    ax.axhline(100.0, color="grey", linestyle="--", linewidth=1.0, label="Full model (100%)")
    ax.fill_between(xs, pct_mse, 100.0, alpha=0.12, color="#2ca02c")
    ax.set_xlabel("Hierarchy levels kept  (last encoder stage)", fontsize=11)
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
        f"Hierarchy Truncation Ablation — {experiment_name}\n"
        f"Last encoder stage: {n_taxonomy_layers} depth levels  |  "
        "deeper channels zeroed at inference, no retraining",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    plot_path = save_dir / "truncation_ablation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {plot_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    save_dir        = resolve_save_dir(config)
    exp_name        = config.get("experiment_name", "unknown")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Save dir:   {save_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    model, _ = load_model(checkpoint_path, device, config)
    model.eval()

    # ── Inspect last encoder stage ────────────────────────────────────────────
    last_stage = model.encoder.taxon_stages[-1]
    if not hasattr(last_stage, "n_taxonomy_layers"):
        raise RuntimeError(
            "The last encoder stage has no 'n_taxonomy_layers' attribute. "
            "Multi-taxon models (MultiTaxonResNetStage) are not supported — "
            "use a taxon / topk_taxon / bias_taxon config."
        )
    n_taxonomy_layers = last_stage.n_taxonomy_layers
    print(
        f"Last stage: {n_taxonomy_layers} depth levels, "
        f"{last_stage.total_out_channels} total channels "
        f"(layout: {last_stage.layer_channels})"
    )

    # ── Data loader ───────────────────────────────────────────────────────────
    dc = config.get("data", {})
    data_root   = args.data_root or dc.get("data_root", "./data/celeba_hq")
    batch_size  = args.batch_size or dc.get("batch_size", 32)
    image_size  = dc.get("image_size", 256)
    val_split   = dc.get("val_split", 0.05)

    loader_obj = CelebAHQLoader(
        data_root=data_root,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        val_split=val_split,
    )
    _, val_loader = loader_obj.get_loaders()
    if val_loader is None:
        raise RuntimeError("No validation split found — check data_root or val_split.")

    # ── Run ablation ──────────────────────────────────────────────────────────
    print(
        f"\nRunning truncation ablation  "
        f"({n_taxonomy_layers} levels × {args.num_batches} batches each)..."
    )
    results = run_ablation(model, val_loader, device, n_taxonomy_layers, args.num_batches)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving to {save_dir} ...")
    save_results(results, save_dir, n_taxonomy_layers, exp_name)
    print("Done.")


if __name__ == "__main__":
    main()
