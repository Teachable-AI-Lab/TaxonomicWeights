#!/usr/bin/env python3
"""Analysis script for the Bias (Sigmoid+Bias) Multi-Taxon Autoencoder on CelebA-HQ.

Thin wrapper that reuses variant-independent analyses from
``analyze_multi_taxon_ae_celeba_hq.py`` with the correct Bias-specific
directory naming convention (``_k{K}_K{H}_bur_{R:.0e}``).

Taxonomy-specific analyses (parse trees, binary splits, taxonomy
distributions, gate distributions, per-stage regs) are skipped because Bias
routing does not use pairwise softmax or vanilla regularisation terms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.dataloader import CelebAHQLoader, CIFAR10Loader

from src.analyze.analyze_multi_taxon_ae_celeba_hq import (
    load_config,
    load_model,
    save_training_curves,
    visualize_stage_filters,
    analyze_filter_similarity,
    visualize_stage_activations,
    analyze_reconstruction_quality,
    visualize_multiple_reconstructions,
    analyze_latent_sparsity,
    analyze_partonomy_sparsity,
    analyze_cross_hierarchy_similarity,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@torch.no_grad()
def visualize_bias_terms(model, save_dir: str) -> None:
    """Bar-plot of final leaf bias, EMA load, and gate bias per stage."""
    stages = model.encoder.multi_taxon_stages
    n_stages = len(stages)

    # Per-hierarchy leaf biases
    for s_idx, stage in enumerate(stages):
        n_hier = len(stage.hierarchies)
        fig, axes = plt.subplots(n_hier, 2, figsize=(14, 3 * n_hier), squeeze=False)
        for h_idx, hier in enumerate(stage.hierarchies):
            bias = hier._node_bias.cpu().numpy()
            ema = hier._ema_load.cpu().numpy()
            n_leaves = len(bias)
            xs = np.arange(n_leaves)

            ax_b = axes[h_idx, 0]
            colors = ["tab:red" if v < 0 else "tab:blue" for v in bias]
            ax_b.bar(xs, bias, color=colors, edgecolor="black", linewidth=0.3)
            ax_b.axhline(0, color="black", linewidth=0.5)
            ax_b.set_title(f"Hierarchy {h_idx+1} — Leaf Biases ({n_leaves} leaves)", fontsize=10)
            ax_b.set_xlabel("Leaf index")
            ax_b.set_ylabel("Bias value")
            ax_b.grid(True, alpha=0.3, axis="y")

            ax_e = axes[h_idx, 1]
            avg = float(ema.mean())
            ax_e.bar(xs, ema, color="tab:green", edgecolor="black", linewidth=0.3)
            ax_e.axhline(avg, color="red", linestyle="--", linewidth=1.0,
                         label=f"avg={avg:.4f}")
            ax_e.set_title(f"Hierarchy {h_idx+1} — EMA Load ({n_leaves} leaves)", fontsize=10)
            ax_e.set_xlabel("Leaf index")
            ax_e.set_ylabel("EMA load")
            ax_e.legend(fontsize=8)
            ax_e.grid(True, alpha=0.3, axis="y")

        plt.suptitle(f"Stage {s_idx+1} — Leaf Bias Terms & EMA Load",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(save_dir, f"bias_terms_stage{s_idx+1}_leaves.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved to {out}")

    # Gate biases (one plot for all stages)
    fig, axes = plt.subplots(1, n_stages, figsize=(5 * n_stages, 4), squeeze=False)
    for s_idx, stage in enumerate(stages):
        gate_b = stage._gate_bias.cpu().numpy()
        gate_e = stage._gate_ema_load.cpu().numpy()
        n_h = len(gate_b)
        xs = np.arange(n_h)
        ax = axes[0, s_idx]
        ax.bar(xs - 0.15, gate_b, width=0.3, label="gate bias", color="tab:blue",
               edgecolor="black", linewidth=0.3)
        ax.bar(xs + 0.15, gate_e, width=0.3, label="gate ema", color="tab:green",
               edgecolor="black", linewidth=0.3)
        ax.set_title(f"Stage {s_idx+1} — Gate Biases", fontsize=10)
        ax.set_xlabel("Hierarchy")
        ax.set_xticks(xs)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    plt.suptitle("Gate Bias Terms", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "bias_terms_gates.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved to {out}")

    # Dump numeric values to JSON
    import json as _json
    data = {}
    for s_idx, stage in enumerate(stages):
        stage_data = {
            "gate_bias": stage._gate_bias.cpu().tolist(),
            "gate_ema_load": stage._gate_ema_load.cpu().tolist(),
            "hierarchies": {},
        }
        for h_idx, hier in enumerate(stage.hierarchies):
            stage_data["hierarchies"][f"h{h_idx+1}"] = {
                "leaf_bias": hier._node_bias.cpu().tolist(),
                "ema_load": hier._ema_load.cpu().tolist(),
            }
        data[f"stage_{s_idx+1}"] = stage_data
    json_path = os.path.join(save_dir, "bias_terms.json")
    with open(json_path, "w") as f:
        _json.dump(data, f, indent=2)
    print(f"  Saved to {json_path}")



def _is_cifar(cfg: dict) -> bool:
    """Infer dataset type from the config (image_size==32 -> CIFAR-10)."""
    return cfg.get("data", {}).get("image_size", 256) == 32

def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args()
    cfg: dict = {}
    if pre_args.config:
        cfg = load_config(pre_args.config)
    d = cfg.get("data", {})
    m = cfg.get("model", {})
    a = cfg.get("analysis", {})
    o = cfg.get("output", {})

    parser = argparse.ArgumentParser(description="Analyze Bias MultiTaxon AE on CelebA-HQ")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output-dir", type=str,
                        default=o.get("output_dir", "./outputs/bias_multi_taxon_ae_celeba_hq_r18"))
    parser.add_argument("--analysis-save-dir", type=str, default=o.get("analysis_save_dir", ""))
    parser.add_argument("--data-root", type=str, default=d.get("data_root", "./data/celeba_hq"))
    parser.add_argument("--image-size", type=int, default=d.get("image_size", 256))
    parser.add_argument("--batch-size", type=int, default=d.get("batch_size", 16))
    parser.add_argument("--num-workers", type=int, default=d.get("num_workers", 4))
    parser.add_argument("--val-split", type=float, default=d.get("val_split", 0.05))
    parser.add_argument("--seed", type=int, default=cfg.get("training", {}).get("seed", 42))
    # Bias-specific
    parser.add_argument("--k", type=int, default=m.get("k", None))
    parser.add_argument("--n-hierarchies", type=int, default=m.get("n_hierarchies", 3))
    parser.add_argument("--bias-update-rate", type=float, default=m.get("bias_update_rate", 0.001))
    # Analysis knobs
    parser.add_argument("--n-latent-batches", type=int, default=a.get("num_latent_batches", 50))
    parser.add_argument("--n-recon-batches", type=int, default=a.get("num_reconstruction_batches", 20))
    parser.add_argument("--n-recon-images", type=int, default=a.get("num_multiple_recon_images", 8))
    parser.add_argument("--n-recon-sets", type=int, default=a.get("num_reconstructions_per_image", 8))
    parser.add_argument("--n-act-images", type=int, default=a.get("num_activation_images", 3))
    parser.add_argument("--n-analysis-batches", type=int, default=a.get("num_taxonomy_batches", 10))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else {}

    # ── run suffix (must match training script) ──
    k_str = f"_k{args.k}" if args.k is not None else ""
    hier_str = f"_K{args.n_hierarchies}"
    bur_str = f"_bur_{args.bias_update_rate:.0e}"
    run_suffix = k_str + hier_str + bur_str

    output_dir = Path(args.output_dir + run_suffix)
    analysis_dir = Path(args.analysis_save_dir or str(output_dir / "analysis"))
    analysis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Bias MultiTaxon Autoencoder — Analysis")
    print("=" * 80)
    print(f"  device={device}  run_suffix={run_suffix}")
    print(f"  output_dir={output_dir}")
    print(f"  analysis_dir={analysis_dir}")
    print("=" * 80)

    # B1. Training curves
    print("\n[B1] Training curves")
    history_path = output_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        save_training_curves(history, str(analysis_dir))
    else:
        print(f"  No training_history.json at {history_path}, skipping.")

    # Load model
    ckpt = args.checkpoint or str(output_dir / "checkpoints" / "best.pt")
    config.setdefault("model", {})["model_variant"] = "bias"
    model, _ = load_model(ckpt, device, config)

    if _is_cifar(config):
        data_loader = CIFAR10Loader(batch_size=args.batch_size, root=args.data_root)
        _, val_loader = data_loader.get_loaders()
    else:
        tf = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        data_loader = CelebAHQLoader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            val_split=args.val_split,
            seed=args.seed,
            pin_memory=(device.type == "cuda"),
            transform=tf,
        )
        _, val_loader = data_loader.get_loaders()
        if val_loader is None:
            raise RuntimeError("val_split must be > 0 to produce a validation loader")

    save_dir = str(analysis_dir)

    print("\n[A1] Stage filter visualisation")
    visualize_stage_filters(model, save_dir)

    print("\n[A1b] Filter similarity analysis")
    analyze_filter_similarity(model, save_dir)

    print("\n[A8] Stage activation maps")
    visualize_stage_activations(model, val_loader, device, save_dir,
                                num_images=args.n_act_images)

    print("\n[B2] Reconstruction quality")
    analyze_reconstruction_quality(model, val_loader, device, save_dir,
                                   num_batches=args.n_recon_batches)

    print("\n[B3] Multiple reconstructions")
    visualize_multiple_reconstructions(model, val_loader, device, save_dir,
                                       num_images=args.n_recon_images,
                                       num_sets=args.n_recon_sets)

    print("\n[B4] Latent space sparsity")
    analyze_latent_sparsity(model, val_loader, device, save_dir,
                            num_batches=args.n_latent_batches)

    print("\n[B5] Partonomy sparsity suite")
    analyze_partonomy_sparsity(model, val_loader, device, save_dir)

    print("\n[C2] Cross-hierarchy cosine similarity")
    analyze_cross_hierarchy_similarity(model, val_loader, device, save_dir,
                                       n_batches=args.n_analysis_batches)

    print("\n[C4] Bias terms (leaf biases, EMA loads & gate biases)")
    visualize_bias_terms(model, save_dir)

    print("\n" + "=" * 80)
    print(f"Analysis complete! All outputs saved to: {analysis_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
