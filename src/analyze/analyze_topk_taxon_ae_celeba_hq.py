#!/usr/bin/env python3
"""Analysis script for the TopK Taxon Autoencoder on CelebA-HQ.

Thin wrapper that reuses variant-independent analyses from
``analyze_celeba_hq_ae.py`` with the correct TopK-specific directory
naming convention (``_auxk_{W:.0e}``).
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

# Reuse all analysis functions from the vanilla script
from src.analyze.analyze_celeba_hq_ae import (
    load_config,
    load_model,
    visualize_stage_filters,
    analyze_filter_similarity,
    visualize_taxonomy_distributions,
    visualize_taxonomy_path_probs,
    visualize_taxonomy_tree,
    analyze_parse_tree,
    analyze_hierarchical_activations,
    analyze_binary_split_maps,
    analyze_latent_sparsity,
    analyze_reconstruction_quality,
    visualize_multiple_reconstructions,
    analyze_partonomy_sparsity,
    visualize_stage_activations,
)



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

    parser = argparse.ArgumentParser(description="Analyze TopK Taxon AE on CelebA-HQ")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output-dir", type=str,
                        default=o.get("output_dir", "./outputs/topk_taxon_ae_celeba_hq_r18"))
    parser.add_argument("--analysis-save-dir", type=str, default=o.get("analysis_save_dir", ""))
    parser.add_argument("--data-root", type=str, default=d.get("data_root", "./data/celeba_hq"))
    parser.add_argument("--image-size", type=int, default=d.get("image_size", 256))
    parser.add_argument("--batch-size", type=int, default=d.get("batch_size", 16))
    parser.add_argument("--num-workers", type=int, default=d.get("num_workers", 4))
    parser.add_argument("--val-split", type=float, default=d.get("val_split", 0.05))
    parser.add_argument("--seed", type=int, default=cfg.get("training", {}).get("seed", 42))
    # TopK-specific
    parser.add_argument("--auxk-weight", type=float, default=cfg.get("training", {}).get("auxk_weight", 0.0))
    # Analysis knobs
    parser.add_argument("--n-latent-batches", type=int, default=a.get("num_latent_batches", 50))
    parser.add_argument("--n-recon-batches", type=int, default=a.get("num_reconstruction_batches", 20))
    parser.add_argument("--n-recon-images", type=int, default=a.get("num_multiple_recon_images", 8))
    parser.add_argument("--n-recon-sets", type=int, default=a.get("num_reconstructions_per_image", 8))
    parser.add_argument("--n-act-images", type=int, default=a.get("num_activation_images", 3))
    parser.add_argument("--n-taxon-batches", type=int, default=a.get("num_taxonomy_batches", 10))
    parser.add_argument("--n-parse-images", type=int, default=a.get("num_parse_tree_images", 4))
    parser.add_argument("--n-hier-images", type=int, default=a.get("num_hier_act_images", 4))
    parser.add_argument("--max-hier-depth", type=int, default=a.get("max_hier_act_depth", 4))
    parser.add_argument("--n-split-images", type=int, default=a.get("num_split_map_images", 4))
    parser.add_argument("--max-split-pairs", type=int, default=a.get("max_split_pairs", 8))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if args.config else {}

    # ── run suffix (must match training script) ──
    auxk_str = f"_auxk_{args.auxk_weight:.0e}" if args.auxk_weight else ""
    run_suffix = auxk_str

    output_dir = Path(args.output_dir + run_suffix)
    analysis_dir = Path(args.analysis_save_dir or str(output_dir / "analysis"))
    analysis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("TopK Taxon Autoencoder — Analysis")
    print("=" * 80)
    print(f"  device={device}  run_suffix={run_suffix}")
    print(f"  output_dir={output_dir}")
    print(f"  analysis_dir={analysis_dir}")
    print("=" * 80)

    # Load model
    ckpt = args.checkpoint or str(output_dir / "checkpoints" / "best.pt")
    # Ensure config has model_variant so load_model picks TopK
    config.setdefault("model", {})["model_variant"] = "topk"
    model, _ = load_model(ckpt, device, config)

    if _is_cifar(config):
        loader = CIFAR10Loader(batch_size=args.batch_size, root=args.data_root)
        _, eval_loader = loader.get_loaders()
    else:
        tf = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        loader = CelebAHQLoader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            val_split=args.val_split,
            seed=args.seed,
            pin_memory=(device.type == "cuda"),
            transform=tf,
        )
        _, val_loader = loader.get_loaders()
        eval_loader = val_loader if val_loader is not None else loader.train_loader
    save_dir = str(analysis_dir)

    print("\n[1] Stage filter visualisation")
    visualize_stage_filters(model, save_dir)

    print("\n[1b] Filter similarity analysis")
    analyze_filter_similarity(model, save_dir)

    print("\n[2] Taxonomy probability distributions")
    visualize_taxonomy_distributions(model, eval_loader, device, save_dir,
                                     num_batches=args.n_taxon_batches)

    print("\n[2b] Taxonomy path probability plots")
    visualize_taxonomy_path_probs(model, eval_loader, device, save_dir)

    print("\n[2c] Taxonomy tree activations")
    visualize_taxonomy_tree(model, eval_loader, device, save_dir,
                            num_images=args.n_act_images)

    print("\n[2d] Parse tree analysis")
    analyze_parse_tree(model, eval_loader, device, save_dir,
                       num_images=args.n_parse_images)

    print("\n[2e] Hierarchical activation maps")
    analyze_hierarchical_activations(model, eval_loader, device, save_dir,
                                     num_images=args.n_hier_images,
                                     max_depth=args.max_hier_depth)

    print("\n[2f] Binary split maps")
    analyze_binary_split_maps(model, eval_loader, device, save_dir,
                              num_images=args.n_split_images,
                              max_depth=args.max_hier_depth,
                              max_pairs=args.max_split_pairs)

    print("\n[3] Latent space sparsity")
    analyze_latent_sparsity(model, eval_loader, device, save_dir,
                            num_batches=args.n_latent_batches)

    print("\n[4] Reconstruction quality")
    analyze_reconstruction_quality(model, eval_loader, device, save_dir,
                                   num_batches=args.n_recon_batches)

    print("\n[5] Multiple reconstructions")
    visualize_multiple_reconstructions(model, eval_loader, device, save_dir,
                                       num_images=args.n_recon_images,
                                       num_sets=args.n_recon_sets)

    print("\n[6] Partonomy sparsity suite")
    analyze_partonomy_sparsity(model, eval_loader, device, save_dir)

    print("\n[7] Stage activation maps")
    visualize_stage_activations(model, eval_loader, device, save_dir,
                                num_images=args.n_act_images)

    print("\n" + "=" * 80)
    print(f"Analysis complete! All outputs saved to: {analysis_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
