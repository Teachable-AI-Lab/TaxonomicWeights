#!/usr/bin/env python3
"""Analyze a trained BaselineConvAutoencoder checkpoint.

Computes the same latent and reconstruction metrics written by the taxon
analysis pipeline so that compare_celeba_hq.py / compare_cifar10.py can
load them from pre-computed npz files instead of computing live:

    outputs/<run>/analysis/latent_statistics.npz
    outputs/<run>/analysis/reconstruction_metrics.npz

All paths are read from the config JSON (output.output_dir /
output.analysis_save_dir, data.*).  A checkpoint or save-dir override can
be passed explicitly if needed.

Usage:
    # CelebA-HQ
    python src/analyze/analyze_baseline_ae.py --config configs/baseline_ae_celeba_hq.json

    # CIFAR-10
    python src/analyze/analyze_baseline_ae.py --config configs/baseline_ae_cifar10.json

    # Custom checkpoint
    python src/analyze/analyze_baseline_ae.py \\
        --config     configs/baseline_ae_celeba_hq.json \\
        --checkpoint ./outputs/baseline_ae_celeba_hq_r18/checkpoints/epoch_50.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.baseline.baseline_ae import BaselineConvAutoencoder
from src.utils.dataloader import CelebAHQLoader, CIFAR10Loader


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> BaselineConvAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = BaselineConvAutoencoder(
        in_channels=a.get("in_channels", 3),
        resnet_variant=a.get("resnet_variant", "18"),
        stage_channels=tuple(a.get("stage_channels", [64, 128, 256, 512])),
        stage_strides=tuple(a.get("stage_strides", [1, 2, 2, 2])),
        stage_blocks=a.get("stage_blocks", None),
        kernel_size=a.get("kernel_size", 3),
        use_stem=a.get("use_stem", True),
        stem_channels=a.get("stem_channels", 64),
        stem_stride=a.get("stem_stride", 2),
        use_stem_maxpool=a.get("use_stem_maxpool", True),
        output_activation=a.get("output_activation", "none"),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}  best_val={ckpt.get('best_val','?'):.6f}")
    return model


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    model: BaselineConvAutoencoder,
    loader,
    device: torch.device,
    max_batches: int = 0,
    sparsity_threshold: float = 0.1,
):
    """Compute latent and reconstruction metrics over a data loader.

    Returns a tuple (latent_stats, recon_stats) matching the format expected
    by the comparison scripts.
    """
    all_latents = []
    all_mse = []
    all_mae = []

    for b_idx, (imgs, _) in enumerate(tqdm(loader, desc="  Computing metrics")):
        if max_batches > 0 and b_idx >= max_batches:
            break
        imgs = imgs.to(device)
        recon, _ = model(imgs)
        z, _     = model.encode(imgs)

        # Reconstruction metrics
        all_mse.extend(((imgs - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
        all_mae.extend(torch.abs(imgs - recon).mean(dim=(1, 2, 3)).cpu().numpy())

        # Latent (flatten spatial dims)
        zn = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        all_latents.append(zn)

    Z   = np.concatenate(all_latents, axis=0)         # [N, D]
    mse = np.array(all_mse)
    mae = np.array(all_mae)

    sp  = np.mean(np.abs(Z) < sparsity_threshold, axis=1)     # per-sample sparsity
    l0  = np.sum(np.abs(Z) > sparsity_threshold, axis=1)
    l1  = np.abs(Z).sum(axis=1)
    l2  = np.sqrt((Z ** 2).sum(axis=1))

    lts = np.mean(np.abs(Z) < sparsity_threshold, axis=0)     # lifetime sparsity [D]
    dead      = float((lts > (1 - 0.01)).mean())
    selective = float(((lts > (1 - 0.2)) & (lts <= (1 - 0.01))).mean())
    moderate  = float(((lts > (1 - 0.5)) & (lts <= (1 - 0.2))).mean())
    dense     = float((lts <= (1 - 0.5)).mean())

    act  = 1.0 - lts   # fraction of samples where each feature is active
    mean_act = Z.mean(axis=0)
    std_act  = Z.std(axis=0)

    latent_stats = {
        "mean_activation":    mean_act.astype(np.float32),
        "std_activation":     std_act.astype(np.float32),
        "sparsity_per_sample":sp.astype(np.float32),
        "l0_norm":            l0.astype(np.float32),
        "l1_norm":            l1.astype(np.float32),
        "l2_norm":            l2.astype(np.float32),
        "mean_sparsity":      np.float32(sp.mean()),
        "mean_l0":            np.float32(l0.mean()),
        "mean_l1":            np.float32(l1.mean()),
        "mean_l2":            np.float32(l2.mean()),
        "latent_dim":         np.int64(Z.shape[1]),
        "num_samples":        np.int64(Z.shape[0]),
        "lifetime_sparsity":  lts.astype(np.float32),
        "dead_frac":          np.float32(dead),
        "selective_frac":     np.float32(selective),
        "moderate_frac":      np.float32(moderate),
        "dense_frac":         np.float32(dense),
        "activation_rate":    act.astype(np.float32),
    }
    recon_stats = {
        "mse": mse.astype(np.float32),
        "mae": mae.astype(np.float32),
    }
    return latent_stats, recon_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _is_cifar(cfg: dict) -> bool:
    """Infer dataset type from the config (image_size==32 → CIFAR-10)."""
    return cfg.get("data", {}).get("image_size", 256) == 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a trained BaselineConvAutoencoder")
    p.add_argument("--config",      type=str, required=True,
                   help="Path to the experiment JSON config (e.g. configs/baseline_ae_celeba_hq.json)")
    p.add_argument("--checkpoint",  type=str, default="",
                   help="Override checkpoint path (default: output_dir/checkpoints/best.pt)")
    p.add_argument("--save-dir",    type=str, default="",
                   help="Override analysis save dir (default: from config output.analysis_save_dir)")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Limit eval to N batches (0 = all)")
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = json.load(f)

    output_dir = Path(cfg["output"]["output_dir"])
    ckpt_path  = Path(args.checkpoint) if args.checkpoint else output_dir / "checkpoints" / "best.pt"
    save_dir   = Path(args.save_dir)   if args.save_dir   else Path(cfg["output"]["analysis_save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Save dir:   {save_dir}")

    model = load_model(ckpt_path, device)

    dc = cfg.get("data", {})
    if not _is_cifar(cfg):
        image_size = dc.get("image_size", 256)
        tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        loader_obj = CelebAHQLoader(
            data_root=dc["data_root"],
            batch_size=dc.get("batch_size", 32),
            num_workers=dc.get("num_workers", 4),
            image_size=image_size,
            val_split=dc.get("val_split", 0.05),
            seed=42,
            pin_memory=(device.type == "cuda"),
            transform=tf,
        )
        _, eval_loader = loader_obj.get_loaders()
        dataset_name = "celeba"
    else:
        loader_obj = CIFAR10Loader(
            batch_size=dc.get("batch_size", 128),
            root=dc["data_root"],
            num_workers=dc.get("num_workers", 4),
        )
        _, eval_loader = loader_obj.get_loaders()
        dataset_name = "cifar10"

    print(f"Computing metrics on {dataset_name} (device={device}) ...")
    latent_stats, recon_stats = compute_metrics(
        model, eval_loader, device, max_batches=args.max_batches
    )

    np.savez_compressed(save_dir / "latent_statistics.npz",      **latent_stats)
    np.savez_compressed(save_dir / "reconstruction_metrics.npz", **recon_stats)

    print(f"\nSaved to {save_dir}/")
    print(f"  mean_mse={recon_stats['mse'].mean():.6f}  mean_mae={recon_stats['mae'].mean():.6f}")
    print(f"  mean_sparsity={float(latent_stats['mean_sparsity']):.4f}  mean_l0={float(latent_stats['mean_l0']):.1f}")
    print(f"  dead={float(latent_stats['dead_frac']):.3f}  selective={float(latent_stats['selective_frac']):.3f}  dense={float(latent_stats['dense_frac']):.3f}")


if __name__ == "__main__":
    main()
