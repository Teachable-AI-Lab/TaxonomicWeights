#!/usr/bin/env python3
"""Per-model paper analysis for ImageNet.

Computes all evaluation metrics for ONE model checkpoint trained on ImageNet
and saves results to ``outputs/paper/imagenet/<run_name>/``.

Metrics
-------
1. Reconstruction  : MSE, MAE, PSNR  (on val set)
2. Sparsity        : mean L0, dead-neuron fraction (threshold 1e-6)
3. Linear probing  : logistic regression on ImageNet 1000-class labels
                     (uses a balanced 50-class × 200-image subset for speed)
4. KNN             : 1-NN and 5-NN accuracy (same 50-class subset)
5. Targeted editing: zero the top-K active channels and decode

Saved files (all under ``<save_dir>/<run_name>/``)
---------------------------------------------------
metrics.json / metrics_summary.csv
latents.npy, images_orig.npy, labels.npy (class ids)
reconstruction_samples.png
editing_samples.png

Usage
-----
python src/paper/paper_analysis_imagenet.py \\
    --config configs/imagenet/taxon_ae_imagenet.json \\
    --device cuda
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
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-use model-loading helpers from the celeba paper analysis
from src.paper.paper_analysis_celeba import (
    _training_output_dir_suffix,
    _resolve_checkpoint,
    _run_name,
    _infer_model_type,
    load_model,
    _to_display,
    _encode,
    _encode_spatial,
    _decode,
)
from src.utils.dataloader import ImageNet1kHFLoader


# ---------------------------------------------------------------------------
# Suffix for ImageNet training scripts
# (same patterns as CelebA-HQ — they use identical training script logic)
# ---------------------------------------------------------------------------
# _training_output_dir_suffix is imported from paper_analysis_celeba.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_latents_and_recon_imagenet(
    model,
    loader,
    device: torch.device,
    dead_threshold: float = 1e-6,
    max_images: int = 5000,
) -> dict:
    """Run the val loader; return latents, images, labels, and recon metrics."""
    all_latents: List[np.ndarray] = []
    all_imgs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_mse: List[float] = []
    all_mae: List[float] = []
    n_seen = 0

    for imgs, labels in tqdm(loader, desc="  Collecting latents"):
        if max_images > 0 and n_seen >= max_images:
            break
        imgs = imgs.to(device)
        out = model(imgs)
        recon = out[0]

        z_flat = _encode(model, imgs)
        all_latents.append(z_flat.numpy())
        all_imgs.append(imgs.cpu().numpy())
        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
        mse_batch = ((imgs - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        mae_batch = torch.abs(imgs - recon).mean(dim=(1, 2, 3)).cpu().numpy()
        all_mse.extend(mse_batch.tolist())
        all_mae.extend(mae_batch.tolist())
        n_seen += imgs.shape[0]

    Z = np.concatenate(all_latents, axis=0)
    imgs_np = np.concatenate(all_imgs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    mse = np.array(all_mse)
    mae = np.array(all_mae)

    l0_per_sample = (np.abs(Z) > dead_threshold).sum(axis=1).astype(float)
    max_act = np.abs(Z).max(axis=0)
    dead_frac = float((max_act < dead_threshold).mean())

    return {
        "Z": Z,
        "imgs": imgs_np,
        "labels": labels_np,
        "mse": mse,
        "mae": mae,
        "l0_per_sample": l0_per_sample,
        "dead_frac": dead_frac,
    }


def run_linear_probe_multiclass(Z_train, Y_train, Z_val, Y_val) -> dict:
    clf = LogisticRegression(
        max_iter=300, solver="saga", C=0.1, n_jobs=4, random_state=0,
        multi_class="multinomial",
    )
    clf.fit(Z_train, Y_train)
    pred = clf.predict(Z_val)
    acc = float(accuracy_score(Y_val, pred))
    return {"probe_top1_acc": acc}


def run_knn_multiclass(Z_train, Y_train, Z_val, Y_val, ks=(1, 5)) -> dict:
    results = {}
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=4, metric="cosine")
        knn.fit(Z_train, Y_train)
        pred = knn.predict(Z_val)
        results[f"knn_k{k}_top1_acc"] = float(accuracy_score(Y_val, pred))
    return results


def _balanced_class_subset(Z, Y, n_classes=50, per_class=200, seed=42):
    """Return balanced (Z_sub, Y_sub, Y_remapped) for ``n_classes`` × ``per_class``."""
    rng = np.random.RandomState(seed)
    unique_cls = np.unique(Y)
    if len(unique_cls) > n_classes:
        unique_cls = rng.choice(unique_cls, size=n_classes, replace=False)
    else:
        n_classes = len(unique_cls)

    class_map = {c: i for i, c in enumerate(unique_cls)}
    idx_list = []
    for c in unique_cls:
        inds = np.where(Y == c)[0]
        chosen = rng.choice(inds, size=min(per_class, len(inds)), replace=False)
        idx_list.append(chosen)
    idx = np.concatenate(idx_list)
    rng.shuffle(idx)
    Y_remapped = np.array([class_map[Y[i]] for i in idx])
    return Z[idx], Y[idx], Y_remapped


@torch.no_grad()
def run_targeted_editing_imagenet(model, val_loader, device, n_images=4,
                                  topk_channels=32):
    imgs_list, orig_recons, edited_recons = [], [], []
    for imgs, _ in val_loader:
        imgs = imgs.to(device)
        need = n_images - len(imgs_list)
        if need <= 0:
            break
        imgs = imgs[:need]

        out = model(imgs)
        recon_orig = out[0].clamp(-1, 1)

        z_spatial = _encode_spatial(model, imgs)
        activity = z_spatial.abs().mean(dim=(2, 3))
        topk_idx = activity.topk(topk_channels, dim=1).indices
        z_edit = z_spatial.clone()
        for b in range(z_edit.shape[0]):
            z_edit[b, topk_idx[b]] = 0.0

        recon_edit = model.decode(z_edit).clamp(-1, 1).cpu()
        imgs_list.append(imgs.cpu())
        orig_recons.append(recon_orig.cpu())
        edited_recons.append(recon_edit)
        if sum(x.shape[0] for x in imgs_list) >= n_images:
            break
    return imgs_list, orig_recons, edited_recons


def _to_display_imagenet(t: torch.Tensor) -> np.ndarray:
    """ImageNet images are normalized to [-1,1]."""
    img = t.detach().cpu().squeeze(0).permute(1, 2, 0)
    img = img * 0.5 + 0.5
    return img.clamp(0, 1).numpy()


def save_recon_samples_imagenet(loader, model, device, save_path, n_images=8):
    img_batch, _ = next(iter(loader))
    img_batch = img_batch[:n_images].to(device)
    with torch.no_grad():
        recon_batch = model(img_batch)[0].clamp(-1, 1).cpu()
    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 4))
    for i in range(n_images):
        axes[0, i].imshow(_to_display_imagenet(img_batch[i:i+1].cpu()))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("orig", fontsize=8)
        axes[1, i].imshow(_to_display_imagenet(recon_batch[i:i+1]))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("recon", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_editing_plot_imagenet(imgs_list, orig_recons, edited_recons, save_path, topk):
    imgs_cat = torch.cat(imgs_list)
    origs_cat = torch.cat(orig_recons)
    edits_cat = torch.cat(edited_recons)
    n = min(len(imgs_cat), 4)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    row_labels = ["Original", "Reconstruction", f"Edited (zero top-{topk} ch)"]
    for row, batch in enumerate([imgs_cat, origs_cat, edits_cat]):
        for col in range(n):
            axes[row, col].imshow(_to_display_imagenet(batch[col:col+1]))
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(row_labels[row], fontsize=9)
    plt.suptitle(f"Targeted editing: zero top-{topk} active channels", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper analysis for one ImageNet model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n-edit", type=int, default=4)
    parser.add_argument("--topk-edit", type=int, default=32)
    parser.add_argument("--n-recon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-val-images", type=int, default=5000,
                        help="Maximum val images for latent collection")
    parser.add_argument("--probe-n-classes", type=int, default=50)
    parser.add_argument("--probe-per-class", type=int, default=200)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(cfg_path) as f:
        cfg = json.load(f)

    dc = cfg.get("data", {})
    image_size = dc.get("image_size", 224)
    batch_size = args.batch_size or dc.get("batch_size", 32)
    num_workers = args.num_workers
    max_train = dc.get("max_train_samples", 100000)
    max_val = dc.get("max_val_samples", 10000)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_name = _run_name(cfg)
    save_dir = Path(args.save_dir or (ROOT / "outputs" / "paper" / "imagenet")) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run : {run_name}")
    print(f"Save: {save_dir}")

    ckpt_path = _resolve_checkpoint(cfg, args.checkpoint)
    print(f"Ckpt: {ckpt_path}")
    model_type = _infer_model_type(cfg)
    print(f"Type: {model_type}")
    model = load_model(ckpt_path, model_type, device)

    # ── data ────────────────────────────────────────────────────────────────
    print("\nLoading ImageNet...")
    loader_obj = ImageNet1kHFLoader(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        max_train_samples=max_train,
        max_val_samples=max_val,
    )
    _, val_loader = loader_obj.get_loaders()

    # ── 1. Latents + reconstruction metrics ─────────────────────────────────
    print("\n[1] Collecting latents and reconstruction metrics...")
    result = collect_latents_and_recon_imagenet(
        model, val_loader, device, max_images=args.max_val_images,
    )
    Z = result["Z"]
    imgs_np = result["imgs"]
    labels_np = result["labels"]
    mse = result["mse"]
    mae = result["mae"]
    l0 = result["l0_per_sample"]
    dead_frac = result["dead_frac"]

    # ImageNet images are in [-1,1]; scale MSE to [0,1] space
    psnr = -10 * np.log10(np.maximum(mse / 4.0, 1e-10))  # divide by 4 for [-1,1] range

    metrics: dict = {
        "mse_mean": float(mse.mean()),
        "mse_std": float(mse.std()),
        "mae_mean": float(mae.mean()),
        "mae_std": float(mae.std()),
        "psnr_mean": float(psnr.mean()),
        "psnr_std": float(psnr.std()),
        "l0_mean": float(l0.mean()),
        "l0_std": float(l0.std()),
        "dead_neuron_frac": dead_frac,
        "latent_dim": int(Z.shape[1]),
        "n_val_images": int(Z.shape[0]),
    }
    print(f"  MSE={metrics['mse_mean']:.5f}  MAE={metrics['mae_mean']:.5f}  "
          f"PSNR={metrics['psnr_mean']:.2f}dB")
    print(f"  L0={metrics['l0_mean']:.1f}  dead={dead_frac*100:.1f}%")

    np.save(save_dir / "latents.npy", Z)
    np.save(save_dir / "images_orig.npy", imgs_np[:min(200, len(imgs_np))])
    np.save(save_dir / "labels.npy", labels_np)

    # ── 2. Reconstruction samples ────────────────────────────────────────────
    print("\n[2] Saving reconstruction samples...")
    save_recon_samples_imagenet(val_loader, model, device,
                                save_dir / "reconstruction_samples.png",
                                n_images=args.n_recon)

    # ── 3. Probing (balanced subset) ─────────────────────────────────────────
    print(f"\n[3] Linear probing & KNN ({args.probe_n_classes} classes)...")
    n_probe = args.probe_n_classes * args.probe_per_class
    if len(Z) >= n_probe * 2:
        Z_sub, _, Y_sub = _balanced_class_subset(
            Z, labels_np, n_classes=args.probe_n_classes,
            per_class=args.probe_per_class, seed=42,
        )
        split = len(Z_sub) // 2
        Z_tr, Y_tr = Z_sub[:split], Y_sub[:split]
        Z_va, Y_va = Z_sub[split:], Y_sub[split:]

        print("  Fitting linear probe...")
        probe_metrics = run_linear_probe_multiclass(Z_tr, Y_tr, Z_va, Y_va)
        metrics.update(probe_metrics)
        print("  Fitting KNN...")
        knn_metrics = run_knn_multiclass(Z_tr, Y_tr, Z_va, Y_va, ks=(1, 5))
        metrics.update(knn_metrics)
        print(f"  probe_top1={metrics['probe_top1_acc']:.3f}  "
              f"knn_k1={metrics.get('knn_k1_top1_acc', float('nan')):.3f}  "
              f"knn_k5={metrics.get('knn_k5_top1_acc', float('nan')):.3f}")
    else:
        print(f"  [skip] Not enough samples (have {len(Z)}, need {n_probe*2}). "
              "Increase --max-val-images.")
        metrics.update({
            "probe_top1_acc": float("nan"),
            "knn_k1_top1_acc": float("nan"),
            "knn_k5_top1_acc": float("nan"),
        })

    # ── 4. Targeted editing ──────────────────────────────────────────────────
    print(f"\n[4] Targeted editing (zero top-{args.topk_edit} channels)...")
    try:
        imgs_list, orig_recons, edited_recons = run_targeted_editing_imagenet(
            model, val_loader, device,
            n_images=args.n_edit, topk_channels=args.topk_edit,
        )
        save_editing_plot_imagenet(imgs_list, orig_recons, edited_recons,
                                   save_dir / "editing_samples.png", topk=args.topk_edit)
    except Exception as e:
        print(f"  [warn] Targeted editing failed: {e}")

    # ── Save metrics ─────────────────────────────────────────────────────────
    print("\n[5] Saving metrics...")
    json_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    json_metrics["run_name"] = run_name
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(json_metrics, f, indent=2)
    with open(save_dir / "metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in json_metrics.items():
            writer.writerow([k, v])

    print(f"\nDone. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
