#!/usr/bin/env python3
"""Train TaxonAutoencoder on CelebA-HQ with DKL=0 ablation.

Identical to train_taxon_ae.py but additionally logs per-taxonomy-node
gradient magnitudes and node usage counts every ``gradient_log_every``
epochs.  This lets us study whether the pairwise softmax collapses to a
single active path when the DKL diversity pressure is removed.

Extra outputs (saved under ``<output_dir>/ablation_logs/``):
  gradient_stats_epoch{E}.npz   — per-node mean |grad| at each depth/stage
  node_usage_epoch{E}.npz       — per-node usage counts (winner-take-all)
  ablation_history.json         — time-series summary of gradient sparsity
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import sys

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.taxon.taxon_ae import TaxonAutoencoder
from src.utils.dataloader import CelebAHQLoader


# ---------------------------------------------------------------------------
# Utilities (same as train_taxon_ae.py)
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
) -> LambdaLR:
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    hard: bool,
    dkl_weight: float,
    entropy_weight: float,
) -> dict:
    model.eval()
    total_loss = total_recon = total_dkl = total_entropy = 0.0
    num_batches = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recon, dkl, entropy = model(images, hard=hard)
        recon_loss = F.mse_loss(recon, images)
        loss = recon_loss + dkl_weight * dkl + entropy_weight * entropy
        total_loss += float(loss.item())
        total_recon += float(recon_loss.item())
        total_dkl += float(dkl.item())
        total_entropy += float(entropy.item())
        num_batches += 1
    if num_batches == 0:
        return {"loss": 0.0, "recon": 0.0, "dkl": 0.0, "entropy": 0.0}
    return {
        "loss": total_loss / num_batches,
        "recon": total_recon / num_batches,
        "dkl": total_dkl / num_batches,
        "entropy": total_entropy / num_batches,
    }


def save_recon_preview(
    model: nn.Module, loader: DataLoader, device: torch.device,
    save_path: Path, hard: bool, num_images: int = 8,
) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:num_images].to(device)
    with torch.no_grad():
        recon, _, _ = model(images, hard=hard)
    vis_input = (images.clamp(-1, 1) + 1.0) * 0.5
    vis_recon = (recon.clamp(-1, 1) + 1.0) * 0.5
    grid = make_grid(torch.cat([vis_input, vis_recon], dim=0), nrow=num_images)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, save_path)


def save_training_curves(history: dict, output_dir: Path) -> None:
    epochs = history["epochs"]
    if not epochs:
        return
    with open(output_dir / "training_history.json", "w") as _f:
        json.dump(history, _f, indent=2)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    panels = [
        ("Total loss", "train_loss", "val_loss"),
        ("Recon loss", "train_recon", "val_recon"),
        ("DKL penalty", "train_dkl", "val_dkl"),
        ("Entropy penalty", "train_entropy", "val_entropy"),
    ]
    for ax, (title, train_key, val_key) in zip(axes, panels):
        ax.plot(epochs, history[train_key], label="train", linewidth=1.5)
        ax.plot(epochs, history[val_key], label="val", linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        if train_key == "train_loss":
            best_ep = epochs[int(min(range(len(history[val_key])),
                                    key=lambda i: history[val_key][i]))]
            ax.axvline(best_ep, color="red", linestyle=":", linewidth=1.0,
                       label=f"best val (ep {best_ep})")
            ax.legend(fontsize=8)
    plt.suptitle("Training curves (DKL=0 ablation)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Ablation-specific: per-taxonomy-node gradient measurement
# ---------------------------------------------------------------------------

def measure_per_node_gradients(
    model: TaxonAutoencoder,
    loader: DataLoader,
    device: torch.device,
    hard: bool,
    n_batches: int = 10,
) -> Dict[str, np.ndarray]:
    """Run a few forward+backward passes and accumulate |grad| per taxonomy node.

    Measures gradients on the actual taxonomy conv layer weights rather than on
    logp tensors.  Specifically, for each encoder stage we inspect the last conv
    layer of the last residual block (``stage.blocks[-1].main[4]``), whose output
    channels map 1-to-1 to taxonomy nodes.  After the backward pass, the per-
    output-channel gradient magnitude ``weight.grad[c].abs().mean()`` tells us
    how much each node's parameters are being updated — sparse gradients here
    confirm that softmax routing makes whole subtrees invisible to the optimiser.

    Returns dict mapping ``"stage{s}_depth{d}"`` → array of shape ``[C_d]``
    containing the mean |grad| for each channel (tree node).
    """
    model.train()
    grad_accum: Dict[str, np.ndarray] = {}
    count = 0

    for b_idx, (images, _) in enumerate(loader):
        if b_idx >= n_batches:
            break
        images = images.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)

        recon, _, _, _ = model(images, hard=hard, return_details=True)
        recon_loss = F.mse_loss(recon, images)
        recon_loss.backward()  # only recon — no DKL in loss by design

        # After backward, read per-output-channel gradient magnitude from the
        # final conv of each stage's last residual block.  Weight shape is
        # [total_out_channels, in_channels, kH, kW]; average over everything
        # except the output-channel dim to get one scalar per taxonomy node.
        for s_idx, stage in enumerate(model.encoder.taxon_stages):
            last_conv = stage.blocks[-1].main[4]  # Conv2d, output dim = total_out_channels
            if last_conv.weight.grad is None:
                continue
            # per_channel shape: [total_out_channels]
            per_channel = last_conv.weight.grad.abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            for d_idx, n_ch in enumerate(stage.layer_channels):
                key = f"stage{s_idx}_depth{d_idx}"
                start = sum(stage.layer_channels[:d_idx])
                node_grads = per_channel[start: start + n_ch]
                if key not in grad_accum:
                    grad_accum[key] = np.zeros_like(node_grads)
                grad_accum[key] += node_grads

        count += 1

    for key in grad_accum:
        grad_accum[key] /= max(1, count)

    return grad_accum


@torch.no_grad()
def measure_node_usage(
    model: TaxonAutoencoder,
    loader: DataLoader,
    device: torch.device,
    hard: bool,
    n_batches: int = 50,
) -> Dict[str, np.ndarray]:
    """Count how often each taxonomy node is the winner (argmax of softmax pair).

    Uses ``return_details=True`` to get per-stage ``logp`` tensors, then
    splits by depth and counts argmax winners at each spatial location.

    Returns dict mapping ``"stage{s}_depth{d}"`` → array of shape ``[C_d]``.
    """
    model.eval()
    usage_counts: Dict[str, np.ndarray] = {}

    for b_idx, (images, _) in enumerate(loader):
        if b_idx >= n_batches:
            break
        images = images.to(device, non_blocking=True)

        _, _, _, details = model(images, hard=False, return_details=True)

        for s_idx, stage_info in enumerate(details["encoder"]["stages"]):
            logp = stage_info["logp"]
            layer_channels = stage_info["layer_channels"]
            logp_splits = torch.split(logp, layer_channels, dim=1)

            for d_idx, logp_d in enumerate(logp_splits):
                key = f"stage{s_idx}_depth{d_idx}"
                prob_d = logp_d.exp()
                winners = prob_d.argmax(dim=1).flatten()  # [B*H*W]
                n_channels = prob_d.shape[1]

                if key not in usage_counts:
                    usage_counts[key] = np.zeros(n_channels, dtype=np.int64)

                counts = torch.bincount(winners, minlength=n_channels)
                usage_counts[key] += counts.cpu().numpy()

    return usage_counts


def compute_gradient_sparsity_stats(
    grad_dict: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Summarize gradient distribution across all nodes.

    Returns:
        gini: Gini coefficient of gradient magnitudes (1 = all grad to 1 node).
        top1_frac: Fraction of total gradient going to the single largest node.
        zero_frac: Fraction of nodes receiving < 1% of the mean gradient.
    """
    if not grad_dict:
        return {"gini": 0.0, "top1_frac": 0.0, "zero_frac": 0.0}
    all_grads = np.concatenate([v for v in grad_dict.values()])
    total = all_grads.sum()
    if total < 1e-12:
        return {"gini": 1.0, "top1_frac": 1.0, "zero_frac": 1.0}

    # Gini coefficient
    sorted_g = np.sort(all_grads)
    n = len(sorted_g)
    idx = np.arange(1, n + 1)
    gini = float((2 * (idx * sorted_g).sum() / (n * sorted_g.sum())) - (n + 1) / n)

    # Top-1 fraction
    top1_frac = float(all_grads.max() / total)

    # Near-zero fraction
    threshold = all_grads.mean() * 0.01
    zero_frac = float((all_grads < threshold).mean())

    return {"gini": gini, "top1_frac": top1_frac, "zero_frac": zero_frac}


def compute_usage_stats(
    usage_dict: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Summarize node usage across all stages/depths.

    Returns:
        gini: Gini of usage counts (1 = only one node ever used).
        dead_frac: Fraction of nodes that are never selected as winner.
        top1_frac: Fraction of total routing going to the most-used node.
    """
    all_counts = np.concatenate([v for v in usage_dict.values()]).astype(np.float64)
    total = all_counts.sum()
    if total < 1:
        return {"gini": 1.0, "dead_frac": 1.0, "top1_frac": 1.0}

    sorted_c = np.sort(all_counts)
    n = len(sorted_c)
    idx = np.arange(1, n + 1)
    gini = float((2 * (idx * sorted_c).sum() / (n * sorted_c.sum())) - (n + 1) / n)
    dead_frac = float((all_counts == 0).mean())
    top1_frac = float(all_counts.max() / total)

    return {"gini": gini, "dead_frac": dead_frac, "top1_frac": top1_frac}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args()
    cfg: dict = {}
    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = json.load(f)
    d = cfg.get("data", {})
    m = cfg.get("model", {})
    t = cfg.get("training", {})
    o = cfg.get("output", {})

    parser = argparse.ArgumentParser(description="Train Taxon AE (DKL=0 ablation) on CelebA-HQ")
    parser.add_argument("--config", type=str, default="")
    # data
    parser.add_argument("--data-root", type=str, default=d.get("data_root", "./data/celeba_hq"))
    parser.add_argument("--output-dir", type=str, default=o.get("output_dir", "./outputs/celeba_hq/taxon_ae_celeba_hq_r18_nodkl_ablation"))
    parser.add_argument("--image-size", type=int, default=d.get("image_size", 256))
    parser.add_argument("--batch-size", type=int, default=d.get("batch_size", 32))
    parser.add_argument("--num-workers", type=int, default=d.get("num_workers", 8))
    parser.add_argument("--val-split", type=float, default=d.get("val_split", 0.05))
    # model
    parser.add_argument("--resnet-variant", type=str, default=m.get("resnet_variant", "18"))
    parser.add_argument("--stage-taxonomy-layers", type=int, nargs=4,
                        default=m.get("stage_taxonomy_layers", [5, 6, 7, 8]))
    parser.add_argument("--stage-strides", type=int, nargs=4,
                        default=m.get("stage_strides", [1, 2, 2, 2]))
    parser.add_argument("--temperature", type=float, default=m.get("temperature", 0.5))
    parser.add_argument("--hard", action="store_true", default=m.get("hard", False))
    # training
    parser.add_argument("--epochs", type=int, default=t.get("epochs", 90))
    parser.add_argument("--learning-rate", type=float, default=t.get("learning_rate", 3e-4))
    parser.add_argument("--weight-decay", type=float, default=t.get("weight_decay", 1e-4))
    parser.add_argument("--warmup-epochs", type=int, default=t.get("warmup_epochs", 3))
    parser.add_argument("--dkl-weight", type=float, default=t.get("dkl_weight", 0.0))
    parser.add_argument("--entropy-weight", type=float, default=t.get("entropy_weight", 0.0))
    parser.add_argument("--save-every", type=int, default=t.get("save_every", 5))
    parser.add_argument("--seed", type=int, default=t.get("seed", 42))
    parser.add_argument("--max-train-steps", type=int, default=t.get("max_train_steps", 0))
    parser.add_argument("--resume", type=str, default="")
    # ablation-specific
    parser.add_argument("--gradient-log-every", type=int,
                        default=t.get("gradient_log_every", 5),
                        help="Log gradient & usage stats every N epochs")
    parser.add_argument("--node-usage-batches", type=int,
                        default=t.get("node_usage_batches", 50),
                        help="Number of val batches for node usage measurement")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    ablation_dir = output_dir / "ablation_logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    celeba_loader = CelebAHQLoader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_split=args.val_split,
        seed=args.seed,
        pin_memory=(device.type == "cuda"),
        transform=tf,
    )
    train_loader, val_loader = celeba_loader.get_loaders()
    if val_loader is None:
        raise RuntimeError("val_split must be > 0")

    _mc = {}
    if args.config:
        with open(args.config) as _f:
            _mc = json.load(_f).get("model", {})

    model = TaxonAutoencoder(
        in_channels=_mc.get("in_channels", 3),
        resnet_variant=args.resnet_variant,
        stage_taxonomy_layers=tuple(args.stage_taxonomy_layers),
        stage_strides=tuple(args.stage_strides),
        stage_blocks=_mc.get("stage_blocks", None),
        temperature=args.temperature,
        hard=args.hard,
        kernel_size=_mc.get("kernel_size", 3),
        use_stem=_mc.get("use_stem", True),
        stem_channels=_mc.get("stem_channels", 64),
        stem_stride=_mc.get("stem_stride", 2),
        use_stem_maxpool=_mc.get("use_stem_maxpool", True),
        output_activation=_mc.get("output_activation", "none"),
        depth_decay=_mc.get("depth_decay", 0.5),
    ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )
    start_epoch = 1
    global_step = 0
    best_val = float("inf")

    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_val = float(state.get("best_val", float("inf")))
        print(f"Resumed from {args.resume} at epoch={start_epoch}")

    print(
        "Training setup (DKL=0 Ablation):\n"
        f"  device={device}\n"
        f"  dkl_weight={args.dkl_weight}  entropy_weight={args.entropy_weight}\n"
        f"  gradient_log_every={args.gradient_log_every}\n"
        f"  train_size={len(celeba_loader.trainset)}  val_size={len(celeba_loader.valset)}\n"
        f"  batch_size={args.batch_size}  epochs={args.epochs}\n"
        f"  stage_taxonomy_layers={tuple(args.stage_taxonomy_layers)}"
    )

    history: dict = {
        "epochs": [], "train_loss": [], "train_recon": [],
        "train_dkl": [], "train_entropy": [],
        "val_loss": [], "val_recon": [], "val_dkl": [], "val_entropy": [],
    }

    ablation_history: dict = {
        "epochs": [],
        "grad_gini": [], "grad_top1_frac": [], "grad_zero_frac": [],
        "usage_gini": [], "usage_dead_frac": [], "usage_top1_frac": [],
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = running_recon = running_dkl = running_entropy = 0.0
        num_batches = 0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            recon, dkl, entropy = model(images, hard=args.hard)
            recon_loss = F.mse_loss(recon, images)
            loss = recon_loss + args.dkl_weight * dkl + args.entropy_weight * entropy

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            running_recon += float(recon_loss.item())
            running_dkl += float(dkl.item())
            running_entropy += float(entropy.item())
            num_batches += 1
            global_step += 1

            if batch_idx % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} step={global_step} "
                    f"lr={lr:.3e} loss={running_loss/num_batches:.5f} "
                    f"recon={running_recon/num_batches:.5f} "
                    f"dkl={running_dkl/num_batches:.5f}"
                )

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        train_stats = {
            "loss": running_loss / max(1, num_batches),
            "recon": running_recon / max(1, num_batches),
            "dkl": running_dkl / max(1, num_batches),
            "entropy": running_entropy / max(1, num_batches),
        }
        val_stats = run_validation(model, val_loader, device, args.hard,
                                   args.dkl_weight, args.entropy_weight)

        elapsed = time.time() - epoch_start
        print(
            f"epoch={epoch:03d} time={elapsed:.1f}s "
            f"train_loss={train_stats['loss']:.5f} train_recon={train_stats['recon']:.5f} "
            f"val_loss={val_stats['loss']:.5f} val_recon={val_stats['recon']:.5f}"
        )

        history["epochs"].append(epoch)
        for k in ("loss", "recon", "dkl", "entropy"):
            history[f"train_{k}"].append(train_stats[k])
            history[f"val_{k}"].append(val_stats[k])

        # ── Ablation logging ──────────────────────────────────────────────
        if epoch % args.gradient_log_every == 0 or epoch == 1:
            print(f"  [Ablation] Measuring per-node gradients & usage (epoch {epoch})...")

            grad_dict = measure_per_node_gradients(
                model, val_loader, device, hard=args.hard, n_batches=10,
            )
            usage_dict = measure_node_usage(
                model, val_loader, device, hard=args.hard,
                n_batches=args.node_usage_batches,
            )

            np.savez(ablation_dir / f"gradient_stats_epoch{epoch:03d}.npz",
                     **{k: v for k, v in grad_dict.items()})
            np.savez(ablation_dir / f"node_usage_epoch{epoch:03d}.npz",
                     **{k: v for k, v in usage_dict.items()})

            grad_stats = compute_gradient_sparsity_stats(grad_dict)
            usage_stats = compute_usage_stats(usage_dict)

            ablation_history["epochs"].append(epoch)
            for k, v in grad_stats.items():
                ablation_history[f"grad_{k}"].append(v)
            for k, v in usage_stats.items():
                ablation_history[f"usage_{k}"].append(v)

            print(
                f"  [Ablation] grad_gini={grad_stats['gini']:.3f} "
                f"grad_top1={grad_stats['top1_frac']:.3f} "
                f"grad_zero={grad_stats['zero_frac']:.3f} | "
                f"usage_gini={usage_stats['gini']:.3f} "
                f"usage_dead={usage_stats['dead_frac']:.3f} "
                f"usage_top1={usage_stats['top1_frac']:.3f}"
            )

            with open(ablation_dir / "ablation_history.json", "w") as _f:
                json.dump(ablation_history, _f, indent=2)

        # ── Checkpointing ────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            state = {
                "epoch": epoch, "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val, "args": vars(args),
                "train_stats": train_stats, "val_stats": val_stats,
            }
            torch.save(state, ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt")
            torch.save(state, ckpt_dir / "latest.pt")

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val, "args": vars(args),
                "train_stats": train_stats, "val_stats": val_stats,
            }, ckpt_dir / "best.pt")

        save_recon_preview(model, val_loader, device,
                           preview_dir / f"epoch_{epoch:03d}.png", args.hard)

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            print(f"Reached max_train_steps={args.max_train_steps}; stopping early.")
            break

    save_training_curves(history, output_dir)
    print(f"\nTraining complete. Outputs at {output_dir}")
    print(f"Ablation logs at {ablation_dir}")


if __name__ == "__main__":
    main()
