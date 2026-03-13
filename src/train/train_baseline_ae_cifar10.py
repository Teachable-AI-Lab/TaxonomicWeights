#!/usr/bin/env python3
"""Train BaselineConvAutoencoder on CIFAR-10 (ResNet-18 stage layout).

Mirrors src/train/train_baseline_ae_celeba_hq.py but uses CIFAR10Loader
(32×32, stem_stride=1, no maxpool).

Loss:
    total = MSE(recon, x)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid, save_image

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.baseline_ae import BaselineConvAutoencoder
from src.utils.dataloader import CIFAR10Loader


# ---------------------------------------------------------------------------
# Utilities
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
    total_steps  = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def run_validation(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_recon = 0.0
    num_batches = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recon, _ = model(images)
        total_recon += float(F.mse_loss(recon, images).item())
        num_batches += 1
    if num_batches == 0:
        return {"loss": 0.0, "recon": 0.0, "sparsity": 0.0}
    v = total_recon / num_batches
    return {"loss": v, "recon": v, "sparsity": 0.0}


def save_recon_preview(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    num_images: int = 8,
) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:num_images].to(device)
    with torch.no_grad():
        recon, _ = model(images)
    vis_input = (images.clamp(-1, 1) + 1.0) * 0.5
    vis_recon = (recon.clamp(-1, 1) + 1.0) * 0.5
    grid = make_grid(torch.cat([vis_input, vis_recon], dim=0), nrow=num_images)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, save_path)


def save_training_curves(history: dict, output_dir: Path) -> None:
    epochs = history["epochs"]
    if not epochs:
        return
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, tk, vk) in zip(axes, [
        ("Total loss",  "train_loss",  "val_loss"),
        ("Recon loss",  "train_recon", "val_recon"),
    ]):
        ax.plot(epochs, history[tk], label="train", linewidth=1.5)
        ax.plot(epochs, history[vk], label="val",   linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=11); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle("Baseline AE CIFAR-10 training curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {out_path}")


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

    d = cfg.get("data", {}); m = cfg.get("model", {}); t = cfg.get("training", {}); o = cfg.get("output", {})

    parser = argparse.ArgumentParser(description="Train BaselineConvAutoencoder on CIFAR-10")
    parser.add_argument("--config",         type=str,   default="")
    parser.add_argument("--data-root",      type=str,   default=d.get("data_root",   "./data"))
    parser.add_argument("--output-dir",     type=str,   default=o.get("output_dir",  "./outputs/baseline_ae_cifar10"))
    parser.add_argument("--batch-size",     type=int,   default=d.get("batch_size",  128))
    parser.add_argument("--num-workers",    type=int,   default=d.get("num_workers", 4))
    parser.add_argument("--resnet-variant", type=str,   default=m.get("resnet_variant", "18"))
    parser.add_argument("--stage-channels", type=int,   nargs=4,
                        default=m.get("stage_channels", [64, 128, 256, 512]))
    parser.add_argument("--stage-strides",  type=int,   nargs=4,
                        default=m.get("stage_strides",  [1, 2, 2, 2]))
    parser.add_argument("--stem-stride",    type=int,   default=m.get("stem_stride", 1))
    parser.add_argument(
        "--use-stem-maxpool",
        action=argparse.BooleanOptionalAction,
        default=m.get("use_stem_maxpool", False),
    )
    parser.add_argument("--epochs",         type=int,   default=t.get("epochs",       90))
    parser.add_argument("--learning-rate",  type=float, default=t.get("learning_rate",3e-4))
    parser.add_argument("--weight-decay",   type=float, default=t.get("weight_decay", 1e-4))
    parser.add_argument("--warmup-epochs",  type=int,   default=t.get("warmup_epochs",3))
    parser.add_argument("--save-every",     type=int,   default=t.get("save_every",   5))
    parser.add_argument("--seed",           type=int,   default=t.get("seed",         42))
    parser.add_argument("--max-train-steps",type=int,   default=t.get("max_train_steps", 0))
    parser.add_argument("--val-split",      type=float, default=0.1)
    parser.add_argument("--resume",         type=str,   default="")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir  = Path(args.output_dir)
    ckpt_dir    = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cifar = CIFAR10Loader(batch_size=args.batch_size, root=args.data_root)
    full_train_loader, test_loader = cifar.get_loaders()

    # Carve a small validation split from the training set.
    full_train = cifar.trainset
    val_size   = int(len(full_train) * args.val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    _mc: dict = {}
    if args.config:
        with open(args.config) as f:
            _mc = json.load(f).get("model", {})

    model = BaselineConvAutoencoder(
        in_channels=3,
        resnet_variant=args.resnet_variant,
        stage_channels=tuple(args.stage_channels),
        stage_strides=tuple(args.stage_strides),
        stage_blocks=_mc.get("stage_blocks", None),
        kernel_size=_mc.get("kernel_size", 3),
        use_stem=_mc.get("use_stem", True),
        stem_channels=_mc.get("stem_channels", 64),
        stem_stride=args.stem_stride,
        use_stem_maxpool=args.use_stem_maxpool,
        output_activation=_mc.get("output_activation", "none"),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        steps_per_epoch=max(1, len(train_loader)),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    start_epoch = 1; global_step = 0; best_val = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_val    = float(state.get("best_val", float("inf")))
        print(f"Resumed from {args.resume} at epoch={start_epoch}")

    print(
        "Training setup (Baseline AE CIFAR-10):\n"
        f"  device={device}\n"
        f"  train_size={train_size}  val_size={val_size}\n"
        f"  batch_size={args.batch_size}  epochs={args.epochs}\n"
        f"  lr={args.learning_rate}  wd={args.weight_decay}\n"
        f"  stage_channels={tuple(args.stage_channels)}"
    )

    history: dict = {
        "epochs":         [],
        "train_loss":     [],
        "train_recon":    [],
        "train_sparsity": [],
        "val_loss":       [],
        "val_recon":      [],
        "val_sparsity":   [],
    }

    _ckpt_args = {
        "in_channels":       3,
        "resnet_variant":    args.resnet_variant,
        "stage_channels":    list(args.stage_channels),
        "stage_strides":     list(args.stage_strides),
        "stage_blocks":      _mc.get("stage_blocks", None),
        "kernel_size":       _mc.get("kernel_size", 3),
        "use_stem":          _mc.get("use_stem", True),
        "stem_channels":     _mc.get("stem_channels", 64),
        "stem_stride":       args.stem_stride,
        "use_stem_maxpool":  args.use_stem_maxpool,
        "output_activation": _mc.get("output_activation", "none"),
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start  = time.time()
        running_recon = 0.0
        num_batches   = 0

        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon, _ = model(images)
            loss = F.mse_loss(recon, images)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_recon += float(loss.item())
            num_batches   += 1
            global_step   += 1
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        train_stats = {"loss": running_recon / max(1, num_batches),
                       "recon": running_recon / max(1, num_batches), "sparsity": 0.0}
        val_stats   = run_validation(model, val_loader, device)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_recon={train_stats['recon']:.5f} | "
            f"val_recon={val_stats['recon']:.5f} | "
            f"{elapsed:.1f}s"
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["train_recon"].append(train_stats["recon"])
        history["train_sparsity"].append(0.0)
        history["val_loss"].append(val_stats["loss"])
        history["val_recon"].append(val_stats["recon"])
        history["val_sparsity"].append(0.0)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"epoch": epoch, "global_step": global_step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val": best_val, "args": _ckpt_args,
                        "train_stats": train_stats, "val_stats": val_stats},
                       ckpt_dir / f"epoch_{epoch:04d}.pt")

        if val_stats["recon"] < best_val:
            best_val = val_stats["recon"]
            torch.save({"epoch": epoch, "global_step": global_step,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_val": best_val, "args": _ckpt_args,
                        "train_stats": train_stats, "val_stats": val_stats},
                       ckpt_dir / "best.pt")
            print(f"  -> New best val_recon={best_val:.6f}  (saved best.pt)")

        if epoch % args.save_every == 0:
            save_recon_preview(model, val_loader, device,
                               preview_dir / f"epoch_{epoch:04d}.png")
            save_training_curves(history, output_dir)

    save_training_curves(history, output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
