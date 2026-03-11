#!/usr/bin/env python3
"""Train BaselineConvAutoencoder on CelebA-HQ (ResNet-18 stage layout).

Mirrors tests/train_sae_celeba_hq.py exactly, but uses the baseline model
which has no sparsity penalty.  The ``reg_term`` in the training loop is
always 0 — kept to share the same checkpoint schema as SAE.

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
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.baseline_ae import BaselineConvAutoencoder
from src.utils.dataloader import CelebAHQLoader


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
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = total_recon = 0.0
    num_batches = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recon, _ = model(images)
        recon_loss = F.mse_loss(recon, images)
        total_loss  += float(recon_loss.item())
        total_recon += float(recon_loss.item())
        num_batches += 1
    if num_batches == 0:
        return {"loss": 0.0, "recon": 0.0, "sparsity": 0.0}
    return {
        "loss":     total_loss  / num_batches,
        "recon":    total_recon / num_batches,
        "sparsity": 0.0,
    }


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
        if tk == "train_loss":
            best_ep = epochs[int(min(range(len(history[vk])),
                                     key=lambda i: history[vk][i]))]
            ax.axvline(best_ep, color="red", linestyle=":", linewidth=1.0,
                       label=f"best val (ep {best_ep})")
            ax.legend(fontsize=8)

    plt.suptitle("Baseline AE CelebA-HQ training curves", fontsize=13, fontweight="bold")
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

    parser = argparse.ArgumentParser(description="Train BaselineConvAutoencoder on CelebA-HQ")
    parser.add_argument("--config",        type=str,   default="")
    parser.add_argument("--data-root",     type=str,   default=d.get("data_root",    "./data/celeba_hq"))
    parser.add_argument("--output-dir",    type=str,   default=o.get("output_dir",   "./outputs/baseline_ae_celeba_hq"))
    parser.add_argument("--image-size",    type=int,   default=d.get("image_size",   256))
    parser.add_argument("--batch-size",    type=int,   default=d.get("batch_size",   32))
    parser.add_argument("--num-workers",   type=int,   default=d.get("num_workers",  8))
    parser.add_argument("--val-split",     type=float, default=d.get("val_split",    0.05))
    parser.add_argument("--resnet-variant",type=str,   default=m.get("resnet_variant", "18"))
    parser.add_argument("--stage-channels",type=int,   nargs=4,
                        default=m.get("stage_channels", [64, 128, 256, 512]))
    parser.add_argument("--stage-strides", type=int,   nargs=4,
                        default=m.get("stage_strides",  [1, 2, 2, 2]))
    parser.add_argument("--stem-stride",   type=int,   default=m.get("stem_stride",  2))
    parser.add_argument(
        "--use-stem-maxpool",
        action=argparse.BooleanOptionalAction,
        default=m.get("use_stem_maxpool", True),
    )
    parser.add_argument("--epochs",        type=int,   default=t.get("epochs",       90))
    parser.add_argument("--learning-rate", type=float, default=t.get("learning_rate",3e-4))
    parser.add_argument("--weight-decay",  type=float, default=t.get("weight_decay", 1e-4))
    parser.add_argument("--warmup-epochs", type=int,   default=t.get("warmup_epochs",3))
    parser.add_argument("--save-every",    type=int,   default=t.get("save_every",   5))
    parser.add_argument("--seed",          type=int,   default=t.get("seed",         42))
    parser.add_argument("--max-train-steps",type=int,  default=t.get("max_train_steps", 0))
    parser.add_argument("--resume",        type=str,   default="")
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
        raise RuntimeError("val_split must be > 0 to produce a validation loader")

    _mc: dict = {}
    if args.config:
        with open(args.config) as f:
            _mc = json.load(f).get("model", {})

    model = BaselineConvAutoencoder(
        in_channels=_mc.get("in_channels", 3),
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
    start_epoch = 1
    global_step = 0
    best_val    = float("inf")

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
        "Training setup (Baseline AE CelebA-HQ):\n"
        f"  device={device}\n"
        f"  train_size={len(celeba_loader.trainset)}  val_size={len(celeba_loader.valset)}\n"
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

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start  = time.time()
        running_loss = running_recon = 0.0
        num_batches  = 0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            recon, _ = model(images)
            loss = F.mse_loss(recon, images)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss  += float(loss.item())
            running_recon += float(loss.item())
            num_batches   += 1
            global_step   += 1

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        train_stats = {
            "loss":     running_loss  / max(1, num_batches),
            "recon":    running_recon / max(1, num_batches),
            "sparsity": 0.0,
        }
        val_stats = run_validation(model, val_loader, device)

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
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch":            epoch,
                    "global_step":      global_step,
                    "model_state":      model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "scheduler_state":  scheduler.state_dict(),
                    "best_val":         best_val,
                    "args": {
                        "in_channels":        3,
                        "resnet_variant":     args.resnet_variant,
                        "stage_channels":     list(args.stage_channels),
                        "stage_strides":      list(args.stage_strides),
                        "stage_blocks":       _mc.get("stage_blocks", None),
                        "kernel_size":        _mc.get("kernel_size", 3),
                        "use_stem":           _mc.get("use_stem", True),
                        "stem_channels":      _mc.get("stem_channels", 64),
                        "stem_stride":        args.stem_stride,
                        "use_stem_maxpool":   args.use_stem_maxpool,
                        "output_activation":  _mc.get("output_activation", "none"),
                    },
                    "train_stats":      train_stats,
                    "val_stats":        val_stats,
                },
                ckpt_path,
            )

        if val_stats["recon"] < best_val:
            best_val = val_stats["recon"]
            torch.save(
                {
                    "epoch":           epoch,
                    "global_step":     global_step,
                    "model_state":     model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val":        best_val,
                    "args": {
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
                    },
                    "train_stats":     train_stats,
                    "val_stats":       val_stats,
                },
                ckpt_dir / "best.pt",
            )
            print(f"  -> New best val_recon={best_val:.6f}  (saved best.pt)")

        if epoch % args.save_every == 0:
            preview_path = preview_dir / f"epoch_{epoch:04d}.png"
            save_recon_preview(model, val_loader, device, preview_path)
            save_training_curves(history, output_dir)

    save_training_curves(history, output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
