#!/usr/bin/env python3
"""Train SoftmaxSparseConvAutoencoder on CelebA-HQ with a ResNet-18 stage layout.

Uses the same softmax gating + DKL/entropy regularization as TaxonAutoencoder
but without any hierarchical tree structure.  This is the direct ablation to
test whether the taxonomic inductive bias adds value beyond the penalty alone.

Loss::

    total = MSE(recon, x) + dkl_weight * dkl + entropy_weight * entropy
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

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.baseline.softmax_sae import SoftmaxSparseConvAutoencoder
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
    """Cosine decay with linear warmup (step-wise)."""
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
    dkl_weight: float,
    entropy_weight: float,
) -> dict:
    model.eval()
    total_loss = total_recon = total_dkl = total_entropy = 0.0
    num_batches = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recon, dkl, entropy = model(images)
        recon_loss = F.mse_loss(recon, images)
        loss = recon_loss + dkl_weight * dkl + entropy_weight * entropy

        total_loss    += float(loss.item())
        total_recon   += float(recon_loss.item())
        total_dkl     += float(dkl.item())
        total_entropy += float(entropy.item())
        num_batches   += 1

    if num_batches == 0:
        return {"loss": 0.0, "recon": 0.0, "dkl": 0.0, "entropy": 0.0}

    return {
        "loss":    total_loss    / num_batches,
        "recon":   total_recon   / num_batches,
        "dkl":     total_dkl     / num_batches,
        "entropy": total_entropy / num_batches,
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
        recon, _, _ = model(images)

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
        ("Total loss",      "train_loss",    "val_loss"),
        ("Recon loss",      "train_recon",   "val_recon"),
        ("DKL penalty",     "train_dkl",     "val_dkl"),
        ("Entropy penalty", "train_entropy", "val_entropy"),
    ]

    for ax, (title, train_key, val_key) in zip(axes, panels):
        ax.plot(epochs, history[train_key], label="train", linewidth=1.5)
        ax.plot(epochs, history[val_key],   label="val",   linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if train_key == "train_loss":
            best_ep = epochs[int(min(range(len(history[val_key])),
                                    key=lambda i: history[val_key][i]))]
            ax.axvline(best_ep, color="red", linestyle=":", linewidth=1.0,
                       label=f"best val (ep {best_ep})")
            ax.legend(fontsize=8)

    plt.suptitle("Softmax SAE CelebA-HQ training curves", fontsize=13, fontweight="bold")
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

    d = cfg.get("data", {})
    m = cfg.get("model", {})
    t = cfg.get("training", {})
    o = cfg.get("output", {})

    parser = argparse.ArgumentParser(description="Train Softmax SAE on CelebA-HQ")
    parser.add_argument("--config", type=str, default="")
    # data
    parser.add_argument("--data-root",   type=str,   default=d.get("data_root", "./data/celeba_hq"))
    parser.add_argument("--output-dir",  type=str,   default=o.get("output_dir", "./outputs/celeba_hq/sae_softmax_celeba_hq_r18"))
    parser.add_argument("--image-size",  type=int,   default=d.get("image_size", 256))
    parser.add_argument("--batch-size",  type=int,   default=d.get("batch_size", 32))
    parser.add_argument("--num-workers", type=int,   default=d.get("num_workers", 8))
    parser.add_argument("--val-split",   type=float, default=d.get("val_split", 0.05))
    # model
    parser.add_argument("--resnet-variant", type=str, default=m.get("resnet_variant", "18"))
    parser.add_argument("--stage-channels", type=int, nargs=4,
                        default=m.get("stage_channels", [64, 128, 256, 512]))
    parser.add_argument("--stage-strides",  type=int, nargs=4,
                        default=m.get("stage_strides", [1, 2, 2, 2]))
    parser.add_argument("--temperature",    type=float, default=m.get("temperature", 1.0),
                        help="Softmax temperature (lower = sparser latent)")
    parser.add_argument("--stem-stride",    type=int,   default=m.get("stem_stride", 2))
    parser.add_argument("--use-stem-maxpool", action="store_true",
                        default=m.get("use_stem_maxpool", True))
    # training
    parser.add_argument("--epochs",          type=int,   default=t.get("epochs",          90))
    parser.add_argument("--learning-rate",   type=float, default=t.get("learning_rate",   3e-4))
    parser.add_argument("--weight-decay",    type=float, default=t.get("weight_decay",    1e-4))
    parser.add_argument("--warmup-epochs",   type=int,   default=t.get("warmup_epochs",   3))
    parser.add_argument("--dkl-weight",      type=float, default=t.get("dkl_weight",      1e-2),
                        help="Weight for the coverage KL penalty")
    parser.add_argument("--entropy-weight",  type=float, default=t.get("entropy_weight",  0.0),
                        help="Weight for the entropy penalty (positive = sparser)")
    parser.add_argument("--save-every",      type=int,   default=t.get("save_every",      5))
    parser.add_argument("--seed",            type=int,   default=t.get("seed",            42))
    parser.add_argument("--max-train-steps", type=int,   default=t.get("max_train_steps", 0))
    parser.add_argument("--resume",          type=str,   default="")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dkl_suffix     = f"_dkl_{args.dkl_weight:.0e}"
    temp_str       = f"{args.temperature:g}".replace(".", "p")
    temp_suffix    = f"_temp_{temp_str}"
    entropy_suffix = f"_ew_{args.entropy_weight:.0e}" if args.entropy_weight else ""
    run_suffix     = dkl_suffix + temp_suffix + entropy_suffix
    output_dir  = Path(args.output_dir + run_suffix)
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
        with open(args.config) as _f:
            _mc = json.load(_f).get("model", {})

    model = SoftmaxSparseConvAutoencoder(
        in_channels=_mc.get("in_channels", 3),
        resnet_variant=args.resnet_variant,
        stage_channels=tuple(args.stage_channels),
        stage_strides=tuple(args.stage_strides),
        stage_blocks=_mc.get("stage_blocks", None),
        temperature=args.temperature,
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
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch  = int(state["epoch"]) + 1
        global_step  = int(state.get("global_step", 0))
        best_val     = float(state.get("best_val", float("inf")))
        print(f"Resumed from {args.resume} at epoch={start_epoch}")

    # Build the args dict that goes into every checkpoint (used by load_model)
    _args_dict = {
        "model_variant":    "softmax_sae",
        "in_channels":      3,
        "resnet_variant":   args.resnet_variant,
        "stage_channels":   list(args.stage_channels),
        "stage_strides":    list(args.stage_strides),
        "stage_blocks":     _mc.get("stage_blocks", None),
        "temperature":      args.temperature,
        "dkl_weight":       args.dkl_weight,
        "entropy_weight":   args.entropy_weight,
        "kernel_size":      _mc.get("kernel_size", 3),
        "use_stem":         _mc.get("use_stem", True),
        "stem_channels":    _mc.get("stem_channels", 64),
        "stem_stride":      args.stem_stride,
        "use_stem_maxpool": args.use_stem_maxpool,
        "output_activation":_mc.get("output_activation", "none"),
    }

    print(
        "Training setup (Softmax SAE CelebA-HQ):\n"
        f"  device={device}\n"
        f"  temperature={args.temperature}  dkl_weight={args.dkl_weight}"
        f"  entropy_weight={args.entropy_weight}\n"
        f"  train_size={len(celeba_loader.trainset)}  val_size={len(celeba_loader.valset)}\n"
        f"  batch_size={args.batch_size}  epochs={args.epochs}\n"
        f"  lr={args.learning_rate}  wd={args.weight_decay}\n"
        f"  output_dir={output_dir}"
    )

    history: dict = {
        "epochs":        [],
        "train_loss":    [],
        "train_recon":   [],
        "train_dkl":     [],
        "train_entropy": [],
        "val_loss":      [],
        "val_recon":     [],
        "val_dkl":       [],
        "val_entropy":   [],
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start     = time.time()
        running_loss    = 0.0
        running_recon   = 0.0
        running_dkl     = 0.0
        running_entropy = 0.0
        num_batches     = 0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            recon, dkl, entropy = model(images)
            recon_loss = F.mse_loss(recon, images)
            loss = recon_loss + args.dkl_weight * dkl + args.entropy_weight * entropy

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss    += float(loss.item())
            running_recon   += float(recon_loss.item())
            running_dkl     += float(dkl.item())
            running_entropy += float(entropy.item())
            num_batches     += 1
            global_step     += 1

            if batch_idx % 50 == 0:
                avg_loss    = running_loss    / num_batches
                avg_recon   = running_recon   / num_batches
                avg_dkl     = running_dkl     / num_batches
                avg_entropy = running_entropy / num_batches
                lr          = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} step={global_step} "
                    f"lr={lr:.3e} loss={avg_loss:.5f} recon={avg_recon:.5f} "
                    f"dkl={avg_dkl:.5f} entropy={avg_entropy:.5f}"
                )

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        train_stats = {
            "loss":    running_loss    / max(1, num_batches),
            "recon":   running_recon   / max(1, num_batches),
            "dkl":     running_dkl     / max(1, num_batches),
            "entropy": running_entropy / max(1, num_batches),
        }

        val_stats = run_validation(
            model=model,
            loader=val_loader,
            device=device,
            dkl_weight=args.dkl_weight,
            entropy_weight=args.entropy_weight,
        )

        elapsed = time.time() - epoch_start
        print(
            f"epoch={epoch:03d} time={elapsed:.1f}s "
            f"train_loss={train_stats['loss']:.5f} train_recon={train_stats['recon']:.5f} "
            f"val_loss={val_stats['loss']:.5f} val_recon={val_stats['recon']:.5f}"
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["train_recon"].append(train_stats["recon"])
        history["train_dkl"].append(train_stats["dkl"])
        history["train_entropy"].append(train_stats["entropy"])
        history["val_loss"].append(val_stats["loss"])
        history["val_recon"].append(val_stats["recon"])
        history["val_dkl"].append(val_stats["dkl"])
        history["val_entropy"].append(val_stats["entropy"])

        if epoch % args.save_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            state = {
                "epoch":           epoch,
                "global_step":     global_step,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val":        best_val,
                "args":            _args_dict,
                "train_stats":     train_stats,
                "val_stats":       val_stats,
            }
            torch.save(state, ckpt_path)
            torch.save(state, ckpt_dir / "latest.pt")

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_state = {
                "epoch":           epoch,
                "global_step":     global_step,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val":        best_val,
                "args":            _args_dict,
                "train_stats":     train_stats,
                "val_stats":       val_stats,
            }
            torch.save(best_state, ckpt_dir / "best.pt")

        save_recon_preview(
            model=model,
            loader=val_loader,
            device=device,
            save_path=preview_dir / f"epoch_{epoch:03d}.png",
            num_images=8,
        )

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            print(f"Reached max_train_steps={args.max_train_steps}; stopping early.")
            break

    save_training_curves(history, output_dir)


if __name__ == "__main__":
    main()
