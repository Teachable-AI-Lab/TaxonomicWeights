#!/usr/bin/env python3
"""Train TopKSparseConvAutoencoder on CIFAR-10.

Mirrors src/train/train_sae_cifar10.py.  Sparsity is controlled exclusively
by ``topk_k`` (number of active channels per spatial position).  The optional
AuxK auxiliary loss is weighted by ``sparsity_weight``.

Loss:
    total = MSE(recon, x) + sparsity_weight * aux_loss
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
from torchvision.utils import make_grid, save_image

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.topk_sae import TopKSparseConvAutoencoder
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
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    sparsity_weight: float,
) -> dict:
    model.eval()
    total_loss = total_recon = total_sparse = 0.0
    num_batches = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recon, aux_loss = model(images)
        recon_loss = F.mse_loss(recon, images)
        loss = recon_loss + sparsity_weight * aux_loss

        total_loss   += float(loss.item())
        total_recon  += float(recon_loss.item())
        total_sparse += float(aux_loss.item())
        num_batches  += 1

    if num_batches == 0:
        return {"loss": 0.0, "recon": 0.0, "sparsity": 0.0}

    return {
        "loss":     total_loss   / num_batches,
        "recon":    total_recon  / num_batches,
        "sparsity": total_sparse / num_batches,
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
    vis_recon = (recon.clamp(-1, 1)  + 1.0) * 0.5
    grid = make_grid(torch.cat([vis_input, vis_recon], dim=0), nrow=num_images)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, save_path)


def save_training_curves(history: dict, output_dir: Path) -> None:
    epochs = history["epochs"]
    if not epochs:
        return

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("Total loss",    "train_loss",     "val_loss"),
        ("Recon loss",    "train_recon",    "val_recon"),
        ("AuxK loss",     "train_sparsity", "val_sparsity"),
    ]
    for ax, (title, tk, vk) in zip(axes, panels):
        ax.plot(epochs, history[tk], label="train", linewidth=1.5)
        ax.plot(epochs, history[vk], label="val",   linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if tk == "train_loss":
            best_ep = epochs[int(min(range(len(history[vk])),
                                    key=lambda i: history[vk][i]))]
            ax.axvline(best_ep, color="red", linestyle=":", linewidth=1.0,
                       label=f"best val (ep {best_ep})")
            ax.legend(fontsize=8)

    plt.suptitle(f"TopK SAE CIFAR-10 training curves  (k={history.get('topk_k', '?')})",
                 fontsize=13, fontweight="bold")
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

    parser = argparse.ArgumentParser(description="Train TopK SAE on CIFAR-10")
    parser.add_argument("--config",          type=str,   default="")
    parser.add_argument("--data-root",       type=str,   default=d.get("data_root",       "./data/cifar10"))
    parser.add_argument("--output-dir",      type=str,   default=o.get("output_dir",      "./outputs/sae_topk_cifar10_r18"))
    parser.add_argument("--batch-size",      type=int,   default=d.get("batch_size",      128))
    parser.add_argument("--num-workers",     type=int,   default=d.get("num_workers",     4))
    parser.add_argument("--resnet-variant",  type=str,   default=m.get("resnet_variant",  "18"))
    parser.add_argument("--stage-channels",  type=int,   nargs=4,
                        default=m.get("stage_channels",  [64, 128, 256, 512]))
    parser.add_argument("--stage-strides",   type=int,   nargs=4,
                        default=m.get("stage_strides",   [1, 2, 2, 2]))
    parser.add_argument("--topk-k",          type=int,   default=m.get("topk_k",          64))
    parser.add_argument("--k-aux",           type=int,   default=m.get("k_aux",           64))
    parser.add_argument("--use-aux-loss",    action=argparse.BooleanOptionalAction,
                        default=m.get("use_aux_loss", True))
    parser.add_argument("--dead-threshold",  type=float, default=m.get("dead_threshold",  1e-3))
    parser.add_argument("--stem-stride",     type=int,   default=m.get("stem_stride",     1))
    parser.add_argument("--use-stem-maxpool", action=argparse.BooleanOptionalAction,
                        default=m.get("use_stem_maxpool", False))
    parser.add_argument("--epochs",          type=int,   default=t.get("epochs",          90))
    parser.add_argument("--learning-rate",   type=float, default=t.get("learning_rate",   3e-4))
    parser.add_argument("--weight-decay",    type=float, default=t.get("weight_decay",    1e-4))
    parser.add_argument("--warmup-epochs",   type=int,   default=t.get("warmup_epochs",   3))
    parser.add_argument("--sparsity-weight", type=float, default=t.get("sparsity_weight", 1e-4))
    parser.add_argument("--save-every",      type=int,   default=t.get("save_every",      5))
    parser.add_argument("--seed",            type=int,   default=t.get("seed",            42))
    parser.add_argument("--max-train-steps", type=int,   default=t.get("max_train_steps", 0))
    parser.add_argument("--resume",          type=str,   default="")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir  = Path(args.output_dir + f"_k{args.topk_k}_sw{args.sparsity_weight:.0e}")
    ckpt_dir    = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader_obj = CIFAR10Loader(batch_size=args.batch_size, root=args.data_root)
    train_loader, val_loader = loader_obj.get_loaders()

    _mc: dict = {}
    if args.config:
        with open(args.config) as f:
            _mc = json.load(f).get("model", {})

    model = TopKSparseConvAutoencoder(
        in_channels=_mc.get("in_channels", 3),
        resnet_variant=args.resnet_variant,
        stage_channels=tuple(args.stage_channels),
        stage_strides=tuple(args.stage_strides),
        stage_blocks=_mc.get("stage_blocks", None),
        topk_k=args.topk_k,
        k_aux=args.k_aux,
        use_aux_loss=args.use_aux_loss,
        dead_threshold=args.dead_threshold,
        kernel_size=_mc.get("kernel_size", 3),
        use_stem=_mc.get("use_stem", True),
        stem_channels=_mc.get("stem_channels", 64),
        stem_stride=args.stem_stride,
        use_stem_maxpool=args.use_stem_maxpool,
        output_activation=_mc.get("output_activation", "none"),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, max(1, len(train_loader)),
                                 args.epochs, args.warmup_epochs)

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
        "Training setup (TopK SAE CIFAR-10):\n"
        f"  device={device}  topk_k={args.topk_k}  sparsity_weight={args.sparsity_weight}\n"
        f"  train_size={len(loader_obj.trainset)}  val_size={len(loader_obj.testset)}\n"
        f"  batch_size={args.batch_size}  epochs={args.epochs}\n"
        f"  lr={args.learning_rate}  wd={args.weight_decay}"
    )

    history: dict = {
        "epochs": [], "topk_k": args.topk_k,
        "train_loss": [], "train_recon": [], "train_sparsity": [],
        "val_loss":   [], "val_recon":   [], "val_sparsity":   [],
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = running_recon = running_sparse = 0.0
        num_batches = 0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            recon, aux_loss = model(images)
            recon_loss = F.mse_loss(recon, images)
            loss = recon_loss + args.sparsity_weight * aux_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss   += float(loss.item())
            running_recon  += float(recon_loss.item())
            running_sparse += float(aux_loss.item())
            num_batches    += 1
            global_step    += 1

            if batch_idx % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} step={global_step} "
                    f"lr={lr:.3e} loss={running_loss/num_batches:.5f} "
                    f"recon={running_recon/num_batches:.5f} aux={running_sparse/num_batches:.5f}"
                )

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        train_stats = {
            "loss":     running_loss   / max(1, num_batches),
            "recon":    running_recon  / max(1, num_batches),
            "sparsity": running_sparse / max(1, num_batches),
        }
        val_stats = run_validation(model, val_loader, device, args.sparsity_weight)

        elapsed = time.time() - epoch_start
        print(
            f"epoch={epoch:03d} time={elapsed:.1f}s "
            f"train_loss={train_stats['loss']:.5f} train_recon={train_stats['recon']:.5f} "
            f"val_loss={val_stats['loss']:.5f} val_recon={val_stats['recon']:.5f}"
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["train_recon"].append(train_stats["recon"])
        history["train_sparsity"].append(train_stats["sparsity"])
        history["val_loss"].append(val_stats["loss"])
        history["val_recon"].append(val_stats["recon"])
        history["val_sparsity"].append(val_stats["sparsity"])

        # Save model args for checkpoint reconstruction
        _args_dict = {
            "model_variant": "topk",
            "in_channels": 3,
            "resnet_variant": args.resnet_variant,
            "stage_channels": list(args.stage_channels),
            "stage_strides": list(args.stage_strides),
            "stage_blocks": _mc.get("stage_blocks", None),
            "topk_k": args.topk_k,
            "k_aux": args.k_aux,
            "use_aux_loss": args.use_aux_loss,
            "dead_threshold": args.dead_threshold,
            "kernel_size": _mc.get("kernel_size", 3),
            "use_stem": _mc.get("use_stem", True),
            "stem_channels": _mc.get("stem_channels", 64),
            "stem_stride": args.stem_stride,
            "use_stem_maxpool": args.use_stem_maxpool,
            "output_activation": _mc.get("output_activation", "none"),
        }

        if epoch % args.save_every == 0:
            state = {
                "epoch": epoch, "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val, "args": _args_dict,
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
                "best_val": best_val, "args": _args_dict,
                "train_stats": train_stats, "val_stats": val_stats,
            }, ckpt_dir / "best.pt")

        save_recon_preview(model, val_loader, device,
                           preview_dir / f"epoch_{epoch:03d}.png")

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            print(f"Reached max_train_steps={args.max_train_steps}; stopping early.")
            break

    save_training_curves(history, output_dir)


if __name__ == "__main__":
    main()
