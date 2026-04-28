#!/usr/bin/env python3
"""Train IntermediateMatryoshkaBatchTopKSparseConvAutoencoder on CelebA-HQ.

Same backbone as the standard Matryoshka Batch-TopK SAE, but Batch-TopK is
applied at *every* encoder stage's output (taxon-AE style) and each
intermediate sparse activation is decoded by its own per-stage decoder head.
The loss is a weighted sum of per-stage MSEs plus the summed AuxK term.

Output directory suffix: _k<k1>-<k2>-..._sw<sparsity_weight>
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

from src.model.cnn.baseline.matryoshka_intermediate_topk_sae import (
    IntermediateMatryoshkaBatchTopKSparseConvAutoencoder,
)
from src.train._baseline_dead_frac import compute_dead_frac
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
    loss_weights: list,
    sparsity_weight: float,
) -> dict:
    model.eval()
    total_loss   = 0.0
    total_recons = [0.0] * len(loss_weights)
    total_sparse = 0.0
    num_batches  = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recons, aux_loss = model.forward_matryoshka(images)

        recon_losses = [float(F.mse_loss(r, images).item()) for r in recons]
        mat_loss = sum(w * rl for w, rl in zip(loss_weights, recon_losses))
        loss = mat_loss + sparsity_weight * float(aux_loss.item())

        total_loss   += loss
        for i, rl in enumerate(recon_losses):
            total_recons[i] += rl
        total_sparse += float(aux_loss.item())
        num_batches  += 1

    if num_batches == 0:
        return {"loss": 0.0, "recon_k0": 0.0, "sparsity": 0.0, "dead_frac": compute_dead_frac(model)}

    return {
        "loss":     total_loss   / num_batches,
        **{f"recon_k{i}": total_recons[i] / num_batches for i in range(len(loss_weights))},
        "sparsity": total_sparse / num_batches,
        "dead_frac": compute_dead_frac(model),
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
        recon, _ = model(images)        # uses largest k for preview
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

    k_values = history.get("k_values", [])
    n_levels = len(k_values)
    fig, axes = plt.subplots(1, 3 + n_levels, figsize=(5 * (3 + n_levels), 4))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="val",   linewidth=1.5, linestyle="--")
    ax.set_title("Total loss", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for i, k in enumerate(k_values):
        ax = axes[1 + i]
        ax.plot(epochs, history[f"train_recon_k{i}"], label="train", linewidth=1.5)
        ax.plot(epochs, history[f"val_recon_k{i}"],   label="val",   linewidth=1.5, linestyle="--")
        ax.set_title(f"Recon MSE (k={k})", fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[-2]
    ax.plot(epochs, history["train_sparsity"], label="train", linewidth=1.5)
    ax.plot(epochs, history["val_sparsity"],   label="val",   linewidth=1.5, linestyle="--")
    ax.set_title("AuxK loss", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[-1]
    ax.plot(epochs, history["train_dead"], label="train", linewidth=1.5)
    ax.plot(epochs, history["val_dead"],   label="val",   linewidth=1.5, linestyle="--")
    ax.set_title("Dead frac", fontsize=11)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Frac"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    k_str = "-".join(str(k) for k in k_values)
    plt.suptitle(
        f"Intermediate Matryoshka Batch-TopK SAE CelebA-HQ  (k per stage = {k_str})",
        fontsize=13, fontweight="bold"
    )
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

    parser = argparse.ArgumentParser(
        description="Train Intermediate-Stage Matryoshka Batch-TopK SAE on CelebA-HQ"
    )
    parser.add_argument("--config",          type=str,   default="")
    parser.add_argument("--data-root",       type=str,   default=d.get("data_root",       "./data/celeba_hq"))
    parser.add_argument("--output-dir",      type=str,   default=o.get("output_dir",      "./outputs/celeba_hq/sae_matryoshka_intermediate_topk_celeba_hq_r18"))
    parser.add_argument("--image-size",      type=int,   default=d.get("image_size",      256))
    parser.add_argument("--batch-size",      type=int,   default=d.get("batch_size",      32))
    parser.add_argument("--num-workers",     type=int,   default=d.get("num_workers",     8))
    parser.add_argument("--val-split",       type=float, default=d.get("val_split",       0.05))
    parser.add_argument("--resnet-variant",  type=str,   default=m.get("resnet_variant",  "18"))
    parser.add_argument("--stage-channels",  type=int,   nargs=4,
                        default=m.get("stage_channels",  [64, 128, 256, 512]))
    parser.add_argument("--stage-strides",   type=int,   nargs=4,
                        default=m.get("stage_strides",   [1, 2, 2, 2]))
    parser.add_argument("--k-values",        type=int,   nargs="+",
                        default=m.get("k_values",        [32, 16, 8, 4]))
    parser.add_argument("--loss-weights",    type=float, nargs="+",
                        default=t.get("loss_weights",    None))
    parser.add_argument("--k-aux",           type=int,   default=m.get("k_aux",           None))
    parser.add_argument("--use-aux-loss",    action=argparse.BooleanOptionalAction,
                        default=m.get("use_aux_loss", True))
    parser.add_argument("--dead-threshold",  type=float, default=m.get("dead_threshold",  1e-3))
    parser.add_argument("--stem-stride",     type=int,   default=m.get("stem_stride",     2))
    parser.add_argument("--use-stem-maxpool", action=argparse.BooleanOptionalAction,
                        default=m.get("use_stem_maxpool", True))
    parser.add_argument("--epochs",          type=int,   default=t.get("epochs",          90))
    parser.add_argument("--learning-rate",   type=float, default=t.get("learning_rate",   3e-4))
    parser.add_argument("--weight-decay",    type=float, default=t.get("weight_decay",    1e-4))
    parser.add_argument("--warmup-epochs",   type=int,   default=t.get("warmup_epochs",   3))
    parser.add_argument("--sparsity-weight", type=float, default=t.get("sparsity_weight", 1e-2))
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

    # Per-stage k_values: keep user order (one per encoder stage, shallow→deep).
    k_values = list(args.k_values)
    k_str    = "-".join(str(k) for k in k_values)

    # Validate / default loss_weights
    if args.loss_weights is None:
        loss_weights = [1.0] * len(k_values)
    else:
        if len(args.loss_weights) != len(k_values):
            raise ValueError(
                f"--loss-weights length ({len(args.loss_weights)}) must match "
                f"--k-values length ({len(k_values)})."
            )
        loss_weights = args.loss_weights

    output_dir  = Path(args.output_dir + f"_k{k_str}_sw{args.sparsity_weight:.0e}")
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
        raise RuntimeError("val_split must be > 0")

    _mc: dict = {}
    if args.config:
        with open(args.config) as f:
            _mc = json.load(f).get("model", {})

    if len(k_values) != len(args.stage_channels):
        raise ValueError(
            f"--k-values length ({len(k_values)}) must equal number of stages "
            f"({len(args.stage_channels)})."
        )

    k_aux = args.k_aux if args.k_aux is not None else min(k_values)

    model = IntermediateMatryoshkaBatchTopKSparseConvAutoencoder(
        in_channels=_mc.get("in_channels", 3),
        resnet_variant=args.resnet_variant,
        stage_channels=tuple(args.stage_channels),
        stage_strides=tuple(args.stage_strides),
        stage_blocks=_mc.get("stage_blocks", None),
        k_values=k_values,
        k_aux=k_aux,
        use_aux_loss=args.use_aux_loss,
        dead_threshold=args.dead_threshold,
        kernel_size=_mc.get("kernel_size", 3),
        use_stem=_mc.get("use_stem", True),
        stem_channels=_mc.get("stem_channels", 64),
        stem_stride=args.stem_stride,
        use_stem_maxpool=args.use_stem_maxpool,
        output_activation=_mc.get("output_activation", "none"),
    ).to(device)

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer, max(1, len(train_loader)), args.epochs, args.warmup_epochs
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "Training setup (Intermediate-Stage Matryoshka Batch-TopK SAE CelebA-HQ):\n"
        f"  device={device}  n_params={n_params:,}\n"
        f"  k_values (per stage, shallow→deep)={k_values}  loss_weights={loss_weights}\n"
        f"  sparsity_weight={args.sparsity_weight}\n"
        f"  train_size={len(celeba_loader.trainset)}  val_size={len(celeba_loader.valset)}\n"
        f"  batch_size={args.batch_size}  epochs={args.epochs}"
    )

    history: dict = {
        "epochs":   [],
        "k_values": k_values,
        "train_loss":    [], "val_loss":    [],
        "train_sparsity":[], "val_sparsity":[],
        "train_dead":    [], "val_dead":    [],
        **{f"train_recon_k{i}": [] for i in range(len(k_values))},
        **{f"val_recon_k{i}":   [] for i in range(len(k_values))},
    }

    _args_dict = {
        "model_variant":    "matryoshka_intermediate_batch_topk",
        "in_channels":      3,
        "resnet_variant":   args.resnet_variant,
        "stage_channels":   list(args.stage_channels),
        "stage_strides":    list(args.stage_strides),
        "stage_blocks":     _mc.get("stage_blocks", None),
        "k_values":         k_values,
        "k_aux":            k_aux,
        "loss_weights":     loss_weights,
        "use_aux_loss":     args.use_aux_loss,
        "dead_threshold":   args.dead_threshold,
        "kernel_size":      _mc.get("kernel_size", 3),
        "use_stem":         _mc.get("use_stem", True),
        "stem_channels":    _mc.get("stem_channels", 64),
        "stem_stride":      args.stem_stride,
        "use_stem_maxpool": args.use_stem_maxpool,
        "output_activation":_mc.get("output_activation", "none"),
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss   = 0.0
        running_recons = [0.0] * len(k_values)
        running_sparse = 0.0
        running_dead   = 0.0
        num_batches    = 0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            recons, aux_loss = model.forward_matryoshka(images)

            recon_losses = [F.mse_loss(r, images) for r in recons]
            mat_loss = sum(w * rl for w, rl in zip(loss_weights, recon_losses))
            loss = mat_loss + args.sparsity_weight * aux_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss   += float(loss.item())
            for i, rl in enumerate(recon_losses):
                running_recons[i] += float(rl.item())
            running_sparse += float(aux_loss.item())
            running_dead   += compute_dead_frac(model)
            num_batches    += 1
            global_step    += 1

            if batch_idx % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                recon0 = running_recons[0] / num_batches
                print(
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} step={global_step} "
                    f"lr={lr:.3e} loss={running_loss/num_batches:.5f} "
                    f"recon_k0={recon0:.5f} aux={running_sparse/num_batches:.5f} "
                    f"dead={running_dead/num_batches:.3f}"
                )

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        nb = max(1, num_batches)
        train_stats = {
            "loss":     running_loss   / nb,
            **{f"recon_k{i}": running_recons[i] / nb for i in range(len(k_values))},
            "sparsity": running_sparse / nb,
            "dead":     running_dead   / nb,
        }
        val_stats = run_validation(model, val_loader, device, loss_weights, args.sparsity_weight)

        elapsed = time.time() - epoch_start
        print(
            f"epoch={epoch:03d} time={elapsed:.1f}s "
            f"train_loss={train_stats['loss']:.5f} "
            f"train_recon_k0={train_stats['recon_k0']:.5f} "
            f"train_dead={train_stats['dead']:.3f} "
            f"val_loss={val_stats['loss']:.5f} "
            f"val_recon_k0={val_stats['recon_k0']:.5f} "
            f"val_dead={val_stats['dead_frac']:.3f}"
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_sparsity"].append(train_stats["sparsity"])
        history["val_sparsity"].append(val_stats["sparsity"])
        history["train_dead"].append(train_stats["dead"])
        history["val_dead"].append(val_stats["dead_frac"])
        for i in range(len(k_values)):
            history[f"train_recon_k{i}"].append(train_stats[f"recon_k{i}"])
            history[f"val_recon_k{i}"].append(val_stats[f"recon_k{i}"])

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

        if epoch % args.save_every == 0:
            torch.save(state, ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt")
            torch.save(state, ckpt_dir / "latest.pt")

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            state["best_val"] = best_val
            torch.save(state, ckpt_dir / "best.pt")

        save_recon_preview(model, val_loader, device, preview_dir / f"epoch_{epoch:03d}.png")

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            print(f"Reached max_train_steps={args.max_train_steps}; stopping early.")
            break

    save_training_curves(history, output_dir)


if __name__ == "__main__":
    main()
