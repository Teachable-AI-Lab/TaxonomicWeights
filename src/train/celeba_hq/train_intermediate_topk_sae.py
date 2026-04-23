#!/usr/bin/env python3
"""Train IntermediateTopKSparseConvAutoencoder on CelebA-HQ.

Per-position TopK is applied at every encoder stage output (exact ``k_values[i]``
active channels at each spatial location). Each stage gets its own decoder head.
Loss = Σ_i loss_weights[i] * MSE(recon_i, x)  +  sparsity_weight * Σ_i auxk_i

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

from src.model.cnn.baseline.intermediate_topk_sae import IntermediateTopKSparseConvAutoencoder
from src.utils.dataloader import CelebAHQLoader


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs) -> LambdaLR:
    total_steps  = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def run_validation(model, loader: DataLoader, device, loss_weights, sparsity_weight) -> dict:
    model.eval()
    total_loss   = 0.0
    total_recons = [0.0] * len(loss_weights)
    total_sparse = 0.0
    nb = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recons, aux_loss = model.forward_matryoshka(images, use_checkpoint=False)
        recon_losses = [float(F.mse_loss(r, images).item()) for r in recons]
        mat_loss = sum(w * rl for w, rl in zip(loss_weights, recon_losses))
        total_loss   += mat_loss + sparsity_weight * float(aux_loss.item())
        for i, rl in enumerate(recon_losses):
            total_recons[i] += rl
        total_sparse += float(aux_loss.item())
        nb += 1
    if nb == 0:
        return {"loss": 0.0, **{f"recon_k{i}": 0.0 for i in range(len(loss_weights))}, "sparsity": 0.0}
    return {
        "loss":     total_loss / nb,
        **{f"recon_k{i}": total_recons[i] / nb for i in range(len(loss_weights))},
        "sparsity": total_sparse / nb,
    }


def save_recon_preview(model, loader: DataLoader, device, save_path: Path, num_images=8) -> None:
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

    k_values  = history.get("k_values", [])
    n_levels  = len(k_values)
    fig, axes = plt.subplots(1, 2 + n_levels, figsize=(5 * (2 + n_levels), 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"],   label="val", linestyle="--")
    axes[0].set_title("Total loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    for i, k in enumerate(k_values):
        ax = axes[1 + i]
        ax.plot(epochs, history[f"train_recon_k{i}"], label="train")
        ax.plot(epochs, history[f"val_recon_k{i}"],   label="val", linestyle="--")
        ax.set_title(f"Recon MSE (k={k})"); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True, alpha=0.3)

    axes[-1].plot(epochs, history["train_sparsity"], label="train")
    axes[-1].plot(epochs, history["val_sparsity"],   label="val", linestyle="--")
    axes[-1].set_title("AuxK loss"); axes[-1].set_xlabel("Epoch")
    axes[-1].legend(); axes[-1].grid(True, alpha=0.3)

    k_str = "-".join(str(k) for k in k_values)
    plt.suptitle(f"Intermediate Per-Position TopK SAE  (k per stage = {k_str})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


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

    p = argparse.ArgumentParser(description="Train Intermediate Per-Position TopK SAE on CelebA-HQ")
    p.add_argument("--config",           type=str,   default="")
    p.add_argument("--data-root",        type=str,   default=d.get("data_root",       "./data/celeba_hq"))
    p.add_argument("--output-dir",       type=str,   default=o.get("output_dir",      "./outputs/celeba_hq/sae_intermediate_topk_celeba_hq_r18"))
    p.add_argument("--image-size",       type=int,   default=d.get("image_size",      256))
    p.add_argument("--batch-size",       type=int,   default=d.get("batch_size",      16))
    p.add_argument("--num-workers",      type=int,   default=d.get("num_workers",     8))
    p.add_argument("--val-split",        type=float, default=d.get("val_split",       0.05))
    p.add_argument("--resnet-variant",   type=str,   default=m.get("resnet_variant",  "18"))
    p.add_argument("--stage-channels",   type=int,   nargs=4,
                   default=m.get("stage_channels",   [64, 128, 256, 512]))
    p.add_argument("--stage-strides",    type=int,   nargs=4,
                   default=m.get("stage_strides",    [1, 2, 2, 2]))
    p.add_argument("--k-values",         type=int,   nargs="+",
                   default=m.get("k_values",         [32, 16, 8, 4]))
    p.add_argument("--loss-weights",     type=float, nargs="+",
                   default=t.get("loss_weights",     None))
    p.add_argument("--k-aux",            type=int,   default=m.get("k_aux",           None))
    p.add_argument("--use-aux-loss",     action=argparse.BooleanOptionalAction,
                   default=m.get("use_aux_loss", True))
    p.add_argument("--dead-threshold",   type=float, default=m.get("dead_threshold",  1e-3))
    p.add_argument("--stem-stride",      type=int,   default=m.get("stem_stride",     2))
    p.add_argument("--use-stem-maxpool", action=argparse.BooleanOptionalAction,
                   default=m.get("use_stem_maxpool", True))
    p.add_argument("--epochs",           type=int,   default=t.get("epochs",          90))
    p.add_argument("--learning-rate",    type=float, default=t.get("learning_rate",   3e-4))
    p.add_argument("--weight-decay",     type=float, default=t.get("weight_decay",    1e-4))
    p.add_argument("--warmup-epochs",    type=int,   default=t.get("warmup_epochs",   3))
    p.add_argument("--sparsity-weight",  type=float, default=t.get("sparsity_weight", 1e-2))
    p.add_argument("--save-every",       type=int,   default=t.get("save_every",      5))
    p.add_argument("--seed",             type=int,   default=t.get("seed",            42))
    p.add_argument("--max-train-steps",  type=int,   default=t.get("max_train_steps", 0))
    p.add_argument("--resume",           type=str,   default="")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    k_values = list(args.k_values)
    k_str    = "-".join(str(k) for k in k_values)

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

    model = IntermediateTopKSparseConvAutoencoder(
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "Training setup (Intermediate Per-Position TopK SAE CelebA-HQ):\n"
        f"  device={device}  n_params={n_params:,}\n"
        f"  k_values (per stage, shallow→deep)={k_values}  loss_weights={loss_weights}\n"
        f"  sparsity_weight={args.sparsity_weight}\n"
        f"  train_size={len(celeba_loader.trainset)}  val_size={len(celeba_loader.valset)}\n"
        f"  batch_size={args.batch_size}  epochs={args.epochs}\n"
        f"  output={output_dir}"
    )

    _args_dict = {
        "model_variant":    "intermediate_topk",
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
        "output_activation": _mc.get("output_activation", "none"),
    }

    history: dict = {
        "epochs":   [],
        "k_values": k_values,
        "train_loss":    [], "val_loss":    [],
        "train_sparsity":[], "val_sparsity":[],
        **{f"train_recon_k{i}": [] for i in range(len(k_values))},
        **{f"val_recon_k{i}":   [] for i in range(len(k_values))},
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start    = time.time()
        running_loss   = 0.0
        running_recons = [0.0] * len(k_values)
        running_sparse = 0.0
        nb             = 0

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
            nb             += 1
            global_step    += 1

            if batch_idx % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} step={global_step} "
                    f"lr={lr:.3e} loss={running_loss/nb:.5f} "
                    f"recon_k0={running_recons[0]/nb:.5f} aux={running_sparse/nb:.5f}"
                )

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        nb = max(1, nb)
        train_stats = {
            "loss":     running_loss   / nb,
            **{f"recon_k{i}": running_recons[i] / nb for i in range(len(k_values))},
            "sparsity": running_sparse / nb,
        }
        val_stats = run_validation(model, val_loader, device, loss_weights, args.sparsity_weight)

        elapsed = time.time() - epoch_start
        print(
            f"epoch={epoch:03d} time={elapsed:.1f}s "
            f"train_loss={train_stats['loss']:.5f} "
            f"train_recon_k0={train_stats['recon_k0']:.5f} "
            f"val_loss={val_stats['loss']:.5f} "
            f"val_recon_k0={val_stats['recon_k0']:.5f}"
        )

        history["epochs"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["val_loss"].append(val_stats["loss"])
        history["train_sparsity"].append(train_stats["sparsity"])
        history["val_sparsity"].append(val_stats["sparsity"])
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
