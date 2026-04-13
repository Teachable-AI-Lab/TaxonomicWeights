#!/usr/bin/env python3
"""Train BiasTaxonAutoencoder on ImageNet-1k with ResNet-18 stage layout.

Uses sigmoid routing with non-differentiable per-leaf biases for load
balancing.  No auxiliary losses — diversity is maintained purely through
the heuristic bias update rule.

Loss:
    recon (MSE only — no regularisation terms)
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

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.taxon.bias_taxon_ae import BiasTaxonAutoencoder
from src.utils.dataloader import ImageNet1kHFLoader


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
) -> dict:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        (recon,) = model(images)
        recon_loss = F.mse_loss(recon, images)
        total_loss += float(recon_loss.item())
        num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "recon": 0.0}

    return {
        "loss":  total_loss / num_batches,
        "recon": total_loss / num_batches,
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
        (recon,) = model(images)

    vis_input = (images.clamp(-1, 1) + 1.0) * 0.5
    vis_recon = (recon.clamp(-1, 1) + 1.0) * 0.5

    grid = make_grid(torch.cat([vis_input, vis_recon], dim=0), nrow=num_images)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, save_path)


def save_training_curves(history: dict, output_dir: Path) -> None:
    import json as _json

    epochs = history["epochs"]
    if not epochs:
        return

    with open(output_dir / "training_history.json", "w") as _f:
        _json.dump(history, _f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="train", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="val",   linewidth=1.5, linestyle="--")
    ax.set_title("Reconstruction Loss", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    best_ep = epochs[int(min(range(len(history["val_loss"])),
                            key=lambda i: history["val_loss"][i]))]
    ax.axvline(best_ep, color="red", linestyle=":", linewidth=1.0,
               label=f"best val (ep {best_ep})")
    ax.legend(fontsize=8)

    plt.suptitle("Training curves (Bias Taxon AE)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = output_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {out_path}")


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

    parser = argparse.ArgumentParser(description="Train Bias Taxon AE on ImageNet-1k")
    parser.add_argument("--config", type=str, default="")
    # data
    parser.add_argument("--data-root", type=str, default=d.get("data_root", "./data/imagenet"))
    parser.add_argument("--output-dir", type=str,
                        default=o.get("output_dir", "./outputs/bias_taxon_ae_imagenet_r18"))
    parser.add_argument("--image-size", type=int, default=d.get("image_size", 224))
    parser.add_argument("--batch-size", type=int, default=d.get("batch_size", 32))
    parser.add_argument("--num-workers", type=int, default=d.get("num_workers", 8))
    parser.add_argument("--val-split", type=float, default=d.get("val_split", 0.0))
    parser.add_argument("--max-train-samples", type=int, default=d.get("max_train_samples"),
                        help="Limit training set size (None = full dataset)")
    parser.add_argument("--max-val-samples",   type=int, default=d.get("max_val_samples"),
                        help="Limit validation set size (None = full dataset)")
    # model
    parser.add_argument("--resnet-variant", type=str, default=m.get("resnet_variant", "18"))
    parser.add_argument("--stage-taxonomy-layers", type=int, nargs=4,
                        default=m.get("stage_taxonomy_layers", [5, 6, 7, 8]))
    parser.add_argument("--stage-strides", type=int, nargs=4,
                        default=m.get("stage_strides", [1, 2, 2, 2]))
    parser.add_argument("--bias-update-rate", type=float, default=m.get("bias_update_rate", 0.001))
    parser.add_argument("--bias-ema-decay", type=float, default=m.get("bias_ema_decay", 0.99))
    parser.add_argument("--temperature", type=float, default=m.get("temperature", 1.0))
    parser.add_argument("--hard", action="store_true", default=m.get("hard", False))
    # training
    parser.add_argument("--epochs", type=int, default=t.get("epochs", 90))
    parser.add_argument("--learning-rate", type=float, default=t.get("learning_rate", 3e-4))
    parser.add_argument("--weight-decay", type=float, default=t.get("weight_decay", 1e-4))
    parser.add_argument("--warmup-epochs", type=int, default=t.get("warmup_epochs", 3))
    parser.add_argument("--save-every", type=int, default=t.get("save_every", 5))
    parser.add_argument("--seed", type=int, default=t.get("seed", 42))
    parser.add_argument("--max-train-steps", type=int, default=t.get("max_train_steps", 0))
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    bur_str = f"_bur_{args.bias_update_rate:.0e}"
    run_suffix = bur_str
    output_dir  = Path(args.output_dir + run_suffix)
    ckpt_dir    = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ImageNet1kHFLoader provides train/validation loaders via HuggingFace datasets.
    imagenet_loader = ImageNet1kHFLoader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        pin_memory=(device.type == "cuda"),
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    train_loader, val_loader = imagenet_loader.get_loaders()

    _mc = {}
    if args.config:
        with open(args.config) as _f:
            _mc = json.load(_f).get("model", {})

    model = BiasTaxonAutoencoder(
        in_channels=_mc.get("in_channels", 3),
        resnet_variant=args.resnet_variant,
        stage_taxonomy_layers=tuple(args.stage_taxonomy_layers),
        stage_strides=tuple(args.stage_strides),
        stage_blocks=_mc.get("stage_blocks", None),
        bias_update_rate=args.bias_update_rate,
        bias_ema_decay=args.bias_ema_decay,
        kernel_size=_mc.get("kernel_size", 3),
        use_stem=_mc.get("use_stem", True),
        stem_channels=_mc.get("stem_channels", 64),
        stem_stride=_mc.get("stem_stride", 2),
        use_stem_maxpool=_mc.get("use_stem_maxpool", True),
        output_activation=_mc.get("output_activation", "none"),
        temperature=args.temperature,
        hard=args.hard,
        depth_decay=_mc.get("depth_decay", 0.5),
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
    best_val = float("inf")

    if args.resume:
        resume_path = Path(args.resume)
        state = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_val = float(state.get("best_val", float("inf")))
        print(f"Resumed from {resume_path} at epoch={start_epoch}")

    print(
        "Training setup:\n"
        f"  device={device}\n"
        f"  train_size={len(imagenet_loader.trainset)} val_size={len(imagenet_loader.valset)}\n"
        f"  batch_size={args.batch_size} epochs={args.epochs}\n"
        f"  lr={args.learning_rate} wd={args.weight_decay}\n"
        f"  bias_update_rate={args.bias_update_rate}\n"
        f"  stage_taxonomy_layers={tuple(args.stage_taxonomy_layers)}"
    )

    history: dict = {
        "epochs":      [],
        "train_loss":  [],
        "train_recon": [],
        "val_loss":    [],
        "val_recon":   [],
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        num_batches = 0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            (recon,) = model(images)
            loss = F.mse_loss(recon, images)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item())
            num_batches += 1
            global_step += 1

            if batch_idx % 50 == 0:
                avg_loss = running_loss / num_batches
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} step={global_step} "
                    f"lr={lr:.3e} loss={avg_loss:.5f}"
                )

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        train_stats = {
            "loss":  running_loss / max(1, num_batches),
            "recon": running_loss / max(1, num_batches),
        }

        val_stats = run_validation(model=model, loader=val_loader, device=device)

        elapsed = time.time() - epoch_start
        print(
            f"epoch={epoch:03d} time={elapsed:.1f}s "
            f"train_loss={train_stats['loss']:.5f} "
            f"val_loss={val_stats['loss']:.5f}"
        )

        # Print per-stage leaf bias terms
        with torch.no_grad():
            for s_idx, stage in enumerate(model.encoder.taxon_stages):
                bias_vals = stage._node_bias.cpu().tolist()
                ema_vals = stage._ema_load.cpu().tolist()
                bias_str = ", ".join(f"{v:+.4f}" for v in bias_vals)
                ema_str = ", ".join(f"{v:.4f}" for v in ema_vals)
                print(f"  Stage {s_idx+1} leaf biases:  [{bias_str}]")
                print(f"  Stage {s_idx+1} ema loads:    [{ema_str}]")

        history["epochs"].append(epoch)
        history["train_loss"].append(train_stats["loss"])
        history["train_recon"].append(train_stats["recon"])
        history["val_loss"].append(val_stats["loss"])
        history["val_recon"].append(val_stats["recon"])

        if epoch % args.save_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            state = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val,
                "args": vars(args),
                "train_stats": train_stats,
                "val_stats": val_stats,
            }
            torch.save(state, ckpt_path)
            torch.save(state, ckpt_dir / "latest.pt")

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_state = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val": best_val,
                "args": vars(args),
                "train_stats": train_stats,
                "val_stats": val_stats,
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
