#!/usr/bin/env python3
"""Train TopKTaxonAutoencoder on CelebA-HQ with ResNet-18 stage layout.

Uses TopK leaf selection per taxonomy stage.  Dead leaves are revived via
an AuxK reconstruction-error loss.

Loss:
    recon + auxk_weight * auxk_loss
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

from src.model.cnn.taxon.topk_taxon_ae import TopKTaxonAutoencoder
from src.utils.dataloader import CelebAHQLoader


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
    auxk_weight: float,
) -> dict:
    model.eval()
    total_loss = total_recon = total_auxk = total_dead = 0.0
    num_batches = 0

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recon, dead_frac = model(images)
        recon_loss = F.mse_loss(recon, images)
        auxk_loss = model.compute_auxk_loss(images, recon)
        loss = recon_loss + auxk_weight * auxk_loss

        total_loss  += float(loss.item())
        total_recon += float(recon_loss.item())
        total_auxk  += float(auxk_loss.item())
        total_dead  += float(dead_frac.item())
        num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "recon": 0.0, "auxk": 0.0, "dead_frac": 0.0}

    return {
        "loss":      total_loss  / num_batches,
        "recon":     total_recon / num_batches,
        "auxk":      total_auxk  / num_batches,
        "dead_frac": total_dead  / num_batches,
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
    import json as _json

    epochs = history["epochs"]
    if not epochs:
        return

    with open(output_dir / "training_history.json", "w") as _f:
        _json.dump(history, _f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    panels = [
        ("Total loss", "train_loss",      "val_loss"),
        ("Recon loss", "train_recon",     "val_recon"),
        ("AuxK loss",  "train_auxk",      "val_auxk"),
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

    plt.suptitle("Training curves (TopK Taxon AE)", fontsize=13, fontweight="bold")
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

    parser = argparse.ArgumentParser(description="Train TopK Taxon AE on CelebA-HQ")
    parser.add_argument("--config", type=str, default="")
    # data
    parser.add_argument("--data-root", type=str, default=d.get("data_root", "./data/celeba_hq"))
    parser.add_argument("--output-dir", type=str, default=o.get("output_dir", "./outputs/topk_taxon_ae_celeba_hq_r18"))
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

    parser.add_argument("--k-aux", type=int, default=m.get("k_aux", None))
    parser.add_argument("--dead-steps", type=int, default=m.get("dead_steps", 2000))
    parser.add_argument("--temperature", type=float, default=m.get("temperature", 1.0))
    parser.add_argument("--hard", action="store_true", default=m.get("hard", False))
    # training
    parser.add_argument("--epochs", type=int, default=t.get("epochs", 90))
    parser.add_argument("--learning-rate", type=float, default=t.get("learning_rate", 3e-4))
    parser.add_argument("--weight-decay", type=float, default=t.get("weight_decay", 1e-4))
    parser.add_argument("--warmup-epochs", type=int, default=t.get("warmup_epochs", 3))
    parser.add_argument("--auxk-weight", type=float, default=t.get("auxk_weight", 1.0 / 32))
    parser.add_argument("--save-every", type=int, default=t.get("save_every", 5))
    parser.add_argument("--seed", type=int, default=t.get("seed", 42))
    parser.add_argument("--max-train-steps", type=int, default=t.get("max_train_steps", 0))
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    auxk_str = f"_auxk_{args.auxk_weight:.0e}" if args.auxk_weight else ""
    run_suffix = auxk_str
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

    _mc = {}
    if args.config:
        with open(args.config) as _f:
            _mc = json.load(_f).get("model", {})

    model = TopKTaxonAutoencoder(
        in_channels=_mc.get("in_channels", 3),
        resnet_variant=args.resnet_variant,
        stage_taxonomy_layers=tuple(args.stage_taxonomy_layers),
        stage_strides=tuple(args.stage_strides),
        stage_blocks=_mc.get("stage_blocks", None),
        k_aux=args.k_aux,
        dead_steps=args.dead_steps,
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
        f"  train_size={len(celeba_loader.trainset)} val_size={len(celeba_loader.valset)}\n"
        f"  batch_size={args.batch_size} epochs={args.epochs}\n"
        f"  lr={args.learning_rate} wd={args.weight_decay}\n"
        f"  k_aux={args.k_aux} dead_steps={args.dead_steps}\n"
        f"  auxk_weight={args.auxk_weight}\n"
        f"  stage_taxonomy_layers={tuple(args.stage_taxonomy_layers)}"
    )

    history: dict = {
        "epochs":        [],
        "train_loss":    [],
        "train_recon":   [],
        "train_auxk":    [],
        "train_dead":    [],
        "val_loss":      [],
        "val_recon":     [],
        "val_auxk":      [],
        "val_dead":      [],
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()

        running = {k: 0.0 for k in ("loss", "recon", "auxk", "dead")}
        num_batches = 0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            recon, dead_frac = model(images)
            recon_loss = F.mse_loss(recon, images)
            auxk_loss = model.compute_auxk_loss(images, recon)
            loss = recon_loss + args.auxk_weight * auxk_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            running["loss"]  += float(loss.item())
            running["recon"] += float(recon_loss.item())
            running["auxk"]  += float(auxk_loss.item())
            running["dead"]  += float(dead_frac.item())
            num_batches += 1
            global_step += 1

            if batch_idx % 50 == 0:
                avg = {k: v / num_batches for k, v in running.items()}
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} batch={batch_idx}/{len(train_loader)} step={global_step} "
                    f"lr={lr:.3e} loss={avg['loss']:.5f} recon={avg['recon']:.5f} "
                    f"auxk={avg['auxk']:.5f} dead={avg['dead']:.3f}"
                )

            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        train_stats = {k: v / max(1, num_batches) for k, v in running.items()}

        val_stats = run_validation(
            model=model,
            loader=val_loader,
            device=device,
            auxk_weight=args.auxk_weight,
        )

        elapsed = time.time() - epoch_start
        print(
            f"epoch={epoch:03d} time={elapsed:.1f}s "
            f"train_loss={train_stats['loss']:.5f} train_recon={train_stats['recon']:.5f} "
            f"val_loss={val_stats['loss']:.5f} val_recon={val_stats['recon']:.5f} "
            f"dead={val_stats['dead_frac']:.3f}"
        )

        history["epochs"].append(epoch)
        for split, stats in (("train", train_stats), ("val", val_stats)):
            history[f"{split}_loss"].append(stats["loss"])
            history[f"{split}_recon"].append(stats["recon"])
            history[f"{split}_auxk"].append(stats["auxk"])
            history[f"{split}_dead"].append(stats.get("dead", stats.get("dead_frac", 0.0)))

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
