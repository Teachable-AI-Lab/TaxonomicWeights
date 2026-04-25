#!/usr/bin/env python3
"""Train BottleneckTopKMultiTaxonAutoencoder on ImageNet-1k.

Plain ResNet stages followed by a single TopK multi-taxon stage at the
bottleneck.  With ``k_leaves=1`` and ``use_inter_hierarchy_gate=False`` each
spatial location's latent has exactly ``K * L`` non-zero channels — one
depth-L taxonomic path per hierarchy.

Loss:
    recon + auxk_weight * auxk_loss + dkl_weight * dkl
  (optional matryoshka prefix loss in lieu of plain recon)
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

from src.model.cnn.taxon.bottleneck_topk_multi_taxon_ae import (
    BottleneckTopKMultiTaxonAutoencoder,
)
from src.utils.dataloader import TinyImageNetLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs) -> LambdaLR:
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def run_validation(model, loader, device, auxk_weight, dkl_weight=0.0) -> dict:
    model.eval()
    total_loss = total_recon = total_auxk = total_dead = total_dkl = 0.0
    num_batches = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        if dkl_weight > 0:
            recon, dead_frac, details = model(images, return_details=True)
            dkl = float(details["encoder"]["dkl"].item())
        else:
            recon, dead_frac = model(images)
            dkl = 0.0
        recon_loss = F.mse_loss(recon, images)
        auxk_loss = model.compute_auxk_loss(images, recon)
        loss = recon_loss + auxk_weight * auxk_loss + dkl_weight * dkl
        total_loss += float(loss.item())
        total_recon += float(recon_loss.item())
        total_auxk += float(auxk_loss.item())
        total_dead += float(dead_frac.item())
        total_dkl += dkl
        num_batches += 1
    if num_batches == 0:
        return {"loss": 0.0, "recon": 0.0, "auxk": 0.0, "dead_frac": 0.0, "dkl": 0.0}
    return {
        "loss": total_loss / num_batches,
        "recon": total_recon / num_batches,
        "auxk": total_auxk / num_batches,
        "dead_frac": total_dead / num_batches,
        "dkl": total_dkl / num_batches,
    }


def save_recon_preview(model, loader, device, save_path, num_images=8) -> None:
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


def save_training_curves(history, output_dir) -> None:
    epochs = history["epochs"]
    if not epochs:
        return
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("Total loss", "train_loss", "val_loss"),
        ("Recon loss", "train_recon", "val_recon"),
        ("AuxK loss", "train_auxk", "val_auxk"),
    ]
    for ax, (title, train_key, val_key) in zip(axes, panels):
        ax.plot(epochs, history[train_key], label="train", linewidth=1.5)
        ax.plot(epochs, history[val_key], label="val", linewidth=1.5, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Training curves (Bottleneck TopK Multi-Taxon AE — ImageNet)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


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

    p = argparse.ArgumentParser(description="Train Bottleneck TopK Multi-Taxon AE on ImageNet")
    p.add_argument("--config", type=str, default="")
    # data
    p.add_argument("--data-root", type=str, default=d.get("data_root", "./data/imagenet"))
    p.add_argument("--output-dir", type=str,
                   default=o.get("output_dir", "./outputs/bottleneck_topk_multi_taxon_ae_imagenet"))
    p.add_argument("--image-size", type=int, default=d.get("image_size", 224))
    p.add_argument("--batch-size", type=int, default=d.get("batch_size", 32))
    p.add_argument("--num-workers", type=int, default=d.get("num_workers", 8))
    p.add_argument("--max-train-samples", type=int, default=d.get("max_train_samples", None))
    p.add_argument("--max-val-samples", type=int, default=d.get("max_val_samples", None))
    # training
    p.add_argument("--epochs", type=int, default=t.get("epochs", 90))
    p.add_argument("--learning-rate", type=float, default=t.get("learning_rate", 3e-4))
    p.add_argument("--weight-decay", type=float, default=t.get("weight_decay", 1e-4))
    p.add_argument("--warmup-epochs", type=int, default=t.get("warmup_epochs", 3))
    p.add_argument("--auxk-weight", type=float, default=t.get("auxk_weight", 0.01))
    p.add_argument("--dkl-weight", type=float, default=t.get("dkl_weight", 0.0))
    p.add_argument("--decoder-max-norm", type=float, default=t.get("decoder_max_norm", 1.0))
    p.add_argument("--save-every", type=int, default=t.get("save_every", 5))
    p.add_argument("--seed", type=int, default=t.get("seed", 42))
    p.add_argument("--max-train-steps", type=int, default=t.get("max_train_steps", 0))
    p.add_argument("--matryoshka", action="store_true", default=t.get("matryoshka", False))
    p.add_argument("--matryoshka-weight", type=float, default=t.get("matryoshka_weight", 1.0))
    p.add_argument("--weighted-matryoshka", action="store_true", default=t.get("weighted_matryoshka", False))
    p.add_argument("--resume", type=str, default="")
    return p.parse_args(), m


def main() -> None:
    args, _mc = parse_args()
    seed_everything(args.seed)

    K = int(_mc.get("n_hierarchies", 4))
    L = int(_mc.get("bottleneck_n_taxonomy_layers", 6))
    gate_k_cfg = int(_mc.get("gate_k", K))
    use_gate = bool(_mc.get("use_inter_hierarchy_gate", False))
    gate_tag = f"_gate{gate_k_cfg}" if (use_gate and gate_k_cfg < K) else ""
    run_suffix = f"_K{K}_L{L}{gate_tag}"
    output_dir = Path(args.output_dir + run_suffix)
    ckpt_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ImageNet1kHFLoader provides train/validation loaders via HuggingFace datasets.
    imagenet_loader = TinyImageNetLoader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        pin_memory=(device.type == "cuda"),
    )
    train_loader, val_loader = imagenet_loader.get_loaders()

    model = BottleneckTopKMultiTaxonAutoencoder(
        in_channels=_mc.get("in_channels", 3),
        plain_stage_channels=tuple(_mc.get("plain_stage_channels", [64, 128, 256])),
        plain_stage_blocks=tuple(_mc.get("plain_stage_blocks", [2, 2, 2])),
        plain_stage_strides=tuple(_mc.get("plain_stage_strides", [1, 2, 2])),
        bottleneck_n_taxonomy_layers=L,
        bottleneck_n_blocks=int(_mc.get("bottleneck_n_blocks", 2)),
        bottleneck_stride=int(_mc.get("bottleneck_stride", 2)),
        n_hierarchies=K,
        topk_k_multiplier=float(_mc.get("topk_k_multiplier", 1.0)),
        k_aux=_mc.get("k_aux", None),
        dead_steps=int(_mc.get("dead_steps", 2000)),
        kernel_size=int(_mc.get("kernel_size", 3)),
        use_stem=bool(_mc.get("use_stem", True)),
        stem_channels=int(_mc.get("stem_channels", 64)),
        stem_stride=int(_mc.get("stem_stride", 2)),
        use_stem_maxpool=bool(_mc.get("use_stem_maxpool", True)),
        output_activation=str(_mc.get("output_activation", "none")),
        temperature=float(_mc.get("temperature", 1.0)),
        hard=bool(_mc.get("hard", False)),
        depth_decay=float(_mc.get("depth_decay", 0.5)),
        use_batch_topk=bool(_mc.get("use_batch_topk", True)),
        warmup_steps=int(_mc.get("warmup_steps", 0)),
        k_leaves=int(_mc.get("k_leaves", 1)),
        use_gate_value=bool(_mc.get("use_gate_value", False)),
        use_inter_hierarchy_gate=bool(_mc.get("use_inter_hierarchy_gate", False)),
        gate_k=int(_mc.get("gate_k", K)),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, max(1, len(train_loader)), args.epochs, args.warmup_epochs)

    start_epoch = 1
    global_step = 0
    best_val = float("inf")

    if args.resume:
        state = torch.load(Path(args.resume), map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_val = float(state.get("best_val", float("inf")))
        print(f"Resumed from {args.resume} at epoch={start_epoch}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "Bottleneck TopK Multi-Taxon AE setup (ImageNet):\n"
        f"  device={device}  n_params={n_params:,}\n"
        f"  K={K}  L={L}  k_leaves={int(_mc.get('k_leaves', 1))}\n"
        f"  use_inter_hierarchy_gate={bool(_mc.get('use_inter_hierarchy_gate', False))}\n"
        f"  train_size={len(imagenet_loader.trainset)} val_size={len(imagenet_loader.valset)}\n"
        f"  batch_size={args.batch_size}  epochs={args.epochs}\n"
        f"  matryoshka={args.matryoshka}  weighted={args.weighted_matryoshka}\n"
        f"  output_dir={output_dir}"
    )

    history = {
        "epochs": [],
        "train_loss": [], "train_recon": [], "train_auxk": [], "train_dead": [],
        "val_loss": [], "val_recon": [], "val_auxk": [], "val_dead": [],
    }

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = {k: 0.0 for k in ("loss", "recon", "auxk", "dead")}
        num_batches = 0

        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if args.matryoshka:
                prefix_recons, enc_details = model.forward_matryoshka(images)
                recon = prefix_recons[-1]
                if args.weighted_matryoshka:
                    n_d = len(prefix_recons)
                    weights = [2 ** d for d in range(n_d)]
                    w_sum = sum(weights)
                    mat_loss = sum(
                        w * F.mse_loss(r, images) / w_sum
                        for w, r in zip(weights, prefix_recons)
                    )
                else:
                    mat_loss = sum(F.mse_loss(r, images) for r in prefix_recons) / len(prefix_recons)
                recon_loss = args.matryoshka_weight * mat_loss
                dead_frac = enc_details["dead_frac"]
                auxk_loss = model.compute_auxk_loss(images, recon)
                dkl = enc_details["dkl"] if args.dkl_weight > 0 else images.new_zeros(())
            else:
                if args.dkl_weight > 0:
                    recon, dead_frac, details = model(images, return_details=True)
                    dkl = details["encoder"]["dkl"]
                else:
                    recon, dead_frac = model(images)
                    dkl = images.new_zeros(())
                recon_loss = F.mse_loss(recon, images)
                auxk_loss = model.compute_auxk_loss(images, recon)

            loss = recon_loss + args.auxk_weight * auxk_loss + args.dkl_weight * dkl
            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.decoder_max_norm > 0:
                with torch.no_grad():
                    for module in model.decoder.modules():
                        if isinstance(module, nn.Conv2d) and module.weight.requires_grad:
                            w = module.weight
                            norms = w.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
                            scale = norms.clamp(min=args.decoder_max_norm) / args.decoder_max_norm
                            module.weight.div_(scale.view(-1, 1, 1, 1))

            running["loss"] += float(loss.item())
            running["recon"] += float(recon_loss.item())
            running["auxk"] += float(auxk_loss.item())
            running["dead"] += float(dead_frac.item())
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
        val_stats = run_validation(model, val_loader, device,
                                   auxk_weight=args.auxk_weight, dkl_weight=args.dkl_weight)

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
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val": best_val,
                    "args": vars(args),
                    "train_stats": train_stats,
                    "val_stats": val_stats,
                },
                ckpt_dir / "best.pt",
            )

        save_recon_preview(model, val_loader, device,
                           preview_dir / f"epoch_{epoch:03d}.png", num_images=8)

        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            print(f"Reached max_train_steps={args.max_train_steps}; stopping early.")
            break

    save_training_curves(history, output_dir)


if __name__ == "__main__":
    main()
