#!/usr/bin/env python3
"""Train BottleneckMultiTaxonAutoencoder on CelebA-HQ.

Plain ResNet stages followed by a single multi-hierarchy (softmax+DKL) taxon
stage at the bottleneck.  Loss = recon + dkl_w*dkl + entropy_w*entropy
+ gate_dkl_w*gate_dkl + gate_entropy_w*gate_entropy.

``hard=True`` enables straight-through argmax in the inter-hierarchy gate so
exactly one hierarchy is selected per spatial position.
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
from torchvision import transforms
from torchvision.utils import make_grid, save_image

import sys
ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.taxon.bottleneck_multi_taxon_ae import BottleneckMultiTaxonAutoencoder
from src.utils.dataloader import CelebAHQLoader


def seed_everything(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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
def run_validation(model, loader, device, dw, ew, gdw, gew, hard) -> dict:
    model.eval()
    tot = {k: 0.0 for k in ("loss", "recon", "dkl", "entropy", "gate_dkl", "gate_entropy")}
    nb = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recon, dkl, ent, gdkl, gent = model(images, hard=hard)
        recon_loss = F.mse_loss(recon, images)
        loss = recon_loss + dw*dkl + ew*ent + gdw*gdkl + gew*gent
        tot["loss"] += float(loss.item()); tot["recon"] += float(recon_loss.item())
        tot["dkl"] += float(dkl.item()); tot["entropy"] += float(ent.item())
        tot["gate_dkl"] += float(gdkl.item()); tot["gate_entropy"] += float(gent.item())
        nb += 1
    if nb == 0:
        return {k: 0.0 for k in tot}
    return {k: v/nb for k, v in tot.items()}


def save_recon_preview(model, loader, device, save_path, hard, num_images=8) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:num_images].to(device)
    with torch.no_grad():
        recon, *_ = model(images, hard=hard)
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
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    panels = [
        ("Total loss", "train_loss", "val_loss"),
        ("Recon loss", "train_recon", "val_recon"),
        ("DKL", "train_dkl", "val_dkl"),
        ("Gate DKL", "train_gate_dkl", "val_gate_dkl"),
    ]
    for ax, (title, tk, vk) in zip(axes, panels):
        ax.plot(epochs, history[tk], label="train")
        ax.plot(epochs, history[vk], label="val", linestyle="--")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle("Bottleneck Multi-Taxon AE")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args()
    cfg: dict = {}
    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = json.load(f)
    d = cfg.get("data", {}); m = cfg.get("model", {})
    t = cfg.get("training", {}); o = cfg.get("output", {})

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="")
    p.add_argument("--data-root", type=str, default=d.get("data_root", "./data/celeba_hq"))
    p.add_argument("--output-dir", type=str,
                   default=o.get("output_dir", "./outputs/bottleneck_multi_taxon_ae_celeba_hq"))
    p.add_argument("--image-size", type=int, default=d.get("image_size", 256))
    p.add_argument("--batch-size", type=int, default=d.get("batch_size", 32))
    p.add_argument("--num-workers", type=int, default=d.get("num_workers", 8))
    p.add_argument("--val-split", type=float, default=d.get("val_split", 0.05))
    p.add_argument("--epochs", type=int, default=t.get("epochs", 90))
    p.add_argument("--learning-rate", type=float, default=t.get("learning_rate", 3e-4))
    p.add_argument("--weight-decay", type=float, default=t.get("weight_decay", 1e-4))
    p.add_argument("--warmup-epochs", type=int, default=t.get("warmup_epochs", 3))
    p.add_argument("--dkl-weight", type=float, default=t.get("dkl_weight", 1e-3))
    p.add_argument("--entropy-weight", type=float, default=t.get("entropy_weight", 0.0))
    p.add_argument("--gate-dkl-weight", type=float, default=t.get("gate_dkl_weight", 1e-3))
    p.add_argument("--gate-entropy-weight", type=float, default=t.get("gate_entropy_weight", 0.0))
    p.add_argument("--decoder-max-norm", type=float, default=t.get("decoder_max_norm", 1.0))
    p.add_argument("--save-every", type=int, default=t.get("save_every", 5))
    p.add_argument("--seed", type=int, default=t.get("seed", 42))
    p.add_argument("--max-train-steps", type=int, default=t.get("max_train_steps", 0))
    p.add_argument("--hard", action="store_true", default=m.get("hard", False))
    p.add_argument("--resume", type=str, default="")
    return p.parse_args(), m


def main() -> None:
    args, _mc = parse_args()
    seed_everything(args.seed)

    K = int(_mc.get("n_hierarchies", 4))
    L = int(_mc.get("bottleneck_n_taxonomy_layers", 6))
    gate_tag = "gate1" if bool(_mc.get("hard", False)) else f"gate{K}"
    run_suffix = f"_K{K}_L{L}_{gate_tag}"
    output_dir = Path(args.output_dir + run_suffix)
    ckpt_dir = output_dir / "checkpoints"; preview_dir = output_dir / "previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    celeba_loader = CelebAHQLoader(
        data_root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, image_size=args.image_size,
        val_split=args.val_split, seed=args.seed,
        pin_memory=(device.type == "cuda"), transform=tf,
    )
    train_loader, val_loader = celeba_loader.get_loaders()
    if val_loader is None:
        raise RuntimeError("val_split must be > 0")

    model = BottleneckMultiTaxonAutoencoder(
        in_channels=_mc.get("in_channels", 3),
        plain_stage_channels=tuple(_mc.get("plain_stage_channels", [64, 128, 256])),
        plain_stage_blocks=tuple(_mc.get("plain_stage_blocks", [2, 2, 2])),
        plain_stage_strides=tuple(_mc.get("plain_stage_strides", [1, 2, 2])),
        bottleneck_n_taxonomy_layers=L,
        bottleneck_n_blocks=int(_mc.get("bottleneck_n_blocks", 2)),
        bottleneck_stride=int(_mc.get("bottleneck_stride", 2)),
        n_hierarchies=K,
        kernel_size=int(_mc.get("kernel_size", 3)),
        use_stem=bool(_mc.get("use_stem", True)),
        stem_channels=int(_mc.get("stem_channels", 64)),
        stem_stride=int(_mc.get("stem_stride", 2)),
        use_stem_maxpool=bool(_mc.get("use_stem_maxpool", True)),
        output_activation=str(_mc.get("output_activation", "none")),
        temperature=float(_mc.get("temperature", 1.0)),
        hard=bool(_mc.get("hard", False)),
        depth_decay=float(_mc.get("depth_decay", 0.5)),
        k_leaves=int(_mc.get("k_leaves", 0)),
        use_gate_value=bool(_mc.get("use_gate_value", False)),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, max(1, len(train_loader)),
                                args.epochs, args.warmup_epochs)

    start_epoch = 1; global_step = 0; best_val = float("inf")
    if args.resume:
        state = torch.load(Path(args.resume), map_location="cpu")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        best_val = float(state.get("best_val", float("inf")))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Bottleneck Multi-Taxon AE setup: device={device} n_params={n_params:,} K={K} L={L} "
          f"hard={args.hard} latent_ch={model.encoder.final_channels} batch={args.batch_size} "
          f"epochs={args.epochs} output={output_dir}")

    history = {"epochs": [],
               "train_loss": [], "train_recon": [], "train_dkl": [], "train_entropy": [],
               "train_gate_dkl": [], "train_gate_entropy": [],
               "val_loss": [], "val_recon": [], "val_dkl": [], "val_entropy": [],
               "val_gate_dkl": [], "val_gate_entropy": []}

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = {k: 0.0 for k in ("loss", "recon", "dkl", "entropy", "gate_dkl", "gate_entropy")}
        nb = 0
        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon, dkl, ent, gdkl, gent = model(images, hard=args.hard)
            recon_loss = F.mse_loss(recon, images)
            loss = (recon_loss + args.dkl_weight*dkl + args.entropy_weight*ent
                    + args.gate_dkl_weight*gdkl + args.gate_entropy_weight*gent)
            loss.backward()
            optimizer.step(); scheduler.step()
            if args.decoder_max_norm > 0:
                with torch.no_grad():
                    for module in model.decoder.modules():
                        if isinstance(module, nn.Conv2d) and module.weight.requires_grad:
                            w = module.weight
                            norms = w.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
                            scale = norms.clamp(min=args.decoder_max_norm) / args.decoder_max_norm
                            module.weight.div_(scale.view(-1, 1, 1, 1))
            running["loss"] += float(loss.item()); running["recon"] += float(recon_loss.item())
            running["dkl"] += float(dkl.item()); running["entropy"] += float(ent.item())
            running["gate_dkl"] += float(gdkl.item()); running["gate_entropy"] += float(gent.item())
            nb += 1; global_step += 1
            if batch_idx % 50 == 0:
                avg = {k: v/nb for k, v in running.items()}
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch={epoch} batch={batch_idx}/{len(train_loader)} step={global_step} "
                      f"lr={lr:.3e} loss={avg['loss']:.5f} recon={avg['recon']:.5f} "
                      f"dkl={avg['dkl']:.5f} gate_dkl={avg['gate_dkl']:.5f}")
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        train_stats = {k: v/max(1, nb) for k, v in running.items()}
        val_stats = run_validation(model, val_loader, device,
                                   args.dkl_weight, args.entropy_weight,
                                   args.gate_dkl_weight, args.gate_entropy_weight, args.hard)
        elapsed = time.time() - epoch_start
        print(f"epoch={epoch:03d} time={elapsed:.1f}s "
              f"train_loss={train_stats['loss']:.5f} val_loss={val_stats['loss']:.5f}")

        history["epochs"].append(epoch)
        for split, stats in (("train", train_stats), ("val", val_stats)):
            for k in ("loss", "recon", "dkl", "entropy", "gate_dkl", "gate_entropy"):
                history[f"{split}_{k}"].append(stats[k])

        state = {"epoch": epoch, "global_step": global_step,
                 "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "scheduler_state": scheduler.state_dict(),
                 "best_val": best_val, "args": vars(args),
                 "train_stats": train_stats, "val_stats": val_stats}
        if epoch % args.save_every == 0:
            torch.save(state, ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt")
            torch.save(state, ckpt_dir / "latest.pt")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]; state["best_val"] = best_val
            torch.save(state, ckpt_dir / "best.pt")
        save_recon_preview(model, val_loader, device,
                           preview_dir / f"epoch_{epoch:03d}.png", args.hard)
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    save_training_curves(history, output_dir)


if __name__ == "__main__":
    main()
