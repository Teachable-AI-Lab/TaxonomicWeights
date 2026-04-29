#!/usr/bin/env python3
"""Train ConvTopKSAEAutoencoder on CelebA-HQ.

Strided-conv encoder/decoder (no ResNet) with a channel-wise TopK bottleneck
and AuxK dead-node revival.

Loss = recon_mse + auxk_weight * auxk
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

from src.model.cnn.conv_topk_sae_ae import ConvTopKSAEAutoencoder
from src.utils.dataloader import CelebAHQLoader


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
def run_validation(model, loader, device, auxk_w) -> dict:
    model.eval()
    tot_loss = tot_recon = tot_auxk = tot_dead = 0.0
    nb = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        recon, dead_frac, _ = model(images)
        recon_loss = F.mse_loss(recon, images)
        auxk_loss = model.compute_auxk_loss(images, recon)
        loss = recon_loss + auxk_w * auxk_loss
        tot_loss += float(loss); tot_recon += float(recon_loss)
        tot_auxk += float(auxk_loss); tot_dead += float(dead_frac); nb += 1
    if nb == 0:
        return {"loss": 0.0, "recon": 0.0, "auxk": 0.0, "dead_frac": 0.0}
    return {"loss": tot_loss/nb, "recon": tot_recon/nb,
            "auxk": tot_auxk/nb, "dead_frac": tot_dead/nb}


def save_recon_preview(model, loader, device, save_path, num_images=8) -> None:
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


def save_training_curves(history, output_dir) -> None:
    if not history["epochs"]:
        return
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    panels = [
        ("Total loss", "train_loss", "val_loss"),
        ("Recon MSE", "train_recon", "val_recon"),
        ("AuxK",      "train_auxk",  "val_auxk"),
    ]
    for ax, (title, tk, vk) in zip(axes, panels):
        ax.plot(history["epochs"], history[tk], label="train")
        ax.plot(history["epochs"], history[vk], label="val", linestyle="--")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle("Conv TopK SAE AE")
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
    p.add_argument("--data-root",   type=str,   default=d.get("data_root", "./data/celeba_hq"))
    p.add_argument("--output-dir",  type=str,   default=o.get("output_dir", "./outputs/conv_topk_sae_ae_celeba_hq"))
    p.add_argument("--image-size",  type=int,   default=d.get("image_size", 256))
    p.add_argument("--batch-size",  type=int,   default=d.get("batch_size", 32))
    p.add_argument("--num-workers", type=int,   default=d.get("num_workers", 8))
    p.add_argument("--val-split",   type=float, default=d.get("val_split", 0.05))
    p.add_argument("--epochs",      type=int,   default=t.get("epochs", 90))
    p.add_argument("--learning-rate",   type=float, default=t.get("learning_rate", 3e-4))
    p.add_argument("--weight-decay",    type=float, default=t.get("weight_decay", 1e-4))
    p.add_argument("--warmup-epochs",   type=int,   default=t.get("warmup_epochs", 3))
    p.add_argument("--auxk-weight",     type=float, default=t.get("auxk_weight", 0.01))
    p.add_argument("--decoder-max-norm", type=float, default=t.get("decoder_max_norm", 1.0))
    p.add_argument("--save-every", type=int,  default=t.get("save_every", 5))
    p.add_argument("--seed",       type=int,  default=t.get("seed", 42))
    p.add_argument("--max-train-steps", type=int, default=t.get("max_train_steps", 0))
    p.add_argument("--resume", type=str, default="")
    return p.parse_args(), m


def main() -> None:
    args, mc = parse_args()
    seed_everything(args.seed)

    k = int(mc.get("topk_k", 64))
    run_suffix = f"_k{k}"
    output_dir = Path(args.output_dir + run_suffix)
    ckpt_dir = output_dir / "checkpoints"
    preview_dir = output_dir / "previews"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
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

    model = ConvTopKSAEAutoencoder(
        in_channels=int(mc.get("in_channels", 3)),
        topk_k=k,
        k_aux=mc.get("k_aux", None),
        dead_steps=int(mc.get("dead_steps", 2000)),
        output_activation=str(mc.get("output_activation", "none")),
        use_batch_topk=bool(mc.get("use_batch_topk", True)),
        warmup_steps=int(mc.get("warmup_steps", 5000)),
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
    print(f"Conv TopK SAE AE: device={device} n_params={n_params:,} k={k} "
          f"latent_ch={model.latent_channels} batch={args.batch_size} "
          f"epochs={args.epochs} output={output_dir}")

    history = {"epochs": [],
               "train_loss": [], "train_recon": [], "train_auxk": [], "train_dead": [],
               "val_loss":   [], "val_recon":   [], "val_auxk":   [], "val_dead":   []}

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = {k: 0.0 for k in ("loss", "recon", "auxk", "dead")}
        nb = 0
        for batch_idx, (images, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon, dead_frac, _ = model(images)
            recon_loss = F.mse_loss(recon, images)
            auxk_loss = model.compute_auxk_loss(images, recon)
            loss = recon_loss + args.auxk_weight * auxk_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.decoder_max_norm > 0:
                with torch.no_grad():
                    for module in model.decoder_net.modules():
                        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and module.weight.requires_grad:
                            w = module.weight
                            norms = w.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
                            scale = norms.clamp(min=args.decoder_max_norm) / args.decoder_max_norm
                            module.weight.div_(scale.view(-1, 1, 1, 1))
            running["loss"] += float(loss); running["recon"] += float(recon_loss)
            running["auxk"] += float(auxk_loss); running["dead"] += float(dead_frac)
            nb += 1; global_step += 1
            if batch_idx % 50 == 0:
                avg = {k: v / nb for k, v in running.items()}
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch={epoch} batch={batch_idx}/{len(train_loader)} step={global_step} "
                      f"lr={lr:.3e} loss={avg['loss']:.5f} recon={avg['recon']:.5f} "
                      f"auxk={avg['auxk']:.5f} dead={avg['dead']:.3f}")
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                break

        train_stats = {k: v / max(1, nb) for k, v in running.items()}
        val_stats = run_validation(model, val_loader, device, args.auxk_weight)
        elapsed = time.time() - epoch_start
        print(f"epoch={epoch:03d} time={elapsed:.1f}s "
              f"train_loss={train_stats['loss']:.5f} val_loss={val_stats['loss']:.5f} "
              f"dead={val_stats['dead_frac']:.3f}")

        history["epochs"].append(epoch)
        for split, stats in (("train", train_stats), ("val", val_stats)):
            history[f"{split}_loss"].append(stats["loss"])
            history[f"{split}_recon"].append(stats["recon"])
            history[f"{split}_auxk"].append(stats["auxk"])
            history[f"{split}_dead"].append(stats.get("dead", stats.get("dead_frac", 0.0)))

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
        save_recon_preview(model, val_loader, device, preview_dir / f"epoch_{epoch:03d}.png")
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    save_training_curves(history, output_dir)


if __name__ == "__main__":
    main()
