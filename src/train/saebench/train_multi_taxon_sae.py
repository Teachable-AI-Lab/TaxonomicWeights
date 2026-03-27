#!/usr/bin/env python3
"""Train MultiTaxonSAE on LLM residual-stream activations using SAEBench conventions.

Supports two data modes:
  1. Streaming via ActivationsStore (--dataset, runs LLM on-the-fly)
  2. Cached activations (--cached-activations-path, loads .pt chunks from disk)

Loss:  MSE(x_hat, x) + dkl_weight * dkl + entropy_weight * entropy
       + gate_dkl_weight * gate_dkl + gate_entropy_weight * gate_entropy
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.saebench.multi_taxon_sae import MultiTaxonSAE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    total_steps = max(1, total_steps)
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ── Cached activation data loader (streaming, one chunk at a time) ─────────
class CachedActivationLoader:
    """Streams pre-cached .pt activation chunks from disk one at a time.

    Instead of loading all chunks into RAM (which can exceed memory for
    large caches), this loader keeps only one chunk resident at a time,
    shuffles within it, and shuffles the chunk order each epoch.
    """

    def __init__(self, cache_dir: str, batch_size: int, device: torch.device,
                 dtype: torch.dtype, seed: int = 42):
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.seed = seed

        self.chunk_files = sorted(glob.glob(os.path.join(cache_dir, "chunk_*.pt")))
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk_*.pt files found in {cache_dir}")

        meta_path = os.path.join(cache_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
            print(f"  Cache: {len(self.chunk_files)} chunks, "
                  f"d_model={self.meta['d_model']}, "
                  f"model={self.meta['model_name']}, "
                  f"hook={self.meta['hook_name']}")
        else:
            self.meta = {}

        # Probe first chunk for shape info
        first = torch.load(self.chunk_files[0], map_location="cpu", weights_only=True)
        self.d_model = first.shape[1]
        self.n_chunks = len(self.chunk_files)
        n_total = first.shape[0] * self.n_chunks  # approximate
        print(f"  Streaming loader: {self.n_chunks} chunks, "
              f"~{n_total:,} vectors of dim {self.d_model} "
              f"(~{n_total * self.d_model * 4 / 1e9:.1f} GB on disk)")
        del first

        self._rng = random.Random(seed)
        self._chunk_order = list(range(self.n_chunks))
        self._rng.shuffle(self._chunk_order)
        self._chunk_idx = 0
        self._current_chunk: torch.Tensor | None = None
        self._pos = 0
        self._load_next_chunk()

    def _load_next_chunk(self) -> None:
        """Load the next chunk from disk and shuffle it."""
        if self._chunk_idx >= self.n_chunks:
            # New epoch: reshuffle chunk order
            self._rng.shuffle(self._chunk_order)
            self._chunk_idx = 0
        file_idx = self._chunk_order[self._chunk_idx]
        self._current_chunk = torch.load(
            self.chunk_files[file_idx], map_location="cpu", weights_only=True
        )
        # Shuffle within chunk
        perm = torch.randperm(self._current_chunk.shape[0])
        self._current_chunk = self._current_chunk[perm]
        self._pos = 0
        self._chunk_idx += 1

    def next_batch(self) -> torch.Tensor:
        if self._current_chunk is None or self._pos + self.batch_size > self._current_chunk.shape[0]:
            self._load_next_chunk()
        batch = self._current_chunk[self._pos : self._pos + self.batch_size]
        self._pos += self.batch_size
        return batch.to(device=self.device, dtype=self.dtype)


def save_training_curves(history: dict, output_dir: Path, title: str) -> None:
    steps = history["step"]
    if not steps:
        return

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(2, 4, figsize=(28, 8))
    panels = [
        ("Total loss", "loss"),
        ("Recon loss (MSE)", "recon"),
        ("DKL penalty", "dkl"),
        ("Entropy penalty", "entropy"),
        ("Gate DKL", "gate_dkl"),
        ("Gate Entropy", "gate_entropy"),
        ("L0 sparsity", "l0"),
        ("Alive features (%)", "alive_pct"),
    ]

    for ax, (title_str, key) in zip(axes.flat, panels):
        if key in history and history[key]:
            ax.plot(steps, history[key], linewidth=1.2)
        ax.set_title(title_str, fontsize=11)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{title} training curves", fontsize=13, fontweight="bold")
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

    parser = argparse.ArgumentParser(description="Train MultiTaxonSAE (SAEBench-compatible)")
    parser.add_argument("--config",               type=str,   default="")
    # data
    parser.add_argument("--model-name",           type=str,   default=d.get("model_name", "pythia-160m-deduped"))
    parser.add_argument("--hook-layer",           type=int,   default=d.get("hook_layer", 8))
    parser.add_argument("--hook-name",            type=str,   default=d.get("hook_name", ""))
    parser.add_argument("--dataset",              type=str,   default=d.get("dataset", "Skylion007/openwebtext"))
    parser.add_argument("--context-length",       type=int,   default=d.get("context_length", 128))
    parser.add_argument("--n-tokens",             type=int,   default=d.get("n_tokens", 500_000_000))
    parser.add_argument("--batch-size",           type=int,   default=d.get("batch_size", 4096))
    parser.add_argument("--dtype",                type=str,   default=d.get("dtype", "float32"))
    parser.add_argument("--cached-activations-path", type=str,
                        default=d.get("cached_activations_path", ""))
    # model
    parser.add_argument("--n-taxonomy-layers",    type=int,   default=m.get("n_taxonomy_layers", 6))
    parser.add_argument("--n-hierarchies",        type=int,   default=m.get("n_hierarchies", 8))
    parser.add_argument("--temperature",          type=float, default=m.get("temperature", 0.5))
    parser.add_argument("--hard",                 action="store_true", default=m.get("hard", False))
    parser.add_argument("--depth-decay",          type=float, default=m.get("depth_decay", 0.5))
    # training
    parser.add_argument("--output-dir",           type=str,   default=o.get("output_dir",
                                                              "./outputs/saebench/multi_taxon_sae"))
    parser.add_argument("--total-steps",          type=int,   default=t.get("total_steps", 30_000))
    parser.add_argument("--learning-rate",        type=float, default=t.get("learning_rate", 3e-4))
    parser.add_argument("--weight-decay",         type=float, default=t.get("weight_decay", 0.0))
    parser.add_argument("--warmup-steps",         type=int,   default=t.get("warmup_steps", 1000))
    parser.add_argument("--dkl-weight",           type=float, default=t.get("dkl_weight", 1e-2))
    parser.add_argument("--entropy-weight",       type=float, default=t.get("entropy_weight", 0.0))
    parser.add_argument("--gate-dkl-weight",      type=float, default=t.get("gate_dkl_weight", 1e-2))
    parser.add_argument("--gate-entropy-weight",  type=float, default=t.get("gate_entropy_weight", 0.0))
    parser.add_argument("--save-every",           type=int,   default=t.get("save_every", 5000))
    parser.add_argument("--log-every",            type=int,   default=t.get("log_every", 100))
    parser.add_argument("--seed",                 type=int,   default=t.get("seed", 42))
    parser.add_argument("--normalize-decoder",    action=argparse.BooleanOptionalAction,
                        default=t.get("normalize_decoder", True))
    parser.add_argument("--resume",               type=str,   default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    use_cache = bool(args.cached_activations_path)

    # ── Output directory ────────────────────────────────────────────────
    dkl_suffix = f"_dkl{args.dkl_weight:.0e}"
    temp_str = f"{args.temperature:g}".replace(".", "p")
    hard_str = "_hard" if args.hard else ""
    suffix = f"_L{args.n_taxonomy_layers}_K{args.n_hierarchies}{dkl_suffix}_t{temp_str}{hard_str}"
    output_dir = Path(args.output_dir + suffix)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    hook_name = args.hook_name or f"blocks.{args.hook_layer}.hook_resid_post"

    # ── Data source ─────────────────────────────────────────────────────
    if use_cache:
        print(f"Loading cached activations from {args.cached_activations_path} ...")
        loader = CachedActivationLoader(
            args.cached_activations_path, args.batch_size, device, dtype, args.seed
        )
        d_model = loader.d_model
        llm = None
    else:
        from transformer_lens import HookedTransformer
        print(f"Loading {args.model_name} on {device} ({args.dtype}) ...")
        t0 = time.time()
        llm = HookedTransformer.from_pretrained(args.model_name, device=str(device), dtype=dtype)
        d_model = llm.cfg.d_model
        print(f"  Model loaded in {time.time() - t0:.1f}s  (d_model={d_model})")

    # ── Build SAE ───────────────────────────────────────────────────────
    sae = MultiTaxonSAE(
        d_in=d_model,
        n_taxonomy_layers=args.n_taxonomy_layers,
        n_hierarchies=args.n_hierarchies,
        model_name=args.model_name,
        hook_layer=args.hook_layer,
        device=device,
        dtype=dtype,
        temperature=args.temperature,
        hard=args.hard,
        depth_decay=args.depth_decay,
        hook_name=hook_name,
    )

    # ── Build ActivationsStore (streaming mode only) ────────────────────
    if not use_cache:
        from sae_lens.training.activations_store import ActivationsStore
        store_batch = max(1, args.batch_size // args.context_length)
        act_store = ActivationsStore(
            model=llm,
            dataset=args.dataset,
            streaming=True,
            hook_name=hook_name,
            hook_head_index=None,
            context_size=args.context_length,
            d_in=d_model,
            n_batches_in_buffer=32,
            total_training_tokens=args.n_tokens,
            store_batch_size_prompts=store_batch,
            train_batch_size_tokens=args.batch_size,
            prepend_bos=True,
            normalize_activations="none",
            device=device,
            dtype=dtype,
            dataset_trust_remote_code=True,
        )

    optimizer = AdamW(sae.parameters(), lr=args.learning_rate,
                      betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.total_steps, args.warmup_steps)

    start_step = 0
    best_loss = float("inf")

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        sae.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_step = int(state.get("global_step", 0))
        best_loss = float(state.get("best_loss", float("inf")))
        print(f"Resumed from {args.resume} at step={start_step}")

    n_params = sum(p.numel() for p in sae.parameters() if p.requires_grad)
    print(
        f"\n{'='*60}\n"
        f"  Training MultiTaxonSAE\n"
        f"{'='*60}\n"
        f"  device      = {device}\n"
        f"  dtype       = {args.dtype}\n"
        f"  n_params    = {n_params:,}\n"
        f"  d_model     = {d_model}\n"
        f"  d_sae       = {sae.cfg.d_sae}\n"
        f"  n_tax_layers= {args.n_taxonomy_layers}\n"
        f"  n_hierarchs = {args.n_hierarchies}\n"
        f"  temperature = {args.temperature}\n"
        f"  dkl_weight  = {args.dkl_weight}\n"
        f"  ent_weight  = {args.entropy_weight}\n"
        f"  gate_dkl_w  = {args.gate_dkl_weight}\n"
        f"  gate_ent_w  = {args.gate_entropy_weight}\n"
        f"  total_steps = {args.total_steps}\n"
        f"  batch_size  = {args.batch_size}\n"
        f"  data_source = {'CACHED' if use_cache else 'streaming'}\n"
        f"  output_dir  = {output_dir}\n"
        f"{'='*60}"
    )

    keys = ["step", "loss", "recon", "dkl", "entropy", "gate_dkl", "gate_entropy",
            "l0", "alive_pct"]
    history: dict = {k: [] for k in keys}
    ckpt_args = sae.get_checkpoint_args()

    sae.train()
    running = {"loss": 0.0, "recon": 0.0, "dkl": 0.0, "entropy": 0.0,
               "gate_dkl": 0.0, "gate_entropy": 0.0, "l0": 0.0}
    alive_tracker = torch.zeros(sae.cfg.d_sae, device=device)
    n_avg = 0
    t_start = time.time()

    pbar = tqdm(range(start_step, args.total_steps), desc="Training",
                initial=start_step, total=args.total_steps, dynamic_ncols=True,
                file=sys.stdout, mininterval=30)

    for step in pbar:
        # ── Get batch ───────────────────────────────────────────────
        if use_cache:
            x = loader.next_batch()
        else:
            x = act_store.next_batch().to(device=device, dtype=dtype)

        optimizer.zero_grad(set_to_none=True)

        x_hat, info = sae.forward_with_loss(x)
        recon_loss = F.mse_loss(x_hat, x)
        dkl = info["dkl"]
        entropy = info["entropy"]
        gate_dkl = info.get("gate_dkl", torch.tensor(0.0))
        gate_entropy = info.get("gate_entropy", torch.tensor(0.0))

        loss = (recon_loss
                + args.dkl_weight * dkl
                + args.entropy_weight * entropy
                + args.gate_dkl_weight * gate_dkl
                + args.gate_entropy_weight * gate_entropy)

        loss.backward()
        optimizer.step()
        scheduler.step()

        if args.normalize_decoder:
            sae.normalize_decoder()

        # ── Track metrics ───────────────────────────────────────────
        with torch.no_grad():
            z = sae.encode(x)
            l0 = (z > 0).float().sum(dim=-1).mean().item()
            alive_tracker += (z > 0).any(dim=0).float()

        running["loss"] += loss.item()
        running["recon"] += recon_loss.item()
        running["dkl"] += dkl.item()
        running["entropy"] += entropy.item()
        running["gate_dkl"] += gate_dkl.item()
        running["gate_entropy"] += gate_entropy.item()
        running["l0"] += l0
        n_avg += 1

        # ── Progress bar update ─────────────────────────────────────
        pbar.set_postfix_str(
            f"loss={loss.item():.4f} recon={recon_loss.item():.4f} "
            f"dkl={dkl.item():.4f} g_dkl={gate_dkl.item():.4f} L0={l0:.0f}",
            refresh=False,
        )

        # ── Periodic logging ────────────────────────────────────────
        if (step + 1) % args.log_every == 0:
            avg = {k: v / max(1, n_avg) for k, v in running.items()}
            alive_pct = (alive_tracker > 0).float().mean().item() * 100
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            steps_done = step + 1 - start_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            eta_sec = (args.total_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0

            tqdm.write(
                f"[step {step+1:>6d}/{args.total_steps}] "
                f"lr={lr:.2e}  loss={avg['loss']:.5f}  "
                f"recon={avg['recon']:.5f}  dkl={avg['dkl']:.5f}  "
                f"g_dkl={avg['gate_dkl']:.5f}  g_ent={avg['gate_entropy']:.5f}  "
                f"L0={avg['l0']:.1f}  alive={alive_pct:.1f}%  "
                f"speed={steps_per_sec:.1f} step/s  "
                f"ETA={eta_sec/60:.0f}min",
                file=sys.stdout,
            )

            for k in ["loss", "recon", "dkl", "entropy", "gate_dkl", "gate_entropy", "l0"]:
                history[k].append(avg[k])
            history["alive_pct"].append(alive_pct)
            history["step"].append(step + 1)

            if avg["loss"] < best_loss:
                best_loss = avg["loss"]
                torch.save({
                    "global_step": step + 1,
                    "model_state": sae.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "args": ckpt_args,
                }, ckpt_dir / "best.pt")
                tqdm.write(f"  ★ New best loss: {best_loss:.6f}", file=sys.stdout)

            running = {k: 0.0 for k in running}
            alive_tracker.zero_()
            n_avg = 0

        # ── Periodic checkpoint ─────────────────────────────────────
        if (step + 1) % args.save_every == 0:
            torch.save({
                "global_step": step + 1,
                "model_state": sae.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": ckpt_args,
            }, ckpt_dir / f"step_{step+1:06d}.pt")
            torch.save({
                "global_step": step + 1,
                "model_state": sae.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": ckpt_args,
            }, ckpt_dir / "latest.pt")
            tqdm.write(f"  Checkpoint saved: step_{step+1:06d}.pt", file=sys.stdout)

    pbar.close()

    # ── Final save ──────────────────────────────────────────────────────
    torch.save({
        "global_step": args.total_steps,
        "model_state": sae.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_loss": best_loss,
        "args": ckpt_args,
    }, ckpt_dir / "final.pt")

    save_training_curves(history, output_dir, "MultiTaxonSAE")
    total_time = time.time() - t_start
    print(
        f"\n{'='*60}\n"
        f"  Training complete!\n"
        f"  Best loss: {best_loss:.6f}\n"
        f"  Total time: {total_time/3600:.1f}h ({total_time/60:.0f}min)\n"
        f"  Checkpoints: {ckpt_dir}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
