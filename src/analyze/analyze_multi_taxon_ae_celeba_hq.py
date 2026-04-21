"""Comprehensive analysis script for the MultiTaxon Autoencoder on CelebA-HQ.

All analyses from the regular taxon script are replicated here, with each
per-hierarchy analysis organised into subdirectories:

    <analysis_type>/stage{S}/          – stage-level outputs
    <analysis_type>/stage{S}/hier{K}/  – per-hierarchy outputs within a stage

Analyses
--------
A.  Per-(stage, hierarchy) — mirrors every analysis in analyze_celeba_hq_ae.py
    A1.  Stage filter visualisation
    A2.  Taxonomy regularisation distributions (entropy / DKL over batches)
    A3.  Path-probability plots (one image, every patch)
    A4.  Taxonomy tree activations (binary tree per image)
    A5.  Parse-tree analysis (selected path, JSON + figure)
    A6.  Hierarchical activation maps (depth-by-depth gated activations)
    A7.  Binary split maps (conditional left/right per sibling pair)
    A8.  Stage activation maps (per-image feature map grids)

B.  Full-model / global
    B1.  Training curves (6-panel)
    B2.  Reconstruction quality metrics (MSE / MAE histograms)
    B3.  Multiple reconstruction comparisons
    B4.  Latent space sparsity & norm statistics
    B5.  Partonomy sparsity suite (Jaccard, selectivity, ablation,
         cross-stage, clustering)

C.  Multi-hierarchy extras (unique to this script)
    C1.  Inter-hierarchy gate distributions (per stage)
    C2.  Cross-hierarchy cosine-similarity matrices (per stage)
    C3.  Per-stage regularisation bar charts
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid, save_image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.taxon.multi_taxon_ae import MultiTaxonAutoencoder
from src.model.cnn.taxon.topk_multi_taxon_ae import TopKMultiTaxonAutoencoder
from src.model.cnn.taxon.bias_multi_taxon_ae import BiasMultiTaxonAutoencoder
from src.model.cnn.taxon.encoder import TaxonResNetStage
from src.utils.dataloader import CelebAHQLoader, CIFAR10Loader, ImageNet1kHFLoader


def _detect_multi_taxon_variant(state_dict: dict) -> str:
    """Auto-detect multi-taxon model variant from state_dict keys."""
    for k in state_dict:
        if '_steps_since_active' in k:
            return 'topk'
        if '_bias' in k and 'multi_taxon_stages' in k:
            return 'bias'
    return 'vanilla'


def _has_vanilla_taxonomy(model) -> bool:
    """Return True if model hierarchies support vanilla taxonomy methods."""
    hier = model.encoder.multi_taxon_stages[0].hierarchies[0]
    return hasattr(hier, '_taxon_logits_per_depth')


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _to_display(tensor: torch.Tensor, normalized: bool = True) -> np.ndarray:
    t = tensor.detach().cpu()
    if normalized:
        t = t * 0.5 + 0.5
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()


def load_model(
    checkpoint_path: str,
    device: torch.device,
    config: dict,
) -> Tuple[MultiTaxonAutoencoder, dict]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    print(f"Loading checkpoint: {checkpoint_path} ({os.path.getsize(checkpoint_path)/1e6:.1f} MB)")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mc = dict(config.get("model", {}))
    # Auto-detect n_hierarchies from state_dict to avoid mismatch with config defaults
    state = checkpoint.get("model_state", checkpoint)
    hier_keys = [k for k in state if "multi_taxon_stages.0.hierarchies." in k]
    if hier_keys:
        detected_n = max(int(k.split(".hierarchies.")[1].split(".")[0]) for k in hier_keys) + 1
        mc["n_hierarchies"] = detected_n

    variant = mc.get("model_variant", _detect_multi_taxon_variant(state))
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}

    # Infer skip-rank from state when config does not specify it.
    inferred_skip_rank = 0
    if "skip_down.weight" in state and state["skip_down.weight"].ndim == 4:
        inferred_skip_rank = int(state["skip_down.weight"].shape[0])

    common_kw = dict(
        in_channels=mc.get("in_channels", 3),
        resnet_variant=mc.get("resnet_variant", "18"),
        stage_taxonomy_layers=tuple(mc.get("stage_taxonomy_layers", [5, 6, 7, 8])),
        stage_strides=tuple(mc.get("stage_strides", [1, 2, 2, 2])),
        stage_blocks=mc.get("stage_blocks", None),
        n_hierarchies=mc.get("n_hierarchies", 3),
        kernel_size=mc.get("kernel_size", 3),
        use_stem=mc.get("use_stem", True),
        stem_channels=mc.get("stem_channels", 64),
        stem_stride=mc.get("stem_stride", 2),
        use_stem_maxpool=mc.get("use_stem_maxpool", True),
        output_activation=mc.get("output_activation", "none"),
        depth_decay=mc.get("depth_decay", 0.5),
    )

    if variant == 'topk':
        model = TopKMultiTaxonAutoencoder(
            **common_kw,
            topk_k_multiplier=mc.get("topk_k_multiplier", 1.0),
            k_aux=mc.get("k_aux", None),
            dead_steps=mc.get("dead_steps", 2000),
            gate_k=mc.get("gate_k", 2),
            temperature=mc.get("temperature", 1.0),
            hard=mc.get("hard", False),
            use_batch_topk=mc.get("use_batch_topk", ckpt_args.get("use_batch_topk", True)),
            warmup_steps=mc.get("warmup_steps", ckpt_args.get("warmup_steps", 0)),
            skip_rank=mc.get("skip_rank", ckpt_args.get("skip_rank", inferred_skip_rank)),
            k_leaves=mc.get("k_leaves", ckpt_args.get("k_leaves", 0)),
        )
    elif variant == 'bias':
        model = BiasMultiTaxonAutoencoder(
            **common_kw,
            bias_update_rate=mc.get("bias_update_rate", 0.001),
            bias_ema_decay=mc.get("bias_ema_decay", 0.99),
            gate_k=mc.get("gate_k", 1),
            temperature=mc.get("temperature", 1.0),
            hard=mc.get("hard", False),
        )
    else:
        model = MultiTaxonAutoencoder(
            **common_kw,
            temperature=mc.get("temperature", 1.0),
            hard=mc.get("hard", False),
        )

    # Keep analysis backward/forward compatible across v1/v2/v3 checkpoints.
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    if "val_stats" in checkpoint:
        vs = checkpoint["val_stats"]
        parts = []
        for label, key in [("val_loss", "loss"), ("val_recon", "recon"), ("val_gate_dkl", "gate_dkl")]:
            v = vs.get(key)
            if isinstance(v, (int, float)):
                parts.append(f"{label}={v:.6f}")
        if parts:
            print("  " + "  ".join(parts))
    return model, checkpoint


@torch.no_grad()
def _extract_stage_inputs(
    model: MultiTaxonAutoencoder,
    images: torch.Tensor,
    device: torch.device,
) -> List[torch.Tensor]:
    """Return the feature-map tensor fed into each MultiTaxonResNetStage.

    Returns a list of length n_stages; element s has shape [B, C_s, H_s, W_s].
    """
    model.eval()
    images = images.to(device)
    x = model.encoder.stem(images)
    stage_inputs: List[torch.Tensor] = []
    for stage in model.encoder.multi_taxon_stages:
        stage_inputs.append(x.clone())
        x, _, _ = stage(x)
    return stage_inputs


@torch.no_grad()
def _extract_hierarchy_depths(
    hierarchy: TaxonResNetStage,
    x_s: torch.Tensor,
) -> List[dict]:
    """Extract depth-resolved activations for one hierarchy on features x_s.

    Returns list of dicts (one per taxonomy depth) with keys:
        depth, n_nodes, gated_acts, raw_logits, probs
    All tensors have shape [B, n_nodes, H, W] as CPU float.
    """
    logits_per_depth = hierarchy._taxon_logits_per_depth(x_s)
    depths = []
    prev_logp: Optional[torch.Tensor] = None
    for d_idx, logits in enumerate(logits_per_depth):
        log_cond = hierarchy._pairwise_log_softmax(logits)
        logp = (
            log_cond if prev_logp is None
            else log_cond + prev_logp.repeat_interleave(2, dim=1)
        )
        prob = logp.exp()
        out = logits * prob
        depths.append({
            "depth": d_idx + 1,
            "n_nodes": logits.shape[1],
            "gated_acts":  out.cpu().float(),
            "raw_logits":  logits.cpu().float(),
            "probs":       prob.cpu().float(),
        })
        prev_logp = logp
    return depths


# ══════════════════════════════════════════════════════════════════════════════
# B1. Training curves
# ══════════════════════════════════════════════════════════════════════════════

def save_training_curves(history: dict, save_dir: str) -> None:
    epochs = history.get("epochs", [])
    if not epochs:
        print("  No training history found, skipping curves.")
        return
    panels = [
        ("Total loss",           "train_loss",         "val_loss"),
        ("Recon loss",           "train_recon",        "val_recon"),
        ("DKL penalty",          "train_dkl",          "val_dkl"),
        ("Entropy penalty",      "train_entropy",      "val_entropy"),
        ("Gate DKL penalty",     "train_gate_dkl",     "val_gate_dkl"),
        ("Gate Entropy penalty", "train_gate_entropy", "val_gate_entropy"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(30, 4))
    for ax, (title, tk, vk) in zip(axes, panels):
        if tk in history and vk in history:
            ax.plot(epochs, history[tk], label="train", linewidth=1.5)
            ax.plot(epochs, history[vk], label="val",   linewidth=1.5, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Training curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# A1. Stage filter visualisation
# ══════════════════════════════════════════════════════════════════════════════

def visualize_stage_filters(
    model: MultiTaxonAutoencoder,
    save_dir: str,
    n_cols: int = 8,
) -> None:
    """Save filter grids for the first residual block of each (stage, hierarchy)."""
    base_dir = os.path.join(save_dir, "stage_filters")
    for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
        for k, hierarchy in enumerate(mt_stage.hierarchies):
            out_dir = os.path.join(base_dir, f"stage{s_idx}", f"hier{k}")
            os.makedirs(out_dir, exist_ok=True)
            first_block = hierarchy.blocks[0]
            conv = next(
                (m for m in first_block.main.modules() if isinstance(m, nn.Conv2d)), None
            )
            if conv is None:
                print(f"  Stage {s_idx} Hier {k}: no Conv2d found, skipping.")
                continue
            w = conv.weight.detach().cpu().numpy()      # (out, in, kk, kk)
            out_ch, in_ch, kk, _ = w.shape
            w_disp = w.mean(axis=1)
            w_min = w_disp.min(axis=(1, 2), keepdims=True)
            w_max = w_disp.max(axis=(1, 2), keepdims=True)
            w_norm = (w_disp - w_min) / (w_max - w_min + 1e-8)
            n_show = min(out_ch, n_cols * 8)
            n_rows = max(1, (n_show + n_cols - 1) // n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
            axes = np.array(axes).flatten()
            for i in range(n_show):
                axes[i].imshow(w_norm[i], cmap="viridis")
                axes[i].axis("off")
            for ax in axes[n_show:]:
                ax.axis("off")
            plt.suptitle(
                f"Stage {s_idx} Hier {k} — first-block conv1 filters "
                f"({n_show}/{out_ch}, avg {in_ch} in-ch, {kk}\xd7{kk})",
                fontsize=10,
            )
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"stage{s_idx}_hier{k}_filters.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Stage {s_idx} Hier {k}: {n_show}/{out_ch} filters -> {out_path}")
    print(f"Stage filters saved to {base_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# A1b. Filter similarity analysis
# ══════════════════════════════════════════════════════════════════════════════

def _normalized_filter_matrix(conv: nn.Conv2d) -> np.ndarray:
    """Return L2-normalised filter matrix (n_out, in_ch*kH*kW)."""
    w = conv.weight.detach().cpu().float()
    W = w.view(w.shape[0], -1).numpy()
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return W / (norms + 1e-8)


def _save_simmat(
    sim: np.ndarray,
    title: str,
    path: str,
    leaf_start: int = -1,
) -> None:
    """Save a cosine-similarity heatmap, optionally marking the leaf boundary."""
    n = len(sim)
    sz = max(3.5, min(n * 0.08 + 1.0, 14.0))
    fig, ax = plt.subplots(figsize=(sz, sz))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if 0 < leaf_start < n:
        ax.axhline(leaf_start - 0.5, color="lime", linewidth=1.5, label="leaf start")
        ax.axvline(leaf_start - 0.5, color="lime", linewidth=1.5)
        ax.legend(fontsize=7, loc="upper right")
    ax.set_title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_filter_similarity(model: MultiTaxonAutoencoder, save_dir: str) -> None:
    """Pairwise cosine-similarity of all and leaf filters per (stage, hierarchy),
    plus cross-hierarchy comparisons.

    Per (stage, hierarchy):
      filter_similarity/stage{S}/hier{K}/block{B}_conv{C}_simmat.png
      filter_similarity/stage{S}/hier{K}/leaf_simmat.png
      filter_similarity/stage{S}/hier{K}/cross_block_simmat.png
      filter_similarity/stage{S}/hier{K}/output_vs_leaf_hist.png
    Per stage (cross-hierarchy):
      filter_similarity/stage{S}/cross_hier_leaf_simmat.png
      filter_similarity/stage{S}/cross_hier_leaf_diversity.png
    Summary:
      filter_similarity/summary_mean_sim.png
      filter_similarity/filter_similarity_stats.npz
    """
    base_dir = os.path.join(save_dir, "filter_similarity")
    os.makedirs(base_dir, exist_ok=True)

    K = model.n_hierarchies
    n_stages = len(model.encoder.multi_taxon_stages)
    all_stage_stats: Dict[int, Dict[int, dict]] = {}

    for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
        stage_dir = os.path.join(base_dir, f"stage{s_idx}")
        os.makedirs(stage_dir, exist_ok=True)

        n_leaf = mt_stage.hierarchies[0].layer_channels[-1]
        total  = mt_stage.hierarchies[0].total_out_channels
        L      = mt_stage.hierarchies[0].n_taxonomy_layers

        hier_leaf_Ws: List[np.ndarray] = []  # one per hierarchy, shape (n_leaf, dim)
        hier_stats: Dict[int, dict] = {}

        for k, hierarchy in enumerate(mt_stage.hierarchies):
            hier_dir = os.path.join(stage_dir, f"hier{k}")
            os.makedirs(hier_dir, exist_ok=True)

            # ── per-block, per-Conv2d sim matrices ─────────────────────────
            output_convs: List[np.ndarray] = []
            for b_idx, block in enumerate(hierarchy.blocks):
                convs_in_block = [(name, m) for name, m in block.main.named_modules()
                                  if isinstance(m, nn.Conv2d)]
                for c_name, conv in convs_in_block:
                    W = _normalized_filter_matrix(conv)
                    sim = W @ W.T
                    leaf_start = total - n_leaf if W.shape[0] == total else -1
                    _save_simmat(
                        sim,
                        title=(f"Stage {s_idx} Hier {k} block{b_idx} conv{c_name}\n"
                               f"({W.shape[0]} filters, {W.shape[1]} dims)"),
                        path=os.path.join(hier_dir, f"block{b_idx}_conv{c_name}_simmat.png"),
                        leaf_start=leaf_start,
                    )
                if convs_in_block:
                    output_convs.append(_normalized_filter_matrix(convs_in_block[-1][1]))

            # ── cross-block ─────────────────────────────────────────────────
            if len(output_convs) >= 2 and all(W.shape == output_convs[0].shape for W in output_convs):
                n_blk = len(output_convs)
                fig, axes = plt.subplots(n_blk, n_blk,
                                         figsize=(4 * n_blk, 4 * n_blk),
                                         squeeze=False)
                for bi in range(n_blk):
                    for bj in range(n_blk):
                        cross = output_convs[bi] @ output_convs[bj].T
                        axes[bi, bj].imshow(cross, cmap="RdBu_r", vmin=-1, vmax=1,
                                            interpolation="nearest")
                        axes[bi, bj].set_title(f"b{bi} vs b{bj}", fontsize=8)
                        axes[bi, bj].set_xticks([]); axes[bi, bj].set_yticks([])
                plt.suptitle(f"Stage {s_idx} Hier {k}: cross-block output-conv cosine sim",
                             fontsize=10, fontweight="bold")
                plt.tight_layout()
                plt.savefig(os.path.join(hier_dir, "cross_block_simmat.png"),
                            dpi=150, bbox_inches="tight")
                plt.close()

            # ── leaf and output filter sims ─────────────────────────────────
            last_W   = output_convs[-1]
            leaf_W   = last_W[total - n_leaf:]
            hier_leaf_Ws.append(leaf_W)
            sim_leaf  = leaf_W @ leaf_W.T
            sim_all   = last_W @ last_W.T
            _save_simmat(
                sim_leaf,
                title=(f"Stage {s_idx} Hier {k}: leaf filter cosine sim\n"
                       f"({n_leaf} leaf channels, L={L})"),
                path=os.path.join(hier_dir, "leaf_simmat.png"),
            )
            mask_all  = ~np.eye(total,  dtype=bool)
            mask_leaf = ~np.eye(n_leaf, dtype=bool)
            vals_all  = sim_all[mask_all]
            vals_leaf = sim_leaf[mask_leaf]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(vals_all,  bins=60,             alpha=0.6, density=True,
                    color="steelblue",
                    label=f"All output filters ({total}ch)  mean={vals_all.mean():.3f}")
            ax.hist(vals_leaf, bins=max(10, n_leaf), alpha=0.6, density=True,
                    color="tomato",
                    label=f"Leaf filters ({n_leaf}ch)  mean={vals_leaf.mean():.3f}")
            ax.axvline(vals_all.mean(),  color="steelblue", linestyle="--", linewidth=1.5)
            ax.axvline(vals_leaf.mean(), color="tomato",    linestyle="--", linewidth=1.5)
            ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density")
            ax.set_title(f"Stage {s_idx} Hier {k}: pairwise filter cosine similarity")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(hier_dir, "output_vs_leaf_hist.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

            hier_stats[k] = dict(
                mean_sim_all=float(vals_all.mean()),   std_sim_all=float(vals_all.std()),
                mean_sim_leaf=float(vals_leaf.mean()), std_sim_leaf=float(vals_leaf.std()),
            )
            print(f"  Stage {s_idx} Hier {k}: all={total} (mean={vals_all.mean():.3f}), "
                  f"leaf={n_leaf} (mean={vals_leaf.mean():.3f})")

        # ── cross-hierarchy: K×K mean leaf filter cosine sim ───────────────
        cross_mean = np.zeros((K, K))
        for ki in range(K):
            for kj in range(K):
                cross_mean[ki, kj] = float(
                    (hier_leaf_Ws[ki] * hier_leaf_Ws[kj]).sum(axis=1).mean()
                )
        fig, ax = plt.subplots(figsize=(max(3, K * 0.9 + 1), max(3, K * 0.9 + 1)))
        im = ax.imshow(cross_mean, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([f"H{k}" for k in range(K)])
        ax.set_yticklabels([f"H{k}" for k in range(K)])
        for ki in range(K):
            for kj in range(K):
                ax.text(kj, ki, f"{cross_mean[ki, kj]:.2f}", ha="center", va="center",
                        fontsize=8, color="k" if abs(cross_mean[ki, kj]) < 0.6 else "w")
        ax.set_title(f"Stage {s_idx}: mean cross-hierarchy leaf filter cosine sim\n"
                     f"(mean dot-product per aligned leaf filter index)", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(stage_dir, "cross_hier_leaf_simmat.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        # ── per-leaf-filter-index diversity across hierarchies ─────────────
        off_diag_pairs = [(ki, kj) for ki in range(K) for kj in range(ki + 1, K)]
        if off_diag_pairs:
            # cross_sims[ki,kj,i] = cosine sim of leaf filter i between hier ki and kj
            cross_sims_per_filter = np.array([
                (hier_leaf_Ws[ki] * hier_leaf_Ws[kj]).sum(axis=1)
                for ki, kj in off_diag_pairs
            ])  # (n_pairs, n_leaf)
            mean_cross = cross_sims_per_filter.mean(axis=0)  # (n_leaf,)
            div_cross  = cross_sims_per_filter.std(axis=0)   # (n_leaf,) divergence
            x = np.arange(n_leaf)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            colours = plt.cm.RdYlGn((mean_cross + 1) / 2)
            axes[0].bar(x, mean_cross, color=colours)
            axes[0].set_xlabel("Leaf filter index")
            axes[0].set_ylabel("Mean cross-hierarchy cosine sim")
            axes[0].set_title(f"Stage {s_idx}: mean cross-hierarchy sim per leaf filter")
            axes[0].grid(axis="y", alpha=0.3)
            axes[1].bar(x, div_cross, color="mediumpurple")
            axes[1].set_xlabel("Leaf filter index")
            axes[1].set_ylabel("Std across hierarchy pairs")
            axes[1].set_title(f"Stage {s_idx}: cross-hierarchy divergence per leaf filter")
            axes[1].grid(axis="y", alpha=0.3)
            plt.suptitle(f"Stage {s_idx}: Per-Leaf-Filter Cross-Hierarchy Analysis",
                         fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, "cross_hier_leaf_diversity.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

        all_stage_stats[s_idx] = hier_stats

    # ── summary across all stages and hierarchies ──────────────────────────
    x_labels = [f"S{s}H{k}" for s in range(1, n_stages + 1) for k in range(K)]
    xs = np.arange(len(x_labels))
    colours = plt.cm.tab10(np.array([k % 10 for _ in range(n_stages) for k in range(K)]))
    vals_all_m  = [all_stage_stats[s][k]["mean_sim_all"]  for s in range(1, n_stages+1) for k in range(K)]
    vals_leaf_m = [all_stage_stats[s][k]["mean_sim_leaf"] for s in range(1, n_stages+1) for k in range(K)]
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(xs) * 1.0), 5))
    axes[0].bar(xs, vals_all_m,  color=colours, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_title("Mean pairwise sim — all output filters"); axes[0].grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean cosine similarity")
    axes[1].bar(xs, vals_leaf_m, color=colours, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_title("Mean pairwise sim — leaf filters only"); axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylabel("Mean cosine similarity")
    plt.suptitle("Filter Similarity Summary — All Stages & Hierarchies",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "summary_mean_sim.png"), dpi=150, bbox_inches="tight")
    plt.close()

    np.savez(
        os.path.join(base_dir, "filter_similarity_stats.npz"),
        **{f"stage{s}_hier{k}_{metric}": np.array(v)
           for s, hd in all_stage_stats.items()
           for k, d in hd.items()
           for metric, v in d.items()},
    )
    print(f"Filter similarity analysis saved to {base_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# A2. Taxonomy regularisation distributions
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_taxonomy_distributions(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_batches: int = 10,
) -> None:
    """Plot per-batch entropy & DKL for each (stage, hierarchy) pair."""
    base_dir = os.path.join(save_dir, "taxonomy_distributions")
    os.makedirs(base_dir, exist_ok=True)
    model.eval()

    n_stages = len(model.encoder.multi_taxon_stages)
    K = model.n_hierarchies
    entropy_acc = [[[] for _ in range(K)] for _ in range(n_stages)]
    dkl_acc     = [[[] for _ in range(K)] for _ in range(n_stages)]

    for b_idx, (images, _) in enumerate(tqdm(data_loader, desc="Taxonomy dists")):
        if b_idx >= num_batches:
            break
        images = images.to(device)
        stage_inputs = _extract_stage_inputs(model, images, device)
        for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages):
            x_s = stage_inputs[s_idx]
            for k, hierarchy in enumerate(mt_stage.hierarchies):
                _, _, regs = hierarchy(x_s)
                entropy_acc[s_idx][k].append(float(regs["entropy"].cpu()))
                dkl_acc[s_idx][k].append(float(regs["dkl"].cpu()))

    for s_idx in range(n_stages):
        for k in range(K):
            out_dir = os.path.join(base_dir, f"stage{s_idx+1}", f"hier{k}")
            os.makedirs(out_dir, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].plot(entropy_acc[s_idx][k], marker="o", ms=3)
            axes[0].set_title(f"Stage {s_idx+1} Hier {k} — Entropy")
            axes[0].set_xlabel("batch")
            axes[1].plot(dkl_acc[s_idx][k], marker="o", ms=3, color="tab:orange")
            axes[1].set_title(f"Stage {s_idx+1} Hier {k} — DKL")
            axes[1].set_xlabel("batch")
            plt.suptitle(f"Stage {s_idx+1} Hier {k} taxonomy regularisation", fontweight="bold")
            plt.tight_layout()
            out = os.path.join(out_dir, "taxonomy_regularisation.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()

    fig, axes = plt.subplots(
        2, n_stages * K,
        figsize=(3.5 * n_stages * K, 7),
        squeeze=False,
    )
    col = 0
    for s_idx in range(n_stages):
        n_layers = model.encoder.multi_taxon_stages[s_idx].hierarchies[0].n_taxonomy_layers
        for k in range(K):
            axes[0, col].plot(entropy_acc[s_idx][k], marker="o", ms=2)
            axes[0, col].set_title(f"S{s_idx+1}H{k} ent\n(d={n_layers})", fontsize=8)
            axes[1, col].plot(dkl_acc[s_idx][k], marker="o", ms=2, color="tab:orange")
            axes[1, col].set_title(f"S{s_idx+1}H{k} dkl", fontsize=8)
            col += 1
    plt.suptitle("Per-(Stage, Hierarchy) Taxonomy Regularisation", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(base_dir, "taxonomy_regularisation_overview.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()

    np.savez(
        os.path.join(base_dir, "taxonomy_stats.npz"),
        **{f"s{s+1}_h{k}_entropy": np.array(entropy_acc[s][k])
           for s in range(n_stages) for k in range(K)},
        **{f"s{s+1}_h{k}_dkl":     np.array(dkl_acc[s][k])
           for s in range(n_stages) for k in range(K)},
    )
    print(f"Taxonomy distributions saved to {base_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# A3. Path-probability plots
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_taxonomy_path_probs(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_patches: int = 20,
    sample_index: int = 0,
    alpha: float = 0.6,
) -> None:
    """Path-probability plots for each (stage, hierarchy) pair."""
    base_dir = os.path.join(save_dir, "path_probability_plots")
    model.eval()
    images, _ = next(iter(data_loader))
    images = images.to(device)
    stage_inputs = _extract_stage_inputs(model, images, device)

    for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
        x_s = stage_inputs[s_idx - 1]
        for k, hierarchy in enumerate(mt_stage.hierarchies):
            out_dir = os.path.join(base_dir, f"stage{s_idx}", f"hier{k}")
            os.makedirs(out_dir, exist_ok=True)

            depths = _extract_hierarchy_depths(hierarchy, x_s)
            layer_channels = [d["n_nodes"] for d in depths]

            prob_splits = [d["probs"][sample_index:sample_index+1] for d in depths]
            flattened = torch.cat(prob_splits, dim=1)
            layer_sums = torch.stack(
                [p.sum(dim=1) for p in prob_splits], dim=1
            )

            patch_probs = flattened[0].permute(1, 2, 0).reshape(-1, flattened.shape[1])
            depth_sums  = layer_sums[0].permute(1, 2, 0).reshape(-1, len(layer_channels))

            n_show = min(num_patches, patch_probs.shape[0])
            patch_probs = patch_probs[:n_show].numpy()
            depth_sums  = depth_sums[:n_show].numpy()

            total_C = patch_probs.shape[1]
            x_full = np.arange(total_C)
            boundaries = (np.cumsum(layer_channels)[:-1] - 0.5).tolist()

            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [2.4, 1.2]})
            start = 0
            layer_slices = []
            for width in layer_channels:
                end = start + width
                layer_slices.append((x_full[start:end], start, end))
                start = end
            for i in range(n_show):
                for x_seg, s, e in layer_slices:
                    ax.plot(x_seg, patch_probs[i, s:e], alpha=alpha, linewidth=0.8)
            for b in boundaries:
                ax.axvline(b, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Flattened tree channel")
            ax.set_ylabel("Path probability")
            ax.set_title(f"Stage {s_idx} Hier {k} — path dists ({n_show} patches)")

            im = ax2.imshow(depth_sums.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax2.set_xlabel("Patch")
            ax2.set_ylabel("Depth")
            ax2.set_yticks(np.arange(len(layer_channels)))
            ax2.set_yticklabels([f"L{i+1}" for i in range(len(layer_channels))])
            ax2.set_title("Per-layer path sum (\u22481)")
            plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            err = np.abs(depth_sums - 1.0).max()
            ax2.set_xlabel(f"Patch  [max|sum-1|={err:.2e}]")
            plt.tight_layout()
            out = os.path.join(out_dir, f"stage{s_idx}_hier{k}_path_probs.png")
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Stage {s_idx} Hier {k}: path prob plot -> {out}")

    print(f"Path probability plots saved to {base_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# A4. Taxonomy tree activations
# ══════════════════════════════════════════════════════════════════════════════

def _draw_taxonomy_tree_fig(
    s_idx: int,
    k: int,
    img_idx: int,
    depths,
    max_tree_depths: int,
) -> plt.Figure:
    """Render taxonomy tree for one (image, stage, hierarchy) and return the Figure."""
    layer_channels = [d["n_nodes"] for d in depths]
    n_depths_show = min(max_tree_depths, len(layer_channels))
    shown_channels = [d["n_nodes"] for d in depths[:n_depths_show]]

    H, W = depths[0]["gated_acts"].shape[2], depths[0]["gated_acts"].shape[3]
    cy, cx = H // 2, W // 2

    acts  = [depths[d]["gated_acts"][img_idx, :, cy, cx].numpy()  for d in range(n_depths_show)]
    probs = [depths[d]["probs"][img_idx, :, cy, cx].numpy()       for d in range(n_depths_show)]

    all_acts = np.concatenate(acts)
    vmax = max(np.abs(all_acts).max(), 1e-8)

    def node_pos(depth, idx):
        n = layer_channels[depth]
        return (idx + 0.5) / n, -depth

    fig, ax = plt.subplots(
        1, 1,
        figsize=(max(8, 2 * shown_channels[-1]), (n_depths_show + 1) * 2.5 + 1),
    )
    ax.set_xlim(0, 1); ax.set_ylim(-n_depths_show, 1.5)
    ax.set_aspect("auto"); ax.axis("off")
    ax.set_title(
        f"Stage {s_idx} Hier {k}  —  img {img_idx+1}  —  centre ({cy},{cx})\n"
        "Node colour = activation, edge width \u221d cond prob",
        fontsize=9,
    )

    cmap_tree = plt.cm.RdYlGn
    r = 0.012
    vroot_x, vroot_y = 0.5, 1.0
    ax.add_patch(plt.Circle((vroot_x, vroot_y), r * 1.8,
                             color="lightgray", zorder=2,
                             linewidth=0.5, edgecolor="black"))
    ax.text(vroot_x, vroot_y - r * 2.8, "root",
            ha="center", va="top", fontsize=5.5, zorder=3, color="dimgray")
    for i in range(layer_channels[0]):
        x_c, y_c = node_pos(0, i)
        ax.plot([vroot_x, x_c], [vroot_y, y_c],
                color="steelblue", linewidth=1.0, alpha=0.5, zorder=1)
    for d in range(1, n_depths_show):
        n_nodes = layer_channels[d]
        for i in range(n_nodes):
            parent_i = i // 2
            x_c, y_c = node_pos(d, i)
            x_p, y_p = node_pos(d - 1, parent_i)
            cond_prob = float(probs[d][i]) / max(float(probs[d-1][parent_i]), 1e-8)
            cond_prob = min(cond_prob, 1.0)
            ax.plot([x_p, x_c], [y_p, y_c], color="steelblue",
                    linewidth=0.5 + 3.5 * cond_prob,
                    alpha=0.4 + 0.5 * cond_prob, zorder=1)
    for d in range(n_depths_show):
        for i in range(layer_channels[d]):
            x_n, y_n = node_pos(d, i)
            norm_v = (float(acts[d][i]) + vmax) / (2 * vmax)
            circle = plt.Circle(
                (x_n, y_n), r * (1.8 if layer_channels[d] <= 4 else 1.0),
                color=cmap_tree(norm_v), zorder=2,
                linewidth=0.5, edgecolor="black",
            )
            ax.add_patch(circle)
            ax.text(x_n, y_n - r * 2.8,
                    f"c{i}\n{acts[d][i]:.2f}\np={probs[d][i]:.2f}",
                    ha="center", va="top", fontsize=5.5, zorder=3)
        ax.text(0.002, -d, f"L{d+1}\n({layer_channels[d]}ch)",
                va="center", ha="left", fontsize=7, color="dimgray")
    sm = plt.cm.ScalarMappable(cmap=cmap_tree,
                                norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.01, label="Activation")
    plt.tight_layout()
    return fig


def _combine_pngs_horizontal(paths: List[str], out_path: str, title: str) -> None:
    """Stack a list of saved PNG files side-by-side into one combined PNG."""
    from PIL import Image as _PIL
    imgs = [_PIL.open(p) for p in paths if os.path.exists(p)]
    if not imgs:
        return
    max_h = max(im.height for im in imgs)
    total_w = sum(im.width for im in imgs)
    combined = _PIL.new("RGB", (total_w, max_h), (255, 255, 255))
    x_off = 0
    for im in imgs:
        combined.paste(im, (x_off, 0))
        x_off += im.width
    combined.save(out_path)


@torch.no_grad()
def visualize_taxonomy_tree(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 2,
    max_tree_depths: int = 3,
) -> None:
    """Draw binary-tree activation structure.

    Structure: base_dir/image_{N}/stage{S}/hier{K}_tree.png
               base_dir/image_{N}/stage{S}/all_hierarchies.png
    """
    base_dir = os.path.join(save_dir, "taxonomy_tree_activations")
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    stage_inputs = _extract_stage_inputs(model, images, device)

    for img_idx in range(num_images):
        img_dir = os.path.join(base_dir, f"image_{img_idx+1:02d}")
        for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
            stage_dir = os.path.join(img_dir, f"stage{s_idx}")
            os.makedirs(stage_dir, exist_ok=True)
            x_s = stage_inputs[s_idx - 1]
            hier_paths: List[str] = []
            for k, hierarchy in enumerate(mt_stage.hierarchies):
                depths = _extract_hierarchy_depths(hierarchy, x_s)
                fig = _draw_taxonomy_tree_fig(s_idx, k, img_idx, depths, max_tree_depths)
                out = os.path.join(stage_dir, f"hier{k}_tree.png")
                fig.savefig(out, dpi=180, bbox_inches="tight")
                plt.close(fig)
                hier_paths.append(out)
                print(f"  img{img_idx+1} Stage {s_idx} Hier {k} tree \u2192 {out}")
            # Combined image for this (image, stage)
            combined_out = os.path.join(stage_dir, "all_hierarchies.png")
            _combine_pngs_horizontal(hier_paths, combined_out,
                                     f"img{img_idx+1} Stage {s_idx} — all hierarchies")
            print(f"  Combined \u2192 {combined_out}")
    print(f"Taxonomy tree visualisations saved to {base_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# A5. Parse-tree analysis
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _extract_parse_tree_for_hierarchy(
    hierarchy: TaxonResNetStage,
    x_s: torch.Tensor,
    batch_size: int,
) -> list:
    """Extract taxonomy parse-tree decisions for one hierarchy and its input."""
    depths_data = _extract_hierarchy_depths(hierarchy, x_s)
    per_sample = []
    for sample_idx in range(batch_size):
        depths_out = []
        for d_info in depths_data:
            probs = d_info["probs"][sample_idx].mean(dim=(-1, -2)).numpy()
            selected = int(probs.argmax())
            depth_num = d_info["depth"]
            bits = [(selected >> (depth_num - 1 - b)) & 1 for b in range(depth_num)]
            path_str = "-".join("R" if b else "L" for b in bits)
            sibling   = selected ^ 1
            pair_total = float(probs[selected]) + float(probs[sibling])
            cond_prob  = float(probs[selected]) / max(pair_total, 1e-9)
            depths_out.append({
                "depth":     depth_num,
                "n_nodes":   d_info["n_nodes"],
                "probs":     probs,
                "selected":  selected,
                "path_bits": bits,
                "path_str":  path_str,
                "cond_prob": cond_prob,
            })
        per_sample.append(depths_out)
    return per_sample


def _visualize_parse_tree(per_sample, images, save_dir, max_samples=4, title_prefix=""):
    os.makedirs(save_dir, exist_ok=True)
    for sample_idx, depths in enumerate(per_sample[:max_samples]):
        n_depths = len(depths)
        fig_w = 3.5 + 4.5
        fig_h = 6
        fig, (ax_img, ax) = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#1a1a2e")
        for a in (ax_img, ax):
            a.set_facecolor("#1a1a2e")
        img_np = images[sample_idx].cpu().float().numpy()
        img_np = np.clip(img_np * 0.5 + 0.5, 0, 1)
        if img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)
        ax_img.imshow(img_np)
        ax_img.set_title(f"Sample {sample_idx}", color="white", fontsize=10, pad=6)
        ax_img.axis("off")

        cmap_pt = cm.get_cmap("YlOrRd")
        ax.set_xlim(-1, 1); ax.set_ylim(-0.5, n_depths + 0.5)
        ax.set_title(f"{title_prefix}\n({n_depths} depths)", color="white", fontsize=9, pad=6)
        ax.axis("off")
        sel_per_depth = [d["selected"] for d in depths]
        for d_idx, d_info in enumerate(depths):
            y = n_depths - d_idx
            n_nodes = d_info["n_nodes"]
            probs = d_info["probs"]
            selected = d_info["selected"]
            xs = np.linspace(-0.9, 0.9, n_nodes)
            if d_idx + 1 < n_depths:
                y_child = n_depths - (d_idx + 1)
                n_ch = n_nodes * 2
                xs_ch = np.linspace(-0.9, 0.9, n_ch)
                for ni, xp in enumerate(xs):
                    for ci in (ni * 2, ni * 2 + 1):
                        on_path = sel_per_depth[d_idx] == ni and sel_per_depth[d_idx+1] == ci
                        ax.plot([xp, xs_ch[ci]], [y, y_child],
                                color="#e63946" if on_path else "#444466",
                                lw=2.5 if on_path else 0.7, zorder=1)
            for ni, (xn, pv) in enumerate(zip(xs, probs)):
                cc = cmap_pt(float(pv) / max(probs.max(), 1e-9))
                is_sel = ni == selected
                circle = plt.Circle(
                    (xn, y), radius=0.07 if n_nodes <= 32 else 0.04,
                    color=cc, ec="#e63946" if is_sel else "#888899",
                    lw=2.0 if is_sel else 0.5, zorder=2,
                )
                ax.add_patch(circle)
        sm = plt.cm.ScalarMappable(cmap=cmap_pt, norm=plt.Normalize(vmin=0, vmax=float(depths[-1]["probs"].max())))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, orientation="vertical")
        cbar.set_label("joint prob", color="white", fontsize=7)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white", fontsize=6)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.outline.set_edgecolor("white")
        sel_patch = mpatches.Patch(color="#e63946", label="selected path")
        fig.legend(handles=[sel_patch], loc="lower center", ncol=1,
                   facecolor="#1a1a2e", edgecolor="white", labelcolor="white", fontsize=8)
        plt.suptitle(f"Parse Tree - Sample {sample_idx} - {title_prefix}",
                     color="white", fontsize=11, fontweight="bold", y=1.01)
        plt.tight_layout()
        out = os.path.join(save_dir, f"parse_tree_sample_{sample_idx:03d}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"  parse tree \u2192 {out}")


def analyze_parse_trees(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 4,
) -> None:
    """Run parse-tree analysis.

    Structure: base_dir/image_{N}/stage{S}/hier{K}_parse_tree.png
               base_dir/image_{N}/stage{S}/all_hierarchies.png
    """
    base_dir = os.path.join(save_dir, "parse_tree_viz")
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    stage_inputs = _extract_stage_inputs(model, images, device)

    # Pre-extract per_sample for all (stage, hier)
    all_per_sample: Dict[Tuple[int, int], list] = {}
    for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
        x_s = stage_inputs[s_idx - 1]
        for k, hierarchy in enumerate(mt_stage.hierarchies):
            all_per_sample[(s_idx, k)] = _extract_parse_tree_for_hierarchy(
                hierarchy, x_s, num_images
            )

    for img_idx in range(num_images):
        for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
            stage_dir = os.path.join(base_dir, f"image_{img_idx+1:02d}", f"stage{s_idx}")
            os.makedirs(stage_dir, exist_ok=True)
            hier_paths: List[str] = []
            for k in range(len(mt_stage.hierarchies)):
                per_sample = all_per_sample[(s_idx, k)]
                # Save JSON summary for this image
                depths = per_sample[img_idx]
                serialisable = [{
                    "depth": d["depth"], "n_nodes": d["n_nodes"],
                    "selected": d["selected"], "path_bits": d["path_bits"],
                    "path_str": d["path_str"],
                    "cond_prob": round(float(d["cond_prob"]), 6),
                    "joint_prob": round(float(d["probs"][d["selected"]]), 6),
                } for d in depths]
                with open(os.path.join(stage_dir, f"hier{k}_parse_tree_summary.json"), "w") as f:
                    json.dump(serialisable, f, indent=2)
                # Save individual PNG for this (image, stage, hier)
                out = os.path.join(stage_dir, f"hier{k}_parse_tree.png")
                _visualize_parse_tree([per_sample[img_idx]], images.cpu()[img_idx:img_idx+1],
                                      stage_dir, max_samples=1,
                                      title_prefix=f"Stage {s_idx} Hier {k}")
                # _visualize_parse_tree saves as parse_tree_sample_000.png — rename
                tmp = os.path.join(stage_dir, "parse_tree_sample_000.png")
                if os.path.exists(tmp):
                    os.replace(tmp, out)
                hier_paths.append(out)
                print(f"  img{img_idx+1} Stage {s_idx} Hier {k} parse tree \u2192 {out}")
            combined_out = os.path.join(stage_dir, "all_hierarchies.png")
            _combine_pngs_horizontal(hier_paths, combined_out,
                                     f"img{img_idx+1} Stage {s_idx} — all hierarchies")
            print(f"  Combined \u2192 {combined_out}")
    print(f"Parse tree visualisations saved to {base_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# A6. Hierarchical activation maps
# ══════════════════════════════════════════════════════════════════════════════

def _render_hier_acts(s_idx, k, img_idx, depths, max_depth, max_cols, use_nms, nms_threshold, save_dir) -> List[str]:
    """Render hierarchical activation map for one (stage, hier, image).

    Returns list of saved file paths (acts png, and optionally winner-map png).
    """
    os.makedirs(save_dir, exist_ok=True)
    saved: List[str] = []
    depths_show = depths[:max_depth]
    n_rows = len(depths_show)
    max_cols_here = min(depths_show[-1]["n_nodes"], max_cols)
    cell = 1.6
    fig_w = max(6.0, cell * max_cols_here)
    fig_h = cell * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, max_cols_here, figsize=(fig_w, fig_h),
                             squeeze=False, facecolor="#0f0f1c")
    fig.subplots_adjust(hspace=0.55, wspace=0.12, left=0.06, right=0.98, top=0.90, bottom=0.04)
    for ax in axes.flat:
        ax.set_visible(False)

    if use_nms:
        _wmaps, _wprobs, _cmasks = [], [], []
        for d_info in depths_show:
            pm = d_info["probs"][img_idx]
            w = pm.argmax(dim=0).numpy()
            wp = pm.max(dim=0).values.numpy()
            _wmaps.append(w); _wprobs.append(wp); _cmasks.append(wp > nms_threshold)

    for row, d_info in enumerate(depths_show):
        d = d_info["depth"]
        n_nodes = d_info["n_nodes"]
        gated = d_info["gated_acts"][img_idx]
        prob_vals = d_info["probs"][img_idx].mean(dim=(-1, -2)).numpy()
        selected = int(prob_vals.argmax())
        if use_nms:
            _wmap = _wmaps[row]; _cmask = _cmasks[row]
        if n_nodes <= max_cols_here:
            show_nodes = list(range(n_nodes)); col_offset = (max_cols_here - n_nodes) // 2
        else:
            half = max_cols_here // 2
            start = max(0, min(selected - half, n_nodes - max_cols_here))
            show_nodes = list(range(start, start + max_cols_here)); col_offset = 0
        for col_idx, node_i in enumerate(show_nodes):
            col = col_offset + col_idx
            if col >= max_cols_here:
                break
            ax = axes[row, col]; ax.set_visible(True); ax.set_facecolor("#0f0f1c")
            act = gated[node_i].numpy()
            if use_nms:
                nms_mask = (_wmap == node_i) & _cmask
                act = act * nms_mask.astype(act.dtype)
            vmax = float(abs(act).max()) or 1e-6
            ax.imshow(act, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            is_sel = node_i == selected
            ec = "#e63946" if is_sel else "#2a2a4a"; lw = 2.2 if is_sel else 0.5
            for spine in ax.spines.values():
                spine.set_edgecolor(ec); spine.set_linewidth(lw)
            p = float(prob_vals[node_i])
            label = (f"*{node_i}" if is_sel else f"{node_i}") + f"\n{p:.3f}"
            ax.set_title(label, fontsize=5.5, color="#e63946" if is_sel else "#7070a0", pad=1.5)
        left_col = col_offset if n_nodes <= max_cols_here else 0
        axes[row, left_col].set_ylabel(f"depth {d}", color="#aaaacc", fontsize=6.5, rotation=90, labelpad=3)

    nms_tag = " [NMS]" if use_nms else ""
    fig.suptitle(f"Hier Acts{nms_tag}  *  img{img_idx}  *  Stage{s_idx} Hier{k}",
                 color="white", fontsize=10, fontweight="bold")
    suffix = "_nms" if use_nms else ""
    out = os.path.join(save_dir, f"hier_acts_img{img_idx:03d}{suffix}.png")
    plt.savefig(out, dpi=140, facecolor=fig.get_facecolor()); plt.close()
    saved.append(out)
    print(f"  saved {out}")

    if use_nms:
        from matplotlib.colors import BoundaryNorm, ListedColormap as _LC
        wfig, waxes = plt.subplots(1, len(depths_show), figsize=(3.5*len(depths_show), 3.5),
                                   facecolor="#0f0f1c", squeeze=False)
        wfig.subplots_adjust(wspace=0.15, left=0.04, right=0.96, top=0.82, bottom=0.06)
        for co, (d_info, wmap, wprob, cmask) in enumerate(zip(depths_show, _wmaps, _wprobs, _cmasks)):
            ax2 = waxes[0, co]; ax2.set_facecolor("#0f0f1c")
            n_nodes = d_info["n_nodes"]
            disp = np.where(cmask, wmap, -1).astype(float)
            cn = plt.cm.get_cmap("tab20", n_nodes)
            colors = ["#0f0f1c"] + [cn(i / max(n_nodes - 1, 1)) for i in range(n_nodes)]
            lmap = _LC(colors)
            from matplotlib.colors import BoundaryNorm
            lnorm = BoundaryNorm([-1.5] + [i - 0.5 for i in range(n_nodes + 1)], len(colors))
            ax2.imshow(disp, cmap=lmap, norm=lnorm, aspect="auto", interpolation="nearest")
            alpha_map = np.clip(wprob * cmask.astype(float), 0, 1)
            ax2.imshow(alpha_map, cmap="gray", alpha=0.25, aspect="auto", interpolation="nearest")
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_title(f"depth {d_info['depth']} ({n_nodes})", color="#aaaacc", fontsize=8, pad=3)
            for spine in ax2.spines.values(): spine.set_edgecolor("#333355")
        thresh_str = f" (thr={nms_threshold:.2f})" if nms_threshold > 0 else ""
        wfig.suptitle(f"Winner Map{thresh_str}  *  img{img_idx}  *  Stage{s_idx} Hier{k}",
                      color="white", fontsize=10, fontweight="bold")
        wout = os.path.join(save_dir, f"winner_map_img{img_idx:03d}.png")
        wfig.savefig(wout, dpi=140, facecolor=wfig.get_facecolor()); plt.close(wfig)
        saved.append(wout)
        print(f"  saved {wout}")
    return saved


@torch.no_grad()
def analyze_hierarchical_activations(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 4,
    max_depth: int = 4,
    max_cols: int = 16,
    use_nms: bool = False,
    nms_threshold: float = 0.0,
) -> None:
    """Hierarchical activation maps.

    Structure: base_dir/image_{N}/stage{S}/hier{K}_hier_acts.png
               base_dir/image_{N}/stage{S}/all_hierarchies.png
    """
    base_dir = os.path.join(save_dir, "hierarchical_activations")
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    stage_inputs = _extract_stage_inputs(model, images, device)

    # Pre-compute all depths per (stage, hier) once (they don't depend on img_idx)
    all_depths: Dict[Tuple[int, int], list] = {}
    for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
        x_s = stage_inputs[s_idx - 1]
        for k, hierarchy in enumerate(mt_stage.hierarchies):
            all_depths[(s_idx, k)] = _extract_hierarchy_depths(hierarchy, x_s)

    for img_idx in range(num_images):
        for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
            stage_dir = os.path.join(base_dir, f"image_{img_idx+1:02d}", f"stage{s_idx}")
            hier_paths: List[str] = []
            for k in range(len(mt_stage.hierarchies)):
                depths = all_depths[(s_idx, k)]
                # Render into a temp dir, then rename with hier prefix
                tmp_dir = stage_dir
                saved = _render_hier_acts(s_idx, k, img_idx, depths, max_depth, max_cols,
                                          use_nms, nms_threshold, tmp_dir)
                # Rename so filename carries hier{k}_ prefix
                renamed = []
                for p in saved:
                    fname = os.path.basename(p)
                    new_p = os.path.join(stage_dir, f"hier{k}_{fname}")
                    os.replace(p, new_p)
                    renamed.append(new_p)
                # Keep only the activation map (not winner map) for the combined image
                acts_pngs = [p for p in renamed if "hier_acts" in os.path.basename(p)]
                hier_paths.extend(acts_pngs)
            combined_out = os.path.join(stage_dir, "all_hierarchies.png")
            _combine_pngs_horizontal(hier_paths, combined_out,
                                     f"img{img_idx+1} Stage {s_idx} — all hierarchies")
            print(f"  Combined → {combined_out}")
    print(f"Hierarchical activation maps saved to {base_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# A7. Binary split maps
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_binary_split_maps(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 4,
    max_depth: int = 4,
    max_pairs: int = 8,
) -> None:
    """Binary split maps.

    Structure: base_dir/image_{N}/stage{S}/hier{K}_split_map.png
               base_dir/image_{N}/stage{S}/all_hierarchies.png
    """
    base_dir = os.path.join(save_dir, "binary_split_maps")
    rdbu = cm.get_cmap("RdBu_r")
    bg_colour = np.array([0.06, 0.06, 0.11, 1.0])
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    stage_inputs = _extract_stage_inputs(model, images, device)

    # Pre-extract all depths
    all_depths: Dict[Tuple[int, int], list] = {}
    for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
        x_s = stage_inputs[s_idx - 1]
        for k, hierarchy in enumerate(mt_stage.hierarchies):
            all_depths[(s_idx, k)] = _extract_hierarchy_depths(hierarchy, x_s)

    for img_idx in range(num_images):
        for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
            stage_dir = os.path.join(base_dir, f"image_{img_idx+1:02d}", f"stage{s_idx}")
            os.makedirs(stage_dir, exist_ok=True)
            hier_paths: List[str] = []
            for k in range(len(mt_stage.hierarchies)):
                depths = all_depths[(s_idx, k)]
                depths_show = depths[:max_depth]
                n_rows = len(depths_show)
                max_pairs_here = max(1, min(depths_show[-1]["n_nodes"] // 2, max_pairs))
                cell_w, cell_h = 2.0, 2.0
                fig_w = max(5.0, cell_w * max_pairs_here + 1.0)
                fig_h = cell_h * n_rows + 0.9
                fig, axes = plt.subplots(n_rows, max_pairs_here, figsize=(fig_w, fig_h),
                                         squeeze=False, facecolor="#0f0f1c")
                fig.subplots_adjust(hspace=0.65, wspace=0.08, left=0.07, right=0.97, top=0.88, bottom=0.05)
                for ax in axes.flat:
                    ax.set_visible(False)

                for row, d_info in enumerate(depths_show):
                    d = d_info["depth"]
                    n_nodes = d_info["n_nodes"]
                    n_pairs = n_nodes // 2
                    probs = d_info["probs"][img_idx]
                    prob_vals = probs.mean(dim=(-1, -2)).numpy()
                    selected_node = int(prob_vals.argmax())
                    selected_pair = selected_node // 2
                    if n_pairs <= max_pairs_here:
                        show_pairs = list(range(n_pairs)); col_offset = (max_pairs_here - n_pairs) // 2
                    else:
                        half = max_pairs_here // 2
                        start = max(0, min(selected_pair - half, n_pairs - max_pairs_here))
                        show_pairs = list(range(start, start + max_pairs_here)); col_offset = 0
                    p_parent_all = float((probs[0::2] + probs[1::2]).max())
                    p_parent_all = max(p_parent_all, 1e-9)
                    for col_idx, pair_k in enumerate(show_pairs):
                        col_pos = col_offset + col_idx
                        if col_pos >= max_pairs_here: break
                        ax = axes[row, col_pos]; ax.set_visible(True); ax.set_facecolor("#0f0f1c")
                        p_l = probs[2 * pair_k].numpy(); p_r = probs[2 * pair_k + 1].numpy()
                        p_parent = p_l + p_r
                        cond_left = p_l / np.maximum(p_parent, 1e-9)
                        alpha = np.clip(p_parent / p_parent_all * 4.0, 0.0, 1.0)
                        rgba = rdbu(cond_left).astype(np.float64); rgba[..., 3] = alpha
                        H_img, W_img = cond_left.shape
                        bg = np.ones((H_img, W_img, 4)) * bg_colour
                        ax.imshow(bg, aspect="auto", interpolation="nearest")
                        ax.imshow(rgba, aspect="auto", interpolation="nearest")
                        ax.set_xticks([]); ax.set_yticks([])
                        is_sel = pair_k == selected_pair
                        ec = "#e63946" if is_sel else "#1e1e3a"; lw = 2.5 if is_sel else 0.6
                        for spine in ax.spines.values():
                            spine.set_edgecolor(ec); spine.set_linewidth(lw)
                        mean_cond = float(cond_left.mean())
                        arrow = "<- L" if mean_cond > 0.5 else "R ->"
                        conf_val = abs(mean_cond - 0.5) * 2.0
                        ax.set_title(
                            f"{'* ' if is_sel else ''}L{2*pair_k}|R{2*pair_k+1}\n{arrow}  {conf_val:.2f}",
                            fontsize=5.5, color="#e63946" if is_sel else "#8888aa", pad=1.5,
                        )
                    left_col = col_offset if n_pairs <= max_pairs_here else 0
                    axes[row, left_col].set_ylabel(
                        f"depth {d}", color="#aaaacc", fontsize=7, rotation=90, labelpad=3,
                    )

                sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=axes[:, -1], fraction=0.05, pad=0.02, orientation="vertical")
                cbar.set_ticks([0.0, 0.5, 1.0])
                cbar.set_ticklabels(["Right\n(P=1)", "Uncertain", "Left\n(P=1)"], fontsize=5.5)
                cbar.ax.yaxis.set_tick_params(color="white")
                cbar.outline.set_edgecolor("white")
                plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
                cbar.set_label("P(left | parent)", color="white", fontsize=7)
                fig.suptitle(
                    f"Binary Split  *  img{img_idx+1}  *  Stage{s_idx}  Hier{k}\n"
                    "(alpha \u221d parent prob)",
                    color="white", fontsize=9, fontweight="bold",
                )
                out = os.path.join(stage_dir, f"hier{k}_split_map.png")
                plt.savefig(out, dpi=140, facecolor=fig.get_facecolor()); plt.close()
                hier_paths.append(out)
                print(f"  saved {out}")
            combined_out = os.path.join(stage_dir, "all_hierarchies.png")
            _combine_pngs_horizontal(hier_paths, combined_out,
                                     f"img{img_idx+1} Stage {s_idx} \u2014 all hierarchies")
            print(f"  Combined \u2192 {combined_out}")
    print(f"Binary split maps saved to {base_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# A8. Stage activation maps
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_stage_activations(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 3,
) -> None:
    """Feature-map grids per (stage, hierarchy, image)."""
    base_dir = os.path.join(save_dir, "stage_activations")
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)
    stage_inputs = _extract_stage_inputs(model, images, device)

    for img_idx in range(num_images):
        img_dir = os.path.join(base_dir, f"image_{img_idx+1:02d}")
        os.makedirs(img_dir, exist_ok=True)
        plt.imsave(
            os.path.join(img_dir, "original.png"),
            _to_display(images[img_idx:img_idx+1], normalized=True),
        )
        for s_idx, mt_stage in enumerate(model.encoder.multi_taxon_stages, start=1):
            stage_dir = os.path.join(img_dir, f"stage{s_idx}")
            os.makedirs(stage_dir, exist_ok=True)
            x_s = stage_inputs[s_idx - 1][img_idx:img_idx+1]
            hier_paths: List[str] = []
            for k, hierarchy in enumerate(mt_stage.hierarchies):
                out, _, _ = hierarchy(x_s)
                maps = out[0].detach().cpu()
                n_show = min(maps.shape[0], 16)
                nrow = 4
                fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
                axes = np.array(axes).flatten()
                for ch in range(n_show):
                    fm = maps[ch].numpy()
                    fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
                    axes[ch].imshow(fm, cmap="viridis"); axes[ch].axis("off")
                for ax in axes[n_show:]:
                    ax.axis("off")
                plt.suptitle(
                    f"Stage {s_idx} Hier {k} — img {img_idx+1}\n"
                    f"(ch={maps.shape[0]}, showing {n_show})",
                    fontsize=10,
                )
                plt.tight_layout()
                hier_out = os.path.join(stage_dir, f"hier{k}_activations.png")
                plt.savefig(hier_out, dpi=150, bbox_inches="tight")
                plt.close()
                hier_paths.append(hier_out)
            combined_out = os.path.join(stage_dir, "all_hierarchies.png")
            _combine_pngs_horizontal(hier_paths, combined_out,
                                     f"img{img_idx+1} Stage {s_idx} \u2014 all hierarchies")
        print(f"  Image {img_idx+1}: saved to {img_dir}")
    print(f"Stage activation maps saved to {base_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# B1. Skip-connection decomposition (only runs when model has skip_rank > 0)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_skip_decomposition(
    model,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 8,
    num_sets: int = 4,
) -> None:
    """Decompose reconstruction into tree (hierarchy) and skip contributions.

    For each batch visualises:
      row 0 — original
      row 1 — tree-only reconstruction  (encoder + decoder, no skip)
      row 2 — skip contribution         (the additive skip term)
      row 3 — total reconstruction      (tree + skip, with output activation)

    Also saves aggregate metrics:
      - skip energy fraction  ||skip||² / ||total||²
      - skip MSE              ||x - skip||²  (how well skip alone explains x)
      - tree MSE              ||x - tree_only||²
      - cosine similarity     cos(skip_vec, tree_vec)  per image
    """
    import torch.nn.functional as F

    if not (hasattr(model, "skip_rank") and model.skip_rank > 0):
        print("  Model has no skip connection — skipping B1.")
        return

    model.eval()
    skip_dir = os.path.join(save_dir, "skip_decomposition")
    os.makedirs(skip_dir, exist_ok=True)

    skip_energy_fracs: List[float] = []
    skip_mses:  List[float] = []
    tree_mses:  List[float] = []
    cos_sims:   List[float] = []

    it = iter(data_loader)
    for set_idx in range(num_sets):
        try:
            images, _ = next(it)
        except StopIteration:
            it = iter(data_loader)
            images, _ = next(it)
        images = images[:num_images].to(device)

        # ── decomposed forward ─────────────────────────────────────────────
        z, _        = model.encode(images)
        tree_raw, _ = model.decode(z, output_size=images.shape[-2:])
        skip_recon  = model._compute_skip_recon(images)           # additive skip term
        total_recon = model._apply_output_activation(tree_raw + skip_recon)
        tree_only   = model._apply_output_activation(tree_raw)    # tree alone

        # ── aggregate metrics ──────────────────────────────────────────────
        B = images.size(0)
        skip_flat  = skip_recon.view(B, -1)
        total_flat = total_recon.view(B, -1)
        tree_flat  = tree_only.view(B, -1)
        x_flat     = images.view(B, -1)

        e_frac = (skip_flat.pow(2).sum(1) / (total_flat.pow(2).sum(1) + 1e-8)).cpu().tolist()
        s_mse  = ((x_flat - skip_flat).pow(2).mean(1)).cpu().tolist()
        t_mse  = ((x_flat - tree_flat).pow(2).mean(1)).cpu().tolist()
        cos    = F.cosine_similarity(skip_flat, tree_flat, dim=1).cpu().tolist()

        skip_energy_fracs.extend(e_frac)
        skip_mses.extend(s_mse)
        tree_mses.extend(t_mse)
        cos_sims.extend(cos)

        # ── visualise first set ────────────────────────────────────────────
        if set_idx < num_sets:
            rows = 4
            fig, axes = plt.subplots(rows, num_images, figsize=(num_images * 2.5, rows * 2.5))
            row_labels = ["Original", "Tree only", "Skip contribution", "Total recon"]
            tensors    = [images, tree_only, skip_recon, total_recon]
            for row, (label, t) in enumerate(zip(row_labels, tensors)):
                # skip_recon may be negative — centre around 0 for display
                norm = row != 2
                for col in range(num_images):
                    ax = axes[row, col]
                    ax.imshow(_to_display(t[col:col+1], normalized=norm))
                    ax.axis("off")
                    if col == 0:
                        ax.set_title(label, fontweight="bold", fontsize=9, loc="left")
            plt.suptitle(
                f"Skip decomposition — set {set_idx+1}  "
                f"(skip_rank={model.skip_rank})",
                fontsize=11,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(skip_dir, f"decomp_set_{set_idx+1:02d}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close()

    # ── aggregate stats ────────────────────────────────────────────────────
    sef  = np.array(skip_energy_fracs)
    smse = np.array(skip_mses)
    tmse = np.array(tree_mses)
    cs   = np.array(cos_sims)

    stats = {
        "skip_rank":             model.skip_rank,
        "skip_energy_frac_mean": float(sef.mean()),
        "skip_energy_frac_std":  float(sef.std()),
        "skip_mse_mean":         float(smse.mean()),
        "tree_mse_mean":         float(tmse.mean()),
        "skip_tree_cos_sim_mean":float(cs.mean()),
        "skip_tree_cos_sim_std": float(cs.std()),
    }
    with open(os.path.join(skip_dir, "skip_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  skip_energy_frac  = {sef.mean():.3f} ± {sef.std():.3f}")
    print(f"  tree MSE          = {tmse.mean():.6f}   skip MSE = {smse.mean():.6f}")
    print(f"  skip⊥tree cos-sim = {cs.mean():.3f} ± {cs.std():.3f}  (≈0 means orthogonal)")
    print(f"  Results saved to {skip_dir}")

    # ── summary histogram ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(sef, bins=40, edgecolor="black")
    axes[0].axvline(sef.mean(), color="red", linestyle="--",
                    label=f"mean={sef.mean():.3f}")
    axes[0].set_title("Skip energy fraction"); axes[0].legend()

    axes[1].hist(cs, bins=40, edgecolor="black")
    axes[1].axvline(cs.mean(), color="red", linestyle="--",
                    label=f"mean={cs.mean():.3f}")
    axes[1].set_title("Skip ⊥ Tree cosine similarity"); axes[1].legend()

    axes[2].bar(["Tree MSE", "Skip MSE"], [tmse.mean(), smse.mean()],
                color=["steelblue", "darkorange"],
                yerr=[tmse.std(), smse.std()], capsize=5)
    axes[2].set_title("MSE: tree-only vs skip-only")

    plt.suptitle(f"Skip decomposition summary (skip_rank={model.skip_rank})", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(skip_dir, "skip_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# B2. Reconstruction quality
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_reconstruction_quality(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_batches: int = 20,
) -> None:
    model.eval()
    mse_list: List[float] = []
    mae_list: List[float] = []
    print(f"Computing reconstruction metrics on {num_batches} batches...")
    for i, (images, _) in enumerate(tqdm(data_loader, desc="Recon quality")):
        if i >= num_batches:
            break
        images = images.to(device)
        recon, *_ = model(images)
        mse_list.extend(((images - recon)**2).mean(dim=(1,2,3)).cpu().numpy())
        mae_list.extend(torch.abs(images - recon).mean(dim=(1,2,3)).cpu().numpy())
    mse = np.array(mse_list); mae = np.array(mae_list)
    print(f"  Mean MSE={mse.mean():.6f}  Mean MAE={mae.mean():.6f}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(mse, bins=50, edgecolor="black")
    axes[0].axvline(mse.mean(), color="red", linestyle="--", label=f"Mean={mse.mean():.4f}")
    axes[0].set_title("MSE distribution"); axes[0].legend()
    axes[1].hist(mae, bins=50, edgecolor="black")
    axes[1].axvline(mae.mean(), color="red", linestyle="--", label=f"Mean={mae.mean():.4f}")
    axes[1].set_title("MAE distribution"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reconstruction_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    np.savez(os.path.join(save_dir, "reconstruction_metrics.npz"), mse=mse, mae=mae)
    with open(os.path.join(save_dir, "reconstruction_metrics.json"), "w") as f:
        json.dump({"mse": float(mse.mean()), "mae": float(mae.mean())}, f, indent=2)
    print(f"Reconstruction quality saved to {save_dir}")


@torch.no_grad()
def visualize_multiple_reconstructions(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 8,
    num_sets: int = 8,
) -> None:
    recon_dir = os.path.join(save_dir, "multiple_reconstructions")
    os.makedirs(recon_dir, exist_ok=True)
    model.eval()
    it = iter(data_loader)
    for set_idx in range(num_sets):
        try:
            images, _ = next(it)
        except StopIteration:
            it = iter(data_loader)
            images, _ = next(it)
        images = images[:num_images].to(device)
        with torch.no_grad():
            recon, *_ = model(images)
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2.5, 5))
        for i in range(num_images):
            axes[0, i].imshow(_to_display(images[i:i+1]))
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Original", fontweight="bold")
            axes[1, i].imshow(_to_display(recon[i:i+1]))
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_title("Reconstruction", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(recon_dir, f"set_{set_idx+1:02d}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Set {set_idx+1}/{num_sets} saved.")
    print(f"Reconstructions saved to {recon_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# B4. Latent space sparsity
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_latent_sparsity(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_batches: int = 50,
    sparsity_threshold: float = 1e-6,
) -> None:
    model.eval()
    all_latents: List[np.ndarray] = []
    print(f"Encoding {num_batches} batches for latent space analysis...")
    for i, (images, _) in enumerate(tqdm(data_loader, desc="Latent stats")):
        if i >= num_batches:
            break
        images = images.to(device)
        z, _ = model.encode(images)
        z_np = z.cpu().numpy()
        if z_np.ndim > 2:
            z_np = z_np.reshape(z_np.shape[0], -1)
        all_latents.append(z_np)
    Z = np.concatenate(all_latents, axis=0)
    N, D = Z.shape
    print(f"Collected {N} latent vectors, dim={D}")
    mean_act = np.abs(Z).mean(axis=0); std_act = Z.std(axis=0)
    sparsity = np.mean(np.abs(Z) < sparsity_threshold, axis=1)
    l0 = np.sum(np.abs(Z) > sparsity_threshold, axis=1)
    l1 = np.abs(Z).sum(axis=1); l2 = np.sqrt((Z**2).sum(axis=1))
    print(f"  mean_sparsity={sparsity.mean():.4f}  mean_l0={l0.mean():.1f}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].bar(range(min(D, 256)), mean_act[:256])
    axes[0, 0].set_title("Mean |act| per dim (first 256)")
    axes[0, 1].bar(range(min(D, 256)), std_act[:256], color="darkorange")
    axes[0, 1].set_title("Std act per dim (first 256)")
    axes[0, 2].hist(sparsity, bins=50, edgecolor="black")
    axes[0, 2].axvline(sparsity.mean(), color="red", linestyle="--", label=f"Mean={sparsity.mean():.3f}")
    axes[0, 2].set_title("Sparsity per sample"); axes[0, 2].legend()
    axes[1, 0].hist(l0, bins=50, edgecolor="black")
    axes[1, 0].axvline(l0.mean(), color="red", linestyle="--", label=f"Mean={l0.mean():.1f}")
    axes[1, 0].set_title("L0 norm per sample"); axes[1, 0].legend()
    axes[1, 1].hist(l1, bins=50, edgecolor="black")
    axes[1, 1].axvline(l1.mean(), color="red", linestyle="--", label=f"Mean={l1.mean():.2f}")
    axes[1, 1].set_title("L1 norm per sample"); axes[1, 1].legend()
    axes[1, 2].hist(l2, bins=50, edgecolor="black")
    axes[1, 2].axvline(l2.mean(), color="red", linestyle="--", label=f"Mean={l2.mean():.2f}")
    axes[1, 2].set_title("L2 norm per sample"); axes[1, 2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "latent_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    np.savez(os.path.join(save_dir, "latent_statistics.npz"),
             mean_activation=mean_act, std_activation=std_act,
             sparsity_per_sample=sparsity, l0_norm=l0, l1_norm=l1, l2_norm=l2,
             mean_sparsity=float(sparsity.mean()), mean_l0=float(l0.mean()),
             mean_l1=float(l1.mean()), mean_l2=float(l2.mean()),
             latent_dim=D, num_samples=N)
    print(f"Latent space analysis saved to {save_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# B5. Partonomy sparsity suite
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_partonomy_sparsity(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_batches: int = 30,
    sparsity_threshold: float = 1e-6,
    ablation_images: int = 16,
    n_clusters: int = 8,
) -> None:
    prt_dir = os.path.join(save_dir, "partonomy_sparsity")
    os.makedirs(prt_dir, exist_ok=True)
    model.eval()
    all_latents: List[np.ndarray] = []
    hook_data: Dict[int, List[torch.Tensor]] = {}
    ablation_imgs: Optional[torch.Tensor] = None
    hooks = []
    for i, stage in enumerate(model.encoder.multi_taxon_stages):
        def _make_hook(idx):
            def _hook(m, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                hook_data.setdefault(idx, []).append(t.detach().cpu())
            return _hook
        hooks.append(stage.register_forward_hook(_make_hook(i)))

    print(f"Collecting activations over {num_batches} batches...")
    for b_idx, (images, _) in enumerate(tqdm(data_loader, desc="Partonomy data")):
        if b_idx >= num_batches:
            break
        images = images.to(device)
        z, _ = model.encode(images)
        z_np = z.cpu().numpy()
        if z_np.ndim > 2:
            z_np = z_np.reshape(z_np.shape[0], -1)
        all_latents.append(z_np)
        if ablation_imgs is None:
            ablation_imgs = images[:ablation_images]
    for h in hooks:
        h.remove()

    Z = np.concatenate(all_latents, axis=0)
    N, D = Z.shape
    stage_acts: Dict[int, np.ndarray] = {}
    for idx, act_list in hook_data.items():
        pooled = [a.mean(dim=(2, 3)).numpy() for a in act_list]
        stage_acts[idx] = np.concatenate(pooled, axis=0)
    print(f"  {N} samples, latent_dim={D}")
    binary = (np.abs(Z) > sparsity_threshold).astype(np.float32)

    # 1. Jaccard
    print("  [1/5] Jaccard...")
    rng = np.random.default_rng(0)
    n_pairs = min(3000, N * (N - 1) // 2)
    ia = rng.integers(0, N, n_pairs); ib = rng.integers(0, N, n_pairs)
    ib[ia == ib] = (ib[ia == ib] + 1) % N
    inter = (binary[ia] * binary[ib]).sum(axis=1)
    union = ((binary[ia] + binary[ib]) > 0).sum(axis=1).astype(np.float32)
    jaccard = np.where(union > 0, inter / union, 0.0)
    lifetime_sp = binary.mean(axis=0)
    dead = float((lifetime_sp < 0.01).mean())
    selective = float(((lifetime_sp >= 0.01) & (lifetime_sp < 0.2)).mean())
    dense = float((lifetime_sp >= 0.5).mean())
    moderate = 1.0 - dead - selective - dense
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(jaccard, bins=50, edgecolor="black", color="steelblue")
    axes[0].axvline(jaccard.mean(), color="red", linestyle="--", label=f"Mean={jaccard.mean():.3f}")
    axes[0].set_title("Activation Overlap"); axes[0].legend()
    axes[1].bar(range(min(D, 200)), sorted(lifetime_sp, reverse=True)[:200], color="darkorange")
    axes[1].set_title("Lifetime Sparsity (sorted)")
    axes[2].bar(["Dead", "Selective", "Moderate", "Dense"],
                [v * 100 for v in [dead, selective, moderate, dense]],
                color=["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"])
    axes[2].set_ylabel("% Features"); axes[2].set_title("Feature Activity Categories")
    plt.suptitle("1. Activation Overlap & Specialisation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(prt_dir, "01_jaccard_overlap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    np.savez(os.path.join(prt_dir, "01_jaccard_stats.npz"), jaccard=jaccard,
             lifetime_sparsity=lifetime_sp, dead_frac=dead, selective_frac=selective, dense_frac=dense)

    # 2. Selectivity
    print("  [2/5] Selectivity...")
    p = lifetime_sp.clip(1e-6, 1 - 1e-6)
    per_dim_ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    per_dim_max = np.abs(Z).max(axis=0); per_dim_mean = np.abs(Z).mean(axis=0)
    sel_idx = (per_dim_max - per_dim_mean) / (per_dim_max + per_dim_mean + 1e-8)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(per_dim_ent, bins=50, edgecolor="black", color="purple")
    axes[0, 0].axvline(per_dim_ent.mean(), color="red", linestyle="--", label=f"Mean={per_dim_ent.mean():.3f} bits")
    axes[0, 0].set_title("Per-Dim Activation Entropy"); axes[0, 0].legend()
    axes[0, 1].hist(sel_idx, bins=50, edgecolor="black", color="darkgreen")
    axes[0, 1].axvline(sel_idx.mean(), color="red", linestyle="--", label=f"Mean={sel_idx.mean():.3f}")
    axes[0, 1].set_title("Feature Selectivity Index"); axes[0, 1].legend()
    axes[1, 0].scatter(lifetime_sp, per_dim_ent, alpha=0.3, s=8, c="navy")
    axes[1, 0].set_title("Sparsity vs Entropy")
    axes[1, 1].axis("off")
    plt.suptitle("2. Feature Selectivity & Entropy", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(prt_dir, "02_feature_selectivity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    np.savez(os.path.join(prt_dir, "02_selectivity_stats.npz"),
             per_dim_entropy=per_dim_ent, selectivity_idx=sel_idx)

    # 3. Ablations
    print("  [3/5] Ablation...")
    if ablation_imgs is not None:
        ablation_imgs_d = ablation_imgs.to(device)
        z_orig, _ = model.encode(ablation_imgs_d)
        spatial = (z_orig.ndim == 4)
        base_recon, *_ = model(ablation_imgs_d)
        if base_recon.shape[2:] != ablation_imgs_d.shape[2:]:
            base_recon = F.interpolate(base_recon, size=ablation_imgs_d.shape[2:], mode="bilinear", align_corners=False)
        base_mse = ((ablation_imgs_d - base_recon)**2).mean(dim=(1, 2, 3))
        if spatial:
            unit_active = (z_orig.abs().mean(dim=(2, 3)) > sparsity_threshold).cpu().numpy()
        else:
            unit_active = (z_orig.abs() > sparsity_threshold).cpu().numpy()
        candidates = np.where(unit_active.any(axis=0))[0]
        if len(candidates) > 256:
            ma = (z_orig.abs().mean(dim=(2, 3)) if spatial else z_orig.abs()).mean(0).cpu().numpy()
            candidates = candidates[np.argsort(-ma[candidates])[:256]]
        n_units = z_orig.shape[1]
        importance = np.zeros(n_units)
        for d in candidates:
            z_abl = z_orig.clone(); z_abl[:, d] = 0.0
            recon_abl, _ = model.decode(z_abl)
            if recon_abl.shape[2:] != ablation_imgs_d.shape[2:]:
                recon_abl = F.interpolate(recon_abl, size=ablation_imgs_d.shape[2:], mode="bilinear", align_corners=False)
            delta = ((ablation_imgs_d - recon_abl)**2).mean(dim=(1, 2, 3))
            importance[d] = (delta - base_mse).clamp(min=0).mean().item()
        sorted_imp = np.sort(importance[importance > 0])[::-1]
        total_imp = sorted_imp.sum()
        cumulative = np.cumsum(sorted_imp) / (total_imp + 1e-12) if total_imp > 0 else np.zeros(len(sorted_imp))
        p50 = int(np.searchsorted(cumulative, 0.5)) + 1 if total_imp > 0 else 0
        p90 = int(np.searchsorted(cumulative, 0.9)) + 1 if total_imp > 0 else 0
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        show_n = min(50, len(sorted_imp))
        axes[0].bar(range(show_n), sorted_imp[:show_n], color="firebrick")
        axes[0].set_title(f"Top-{show_n} Ablation Importance")
        axes[1].hist(sorted_imp, bins=40, edgecolor="black", color="salmon")
        if len(sorted_imp):
            axes[1].axvline(sorted_imp.mean(), color="blue", linestyle="--", label=f"Mean={sorted_imp.mean():.5f}")
        axes[1].set_title("Importance Distribution"); axes[1].legend()
        if len(sorted_imp):
            axes[2].plot(range(1, len(sorted_imp)+1), cumulative * 100, color="darkblue")
            axes[2].axvline(p50, color="orange", linestyle="--", label=f"50% in top-{p50}")
            axes[2].axvline(p90, color="red",    linestyle="--", label=f"90% in top-{p90}")
        axes[2].set_title("Cumulative Importance"); axes[2].legend()
        plt.suptitle("3. Causal Ablation", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(prt_dir, "03_causal_ablations.png"), dpi=150, bbox_inches="tight")
        plt.close()
        np.savez(os.path.join(prt_dir, "03_ablation_importance.npz"),
                 unit_importance=importance, p50_units=p50, p90_units=p90)
        print(f"    50% importance in top-{p50};  90% in top-{p90}")

    # 4. Cross-stage dependency
    print("  [4/5] Cross-stage dependency...")
    stage_indices = sorted(stage_acts.keys())
    n_st = len(stage_indices)
    if n_st >= 2:
        max_ch = 64
        stage_bin: Dict[int, np.ndarray] = {}
        for idx in stage_indices:
            a = stage_acts[idx]
            thr = sparsity_threshold * np.abs(a).mean()
            stage_bin[idx] = (np.abs(a) > thr).astype(np.float32)
        n_pairs = n_st - 1
        fig, axes = plt.subplots(1, n_pairs, figsize=(8 * n_pairs, 7), squeeze=False)
        dep_sp_list = []
        for pi, (s1, s2) in enumerate(zip(stage_indices[:-1], stage_indices[1:])):
            B1 = stage_bin[s1][:, :min(stage_bin[s1].shape[1], max_ch)]
            B2 = stage_bin[s2][:, :min(stage_bin[s2].shape[1], max_ch)]
            sup1 = B1.sum(axis=0) + 1e-8
            M = (B1.T @ B2) / sup1[:, None]
            dep_sp = float((M < 0.1).mean()); dep_sp_list.append(dep_sp)
            ax = axes[0, pi]
            im = ax.imshow(M, aspect="auto", cmap="hot", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel(f"Stage {s2+1} channels"); ax.set_ylabel(f"Stage {s1+1} channels")
            ax.set_title(f"Stage {s1+1}\u2192{s2+1}  dep={dep_sp*100:.1f}%")
        plt.suptitle("4. Cross-Stage Sparsity Dependency", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(prt_dir, "04_cross_stage_dependency.png"), dpi=150, bbox_inches="tight")
        plt.close()
        np.savez(os.path.join(prt_dir, "04_dependency_stats.npz"), dep_sparsity_per_pair=np.array(dep_sp_list))

    # 5. Clustering
    print("  [5/5] Clustering...")
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA as _PCA
        from sklearn.metrics import silhouette_score
        _sklearn_ok = True
    except ImportError:
        _sklearn_ok = False; print("    sklearn not available; skipping.")
    if _sklearn_ok and N >= n_clusters * 5:
        n_pca = min(50, D, N - 1)
        X = Z if D <= n_pca else _PCA(n_components=n_pca).fit_transform(Z)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else 0.0
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        cluster_mean = np.zeros((n_clusters, D))
        for c in range(n_clusters):
            m = labels == c
            if m.sum() > 0:
                cluster_mean[c] = np.abs(Z[m]).mean(axis=0)
        exclusivity = cluster_mean.max(axis=0) / (cluster_mean.sum(axis=0) + 1e-8)
        n_show = min(D, 64)
        top_dims = np.argsort(-cluster_mean.mean(axis=0))[:n_show]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].bar(range(n_clusters), cluster_sizes, color="steelblue")
        axes[0].set_title(f"Cluster Sizes (Sil={sil:.3f})")
        im = axes[1].imshow(cluster_mean[:, top_dims], aspect="auto", cmap="viridis")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title(f"Cluster \xd7 Feature (top-{n_show})")
        axes[2].hist(exclusivity, bins=40, edgecolor="black", color="darkorange")
        axes[2].axvline(exclusivity.mean(), color="red", linestyle="--", label=f"Mean={exclusivity.mean():.3f}")
        axes[2].set_title("Per-Dim Cluster Exclusivity"); axes[2].legend()
        plt.suptitle("5. Latent Dimension Clustering", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(prt_dir, "05_dimension_clustering.png"), dpi=150, bbox_inches="tight")
        plt.close()
        np.savez(os.path.join(prt_dir, "05_cluster_stats.npz"),
                 labels=labels, cluster_sizes=cluster_sizes,
                 silhouette=sil, exclusivity=exclusivity)
        print(f"    Silhouette={sil:.4f}  mean exclusivity={exclusivity.mean():.4f}")
    print(f"Partonomy sparsity suite saved to {prt_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# C1. Gate distributions
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_gate_distributions(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    n_batches: int = 10,
) -> None:
    gate_dir = os.path.join(save_dir, "gate_distributions")
    os.makedirs(gate_dir, exist_ok=True)
    model.eval()
    n_stages = len(model.encoder.multi_taxon_stages)
    K = model.n_hierarchies
    stage_gates: List[List[torch.Tensor]] = [[] for _ in range(n_stages)]
    for i, (images, _) in enumerate(data_loader):
        if i >= n_batches:
            break
        images = images.to(device)
        _, enc_details = model.encoder(images, hard=False, return_details=True)
        for s_idx, stage_info in enumerate(enc_details["stages"]):
            stage_gates[s_idx].append(stage_info["gate_probs"].cpu())

    for s_idx in range(n_stages):
        all_gates = torch.cat(stage_gates[s_idx], dim=0)  # [N, K, H, W]
        marginal = all_gates.mean(dim=(0, 2, 3)).numpy()
        fig, ax = plt.subplots(figsize=(max(5, K * 0.8 + 1), 3))
        bars = ax.bar(range(K), marginal, color=plt.cm.tab10.colors[:K])
        ax.set_xlabel("Hierarchy"); ax.set_ylabel("Mean gate weight")
        ax.set_title(f"Stage {s_idx+1}: Mean gate probability per hierarchy")
        ax.set_xticks(range(K)); ax.set_xticklabels([f"H{k}" for k in range(K)])
        for b, v in zip(bars, marginal):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(gate_dir, f"stage{s_idx+1}_gate_marginal.png"), dpi=130, bbox_inches="tight")
        plt.close()

        spatial_mean = all_gates.mean(dim=0).numpy()
        vmax = float(spatial_mean.max()) or 1.0
        ncols = min(K, 8); nrows = (K + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5 + 0.5))
        axes_flat = np.array(axes).flatten()
        for k in range(K):
            im = axes_flat[k].imshow(spatial_mean[k], cmap="viridis", vmin=0, vmax=vmax)
            axes_flat[k].set_title(f"H{k}", fontsize=9)
            axes_flat[k].set_xticks([]); axes_flat[k].set_yticks([])
            plt.colorbar(im, ax=axes_flat[k], fraction=0.046, pad=0.04)
        for k in range(K, len(axes_flat)):
            axes_flat[k].set_visible(False)
        plt.suptitle(f"Stage {s_idx+1}: Mean gate spatial distribution", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(gate_dir, f"stage{s_idx+1}_gate_spatial.png"), dpi=130, bbox_inches="tight")
        plt.close()
    print(f"Gate distributions saved to {gate_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# C2. Cross-hierarchy cosine similarity
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_cross_hierarchy_similarity(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    n_batches: int = 10,
) -> None:
    sim_dir = os.path.join(save_dir, "cross_hierarchy_similarity")
    os.makedirs(sim_dir, exist_ok=True)
    model.eval()
    n_stages = len(model.encoder.multi_taxon_stages)
    K = model.n_hierarchies
    stage_sims: List[List[np.ndarray]] = [[] for _ in range(n_stages)]
    for i, (images, _) in enumerate(data_loader):
        if i >= n_batches:
            break
        images = images.to(device)
        _, enc_details = model.encoder(images, hard=False, return_details=True)
        for s_idx, stage_info in enumerate(enc_details["stages"]):
            out_cat = stage_info["output"]
            chunks = out_cat.chunk(K, dim=1)
            vecs = [c.mean(dim=(2, 3)) for c in chunks]
            sim_mat = np.zeros((K, K), dtype=np.float32)
            for ki in range(K):
                for kj in range(K):
                    sim_mat[ki, kj] = float(F.cosine_similarity(vecs[ki], vecs[kj], dim=1).mean().item())
            stage_sims[s_idx].append(sim_mat)
    for s_idx in range(n_stages):
        mean_sim = np.mean(stage_sims[s_idx], axis=0)
        fig, ax = plt.subplots(figsize=(max(4, K * 0.8 + 1), max(4, K * 0.8 + 1)))
        im = ax.imshow(mean_sim, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels([f"H{k}" for k in range(K)])
        ax.set_yticklabels([f"H{k}" for k in range(K)])
        for ki in range(K):
            for kj in range(K):
                ax.text(kj, ki, f"{mean_sim[ki,kj]:.2f}", ha="center", va="center",
                        fontsize=8, color="k" if abs(mean_sim[ki, kj]) < 0.6 else "w")
        ax.set_title(f"Stage {s_idx+1}: Mean cross-hierarchy cosine sim\n(gated outputs, avg-pooled)")
        plt.tight_layout()
        plt.savefig(os.path.join(sim_dir, f"stage{s_idx+1}_cross_hier_cosine_sim.png"), dpi=130, bbox_inches="tight")
        plt.close()
    print(f"Cross-hierarchy similarity saved to {sim_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# C3. Per-stage regularisation bar charts
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_per_stage_regs(
    model: MultiTaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    n_batches: int = 10,
) -> None:
    model.eval()
    n_stages = len(model.encoder.multi_taxon_stages)
    sums = {k: np.zeros(n_stages) for k in ("entropy", "dkl", "gate_entropy", "gate_dkl")}
    count = 0
    for i, (images, _) in enumerate(data_loader):
        if i >= n_batches:
            break
        count += 1
        images = images.to(device)
        _, enc_details = model.encoder(images, hard=False, return_details=True)
        for s_idx, stage_info in enumerate(enc_details["stages"]):
            for k in sums:
                sums[k][s_idx] += float(stage_info[k].item())
    if count > 0:
        for k in sums:
            sums[k] /= count
    x = np.arange(n_stages); labels = [f"Stage {i+1}" for i in range(n_stages)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (key, title) in zip(axes, [
        ("entropy",      "Within-hierarchy Entropy"),
        ("dkl",          "Within-hierarchy DKL"),
        ("gate_entropy", "Gate Entropy"),
        ("gate_dkl",     "Gate DKL"),
    ]):
        bars = ax.bar(x, sums[key], color=plt.cm.tab10.colors[:n_stages])
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title, fontsize=10); ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, sums[key]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)
    plt.suptitle("Per-stage regularisation statistics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "per_stage_regs.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# arg parsing & main
# ══════════════════════════════════════════════════════════════════════════════


def _is_cifar(cfg: dict) -> bool:
    """Infer dataset type from the config (image_size==32 -> CIFAR-10)."""
    return cfg.get("data", {}).get("image_size", 256) == 32


def _is_imagenet(cfg: dict) -> bool:
    """Infer ImageNet from config (image_size==224 and data_root contains 'imagenet')."""
    dc = cfg.get("data", {})
    return dc.get("image_size", 256) == 224 or "imagenet" in dc.get("data_root", "")

def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args()
    cfg: dict = {}
    if pre_args.config:
        cfg = load_config(pre_args.config)
    d = cfg.get("data", {})
    m = cfg.get("model", {})
    t = cfg.get("training", {})
    o = cfg.get("output", {})
    a = cfg.get("analysis", {})
    parser = argparse.ArgumentParser(description="Analyze MultiTaxon AE on CelebA-HQ")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Override checkpoint path; derived from output_dir+run_suffix if omitted")
    parser.add_argument("--output-dir", type=str,
                        default=o.get("output_dir", "./outputs/multi_taxon_ae_celeba_hq_r18"))
    parser.add_argument("--analysis-save-dir", type=str, default="",
                        help="Override analysis dir; derived from output_dir+run_suffix if omitted")
    parser.add_argument("--data-root", type=str,       default=d.get("data_root", "./data/celeba_hq"))
    parser.add_argument("--image-size", type=int,      default=d.get("image_size", 256))
    parser.add_argument("--batch-size", type=int,      default=d.get("batch_size", 16))
    parser.add_argument("--num-workers", type=int,     default=d.get("num_workers", 4))
    parser.add_argument("--val-split", type=float,     default=d.get("val_split", 0.05))
    parser.add_argument("--dkl-weight", type=float,    default=t.get("dkl_weight", 1e-2))
    parser.add_argument("--temperature", type=float,   default=m.get("temperature", 1.0))
    parser.add_argument("--hard", action="store_true", default=m.get("hard", False))
    parser.add_argument("--n-hierarchies", type=int,   default=m.get("n_hierarchies", 3))
    parser.add_argument("--entropy-weight", type=float,      default=t.get("entropy_weight", 0.0))
    parser.add_argument("--gate-dkl-weight", type=float,     default=t.get("gate_dkl_weight", 1e-2))
    parser.add_argument("--gate-entropy-weight", type=float, default=t.get("gate_entropy_weight", 0.0))
    parser.add_argument("--n-latent-batches",    type=int, default=a.get("num_latent_batches", 50))
    parser.add_argument("--n-recon-batches",     type=int, default=a.get("num_reconstruction_batches", 20))
    parser.add_argument("--n-analysis-batches",  type=int, default=a.get("num_taxonomy_batches", 10))
    parser.add_argument("--n-recon-images",      type=int, default=a.get("num_multiple_recon_images", 8))
    parser.add_argument("--n-recon-sets",        type=int, default=a.get("num_reconstructions_per_image", 8))
    parser.add_argument("--n-act-images",        type=int, default=a.get("num_activation_images", 3))
    parser.add_argument("--n-parse-images",      type=int, default=a.get("num_parse_tree_images", 4))
    parser.add_argument("--n-hier-images",       type=int, default=a.get("num_hier_act_images", 4))
    parser.add_argument("--max-hier-depth",      type=int, default=a.get("max_hier_act_depth", 4))
    parser.add_argument("--n-split-images",      type=int, default=a.get("num_split_map_images", 4))
    parser.add_argument("--max-split-pairs",     type=int, default=a.get("max_split_pairs", 8))
    parser.add_argument("--seed", type=int, default=t.get("seed", 42))
    parser.add_argument("--skip-partonomy", action="store_true",
                        help="Skip the partonomy sparsity suite (section B5).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dkl_suffix     = f"_dkl_{args.dkl_weight:.0e}"
    temp_str       = f"{args.temperature:g}".replace(".", "p")
    temp_suffix    = f"_temp_{temp_str}"
    hard_suffix    = "_hard" if args.hard else ""
    hier_suffix    = f"_K{args.n_hierarchies}"
    entropy_suffix = f"_ew_{args.entropy_weight:.0e}"      if args.entropy_weight      else ""
    gdkl_suffix    = f"_gdkl_{args.gate_dkl_weight:.0e}"   if args.gate_dkl_weight     else ""
    gew_suffix     = f"_gew_{args.gate_entropy_weight:.0e}" if args.gate_entropy_weight else ""
    run_suffix = (dkl_suffix + temp_suffix + hard_suffix + hier_suffix
                  + entropy_suffix + gdkl_suffix + gew_suffix)

    output_dir   = Path(args.output_dir + run_suffix)
    analysis_dir = Path(args.analysis_save_dir or str(output_dir / "analysis"))
    analysis_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config) if args.config else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("MultiTaxon Autoencoder — Comprehensive Analysis")
    print("=" * 80)
    print(f"  device={device}  run_suffix={run_suffix}")
    print(f"  output_dir={output_dir}")
    print(f"  analysis_dir={analysis_dir}")
    print("=" * 80)

    # B1. Training curves
    print("\n[B1] Training curves")
    history_path = output_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        save_training_curves(history, str(analysis_dir))
    else:
        print(f"  No training_history.json at {history_path}, skipping.")

    # Load model
    ckpt = args.checkpoint or str(output_dir / "checkpoints" / "best.pt")
    model, _ = load_model(ckpt, device, config)
    vanilla = _has_vanilla_taxonomy(model)
    if not vanilla:
        print("  (Non-vanilla variant detected — taxonomy-specific analyses will be skipped)")

    if _is_cifar(config):
        data_loader = CIFAR10Loader(batch_size=args.batch_size, root=args.data_root)
        _, val_loader = data_loader.get_loaders()
    elif _is_imagenet(config) or args.image_size == 224 or "imagenet" in str(args.data_root):
        tf = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        data_loader = ImageNet1kHFLoader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            pin_memory=(device.type == "cuda"),
            transform=tf,
            max_train_samples=1,
        )
        _, val_loader = data_loader.get_loaders()
    else:
        tf = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        data_loader = CelebAHQLoader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            val_split=args.val_split,
            seed=args.seed,
            pin_memory=(device.type == "cuda"),
            transform=tf,
        )
        _, val_loader = data_loader.get_loaders()
        if val_loader is None:
            raise RuntimeError("val_split must be > 0 to produce a validation loader")

    save_dir = str(analysis_dir)

    print("\n[A1] Stage filter visualisation")
    visualize_stage_filters(model, save_dir)

    print("\n[A1b] Filter similarity analysis")
    analyze_filter_similarity(model, save_dir)

    if vanilla:
        print("\n[A2] Taxonomy regularisation distributions")
        visualize_taxonomy_distributions(model, val_loader, device, save_dir,
                                         num_batches=args.n_analysis_batches)

        print("\n[A3] Path probability plots")
        visualize_taxonomy_path_probs(model, val_loader, device, save_dir)

        print("\n[A4] Taxonomy tree activations")
        visualize_taxonomy_tree(model, val_loader, device, save_dir,
                                num_images=args.n_act_images)

        print("\n[A5] Parse-tree analysis")
        analyze_parse_trees(model, val_loader, device, save_dir,
                            num_images=args.n_parse_images)

        print("\n[A6] Hierarchical activation maps")
        analyze_hierarchical_activations(model, val_loader, device, save_dir,
                                         num_images=args.n_hier_images,
                                         max_depth=args.max_hier_depth)

        print("\n[A7] Binary split maps")
        analyze_binary_split_maps(model, val_loader, device, save_dir,
                                  num_images=args.n_split_images,
                                  max_depth=args.max_hier_depth,
                                  max_pairs=args.max_split_pairs)
    else:
        print("\n[A2-A7] Skipped (taxonomy-specific, not applicable to this variant)")

    print("\n[A8] Stage activation maps")
    visualize_stage_activations(model, val_loader, device, save_dir,
                                num_images=args.n_act_images)

    print("\n[B1] Skip-connection decomposition")
    analyze_skip_decomposition(model, val_loader, device, save_dir,
                               num_images=args.n_recon_images,
                               num_sets=args.n_recon_sets)

    print("\n[B2] Reconstruction quality")
    analyze_reconstruction_quality(model, val_loader, device, save_dir,
                                   num_batches=args.n_recon_batches)

    print("\n[B3] Multiple reconstructions")
    visualize_multiple_reconstructions(model, val_loader, device, save_dir,
                                       num_images=args.n_recon_images,
                                       num_sets=args.n_recon_sets)

    print("\n[B4] Latent space sparsity")
    analyze_latent_sparsity(model, val_loader, device, save_dir,
                            num_batches=args.n_latent_batches)

    print("\n[B5] Partonomy sparsity suite")
    if args.skip_partonomy:
        print("  Skipped (--skip-partonomy).")
    else:
        analyze_partonomy_sparsity(model, val_loader, device, save_dir)

    if vanilla:
        print("\n[C1] Gate distributions")
        analyze_gate_distributions(model, val_loader, device, save_dir,
                                   n_batches=args.n_analysis_batches)
    else:
        print("\n[C1] Skipped (vanilla-only gate distributions)")

    print("\n[C2] Cross-hierarchy cosine similarity")
    analyze_cross_hierarchy_similarity(model, val_loader, device, save_dir,
                                       n_batches=args.n_analysis_batches)

    if vanilla:
        print("\n[C3] Per-stage regularisation stats")
        analyze_per_stage_regs(model, val_loader, device, save_dir,
                               n_batches=args.n_analysis_batches)
    else:
        print("\n[C3] Skipped (vanilla-only per-stage regularisation)")

    print("\n" + "=" * 80)
    print(f"Analysis complete! All outputs saved to: {analysis_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
