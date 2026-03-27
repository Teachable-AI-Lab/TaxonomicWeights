"""Analysis script for the ResNet-based Taxon Autoencoder.

Runs the following analyses and saves results to ``output.analysis_save_dir``:

1.  Encoder stage filter visualization (residual block weights per stage).
2.  Taxonomy probability distributions per encoder stage.
3.  Latent space sparsity & norm statistics.
4.  Reconstruction quality metrics (MSE / MAE).
5.  Multiple reconstruction comparisons.
6.  Partonomy sparsity suite (Jaccard, selectivity, ablation, cross-stage,
    clustering).
7.  Encoder stage activation maps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.taxon.taxon_ae import TaxonAutoencoder
from src.model.cnn.taxon.topk_taxon_ae import TopKTaxonAutoencoder
from src.model.cnn.taxon.bias_taxon_ae import BiasTaxonAutoencoder
from src.utils.dataloader import CelebAHQLoader, CIFAR10Loader
from torchvision import transforms


def _detect_taxon_variant(state_dict: dict) -> str:
    """Auto-detect taxon model variant from state_dict keys."""
    for k in state_dict:
        if '_steps_since_active' in k:
            return 'topk'
        if '_bias' in k and 'taxon_stages' in k:
            return 'bias'
    return 'vanilla'


def _has_vanilla_taxonomy(model) -> bool:
    """Return True if model stages support vanilla taxonomy methods."""
    stage = model.encoder.taxon_stages[0]
    return hasattr(stage, '_taxon_logits_per_depth')


def _is_cifar(cfg: dict) -> bool:
    """Infer dataset type from the config (image_size==32 -> CIFAR-10)."""
    return cfg.get("data", {}).get("image_size", 256) == 32


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _to_display(tensor: torch.Tensor, normalized: bool = True) -> np.ndarray:
    """Convert a model tensor (1, C, H, W) to a displayable (H, W, C) image.

    If ``normalized`` is True the tensor is assumed to be in [-1, 1]
    (trained with Normalize mean=0.5 std=0.5) and is mapped back to [0, 1].
    Otherwise it is assumed to already be in [0, 1].
    """
    t = tensor.detach().cpu()
    if normalized:
        t = t * 0.5 + 0.5
    return t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()


def load_model(
    checkpoint_path: str,
    device: torch.device,
    config: dict,
) -> Tuple[TaxonAutoencoder, dict]:
    """Instantiate TaxonAutoencoder from config and load checkpoint weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path} ({os.path.getsize(checkpoint_path)/1e6:.1f} MB)")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    mc = config.get("model", {})
    state = checkpoint.get("model_state", checkpoint)
    variant = mc.get("model_variant", _detect_taxon_variant(state))
    # Normalise: "topk_taxon" → "topk", "bias_taxon" → "bias"
    variant = variant.replace("_taxon", "").replace("_multi", "")

    common_kw = dict(
        in_channels=mc.get("in_channels", 3),
        resnet_variant=mc.get("resnet_variant", "18"),
        stage_taxonomy_layers=tuple(mc.get("stage_taxonomy_layers", [5, 6, 7, 8])),
        stage_strides=tuple(mc.get("stage_strides", [1, 2, 2, 2])),
        stage_blocks=mc.get("stage_blocks", None),
        kernel_size=mc.get("kernel_size", 3),
        use_stem=mc.get("use_stem", True),
        stem_channels=mc.get("stem_channels", 64),
        stem_stride=mc.get("stem_stride", 2),
        use_stem_maxpool=mc.get("use_stem_maxpool", True),
        output_activation=mc.get("output_activation", "none"),
        depth_decay=mc.get("depth_decay", 0.5),
    )

    if variant == 'topk':
        model = TopKTaxonAutoencoder(
            **common_kw,
            k_aux=mc.get("k_aux", None),
            dead_steps=mc.get("dead_steps", 2000),
            temperature=mc.get("temperature", 1.0),
            hard=mc.get("hard", False),
        )
    elif variant == 'bias':
        model = BiasTaxonAutoencoder(
            **common_kw,
            bias_update_rate=mc.get("bias_update_rate", 0.001),
            bias_ema_decay=mc.get("bias_ema_decay", 0.99),
            temperature=mc.get("temperature", 1.0),
            hard=mc.get("hard", False),
        )
    else:
        model = TaxonAutoencoder(
            **common_kw,
            temperature=mc.get("temperature", 1.0),
            hard=mc.get("hard", False),
        )

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    if "train_stats" in checkpoint:
        tl = checkpoint['train_stats'].get('loss', '?')
        print(f"  train_loss={tl:.6f}" if isinstance(tl, (int, float)) else f"  train_loss={tl}")
    if "val_stats" in checkpoint:
        vl = checkpoint['val_stats'].get('loss', '?')
        print(f"  val_loss={vl:.6f}" if isinstance(vl, (int, float)) else f"  val_loss={vl}")
    return model, checkpoint


# ── 1. Stage filter visualization ────────────────────────────────────────────

def visualize_stage_filters(model: TaxonAutoencoder, save_dir: str, n_cols: int = 8) -> None:
    """Save filter grids for the first residual block in each encoder stage."""
    stage_dir = os.path.join(save_dir, "stage_filters")
    os.makedirs(stage_dir, exist_ok=True)

    for s_idx, stage in enumerate(model.encoder.taxon_stages, start=1):
        first_block = stage.blocks[0]
        conv = next((m for m in first_block.main.modules() if isinstance(m, nn.Conv2d)), None)
        if conv is None:
            print(f"  Stage {s_idx}: no Conv2d found, skipping.")
            continue

        w = conv.weight.detach().cpu().numpy()          # (out, in, k, k)
        out_ch, in_ch, k, _ = w.shape
        w_disp = w.mean(axis=1)                         # avg input channels
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
            f"Stage {s_idx} first-block conv1 filters "
            f"({n_show}/{out_ch} shown, avg of {in_ch} in-ch, {k}x{k})",
            fontsize=10,
        )
        plt.tight_layout()
        out_path = os.path.join(stage_dir, f"stage{s_idx}_filters.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Stage {s_idx}: {n_show}/{out_ch} filters -> {out_path}")


# ── 1b. Filter similarity analysis ──────────────────────────────────────────

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


def analyze_filter_similarity(model: TaxonAutoencoder, save_dir: str) -> None:
    """Pairwise cosine-similarity analysis of all and leaf filters at each stage.

    For every encoder stage this produces:
      filter_similarity/stage{S}/block{B}_{conv}_simmat.png  – per-Conv2d sim matrix
      filter_similarity/stage{S}/leaf_simmat.png             – leaf channels only
      filter_similarity/stage{S}/cross_block_simmat.png      – cross-block output convs
      filter_similarity/stage{S}/output_vs_leaf_hist.png     – histogram comparison
    Plus a cross-stage summary bar chart and a stats npz.
    """
    base_dir = os.path.join(save_dir, "filter_similarity")
    os.makedirs(base_dir, exist_ok=True)

    n_stages = len(model.encoder.taxon_stages)
    stage_stats: Dict[int, dict] = {}

    for s_idx, stage in enumerate(model.encoder.taxon_stages, start=1):
        stage_dir = os.path.join(base_dir, f"stage{s_idx}")
        os.makedirs(stage_dir, exist_ok=True)

        n_leaf = stage.layer_channels[-1]   # 2^L — deepest taxonomy depth
        total  = stage.total_out_channels   # (2^(L+1)) - 2

        # ── per-block, per-Conv2d sim matrices ─────────────────────────────
        output_convs: List[np.ndarray] = []  # last Conv2d of each block
        for b_idx, block in enumerate(stage.blocks):
            convs_in_block = [(name, m) for name, m in block.main.named_modules()
                              if isinstance(m, nn.Conv2d)]
            for c_name, conv in convs_in_block:
                W = _normalized_filter_matrix(conv)
                sim = W @ W.T
                leaf_start = total - n_leaf if W.shape[0] == total else -1
                _save_simmat(
                    sim,
                    title=(f"Stage {s_idx} block{b_idx} conv{c_name}\n"
                           f"({W.shape[0]} filters, {W.shape[1]} dims)"),
                    path=os.path.join(stage_dir, f"block{b_idx}_conv{c_name}_simmat.png"),
                    leaf_start=leaf_start,
                )
            if convs_in_block:
                output_convs.append(_normalized_filter_matrix(convs_in_block[-1][1]))

        # ── cross-block: last Conv2d of each block vs every other block ────
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
            plt.suptitle(f"Stage {s_idx}: cross-block output-conv cosine sim",
                         fontsize=10, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(stage_dir, "cross_block_simmat.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()

        # ── leaf filters from last block's last Conv2d ─────────────────────
        last_W   = output_convs[-1]           # (total, dim)
        leaf_W   = last_W[total - n_leaf:]    # (n_leaf, dim)
        sim_leaf = leaf_W @ leaf_W.T
        _save_simmat(
            sim_leaf,
            title=(f"Stage {s_idx}: leaf filter cosine sim\n"
                   f"({n_leaf} leaf channels, L={stage.n_taxonomy_layers})"),
            path=os.path.join(stage_dir, "leaf_simmat.png"),
        )

        # ── histogram: all output filters vs leaf filters ──────────────────
        sim_all   = last_W @ last_W.T
        mask_all  = ~np.eye(total,  dtype=bool)
        mask_leaf = ~np.eye(n_leaf, dtype=bool)
        vals_all  = sim_all[mask_all]
        vals_leaf = sim_leaf[mask_leaf]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(vals_all,  bins=60,             alpha=0.6, density=True, color="steelblue",
                label=f"All output filters ({total}ch)  mean={vals_all.mean():.3f}")
        ax.hist(vals_leaf, bins=max(10, n_leaf), alpha=0.6, density=True, color="tomato",
                label=f"Leaf filters ({n_leaf}ch)  mean={vals_leaf.mean():.3f}")
        ax.axvline(vals_all.mean(),  color="steelblue", linestyle="--", linewidth=1.5)
        ax.axvline(vals_leaf.mean(), color="tomato",    linestyle="--", linewidth=1.5)
        ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Density")
        ax.set_title(f"Stage {s_idx}: pairwise filter cosine similarity distribution")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(stage_dir, "output_vs_leaf_hist.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        stage_stats[s_idx] = dict(
            mean_sim_all=float(vals_all.mean()),   std_sim_all=float(vals_all.std()),
            mean_sim_leaf=float(vals_leaf.mean()), std_sim_leaf=float(vals_leaf.std()),
            n_filters=int(total), n_leaf=int(n_leaf),
        )
        print(f"  Stage {s_idx}: all={total} filters (mean_sim={vals_all.mean():.3f}), "
              f"leaf={n_leaf} (mean_sim={vals_leaf.mean():.3f})")

    # ── cross-stage summary ────────────────────────────────────────────────
    labels = [f"S{i+1}" for i in range(n_stages)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(labels, [stage_stats[i+1]["mean_sim_all"]  for i in range(n_stages)],
                color="steelblue", edgecolor="black", linewidth=0.5)
    axes[0].set_title("Mean pairwise sim — all output filters")
    axes[0].set_ylabel("Mean cosine similarity"); axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(labels, [stage_stats[i+1]["mean_sim_leaf"] for i in range(n_stages)],
                color="tomato", edgecolor="black", linewidth=0.5)
    axes[1].set_title("Mean pairwise sim — leaf filters only")
    axes[1].set_ylabel("Mean cosine similarity"); axes[1].grid(axis="y", alpha=0.3)
    plt.suptitle("Filter Similarity Summary — All Stages", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "summary_mean_sim.png"), dpi=150, bbox_inches="tight")
    plt.close()

    np.savez(
        os.path.join(base_dir, "filter_similarity_stats.npz"),
        **{f"stage{s}_{k}": np.array(v) for s, d in stage_stats.items() for k, v in d.items()},
    )
    print(f"Filter similarity analysis saved to {base_dir}")


# ── 2. Taxonomy probability distributions ────────────────────────────────────

def visualize_taxonomy_distributions(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_batches: int = 10,
) -> None:
    """Plot per-stage mean entropy & DKL over several batches."""
    dist_dir = os.path.join(save_dir, "taxonomy_distributions")
    os.makedirs(dist_dir, exist_ok=True)

    n_stages = len(model.encoder.taxon_stages)
    entropy_per_stage: List[List[float]] = [[] for _ in range(n_stages)]
    dkl_per_stage: List[List[float]] = [[] for _ in range(n_stages)]

    model.eval()
    with torch.no_grad():
        for b_idx, (images, _) in enumerate(tqdm(data_loader, desc="Taxonomy stats")):
            if b_idx >= num_batches:
                break
            images = images.to(device)
            _, details = model.encode(images, return_details=True)
            for s_idx, stage_info in enumerate(details["stages"]):
                entropy_per_stage[s_idx].append(float(stage_info["entropy"].cpu()))
                dkl_per_stage[s_idx].append(float(stage_info["dkl"].cpu()))

    fig, axes = plt.subplots(2, n_stages, figsize=(4 * n_stages, 6))
    if n_stages == 1:
        axes = axes.reshape(2, 1)

    for s_idx in range(n_stages):
        stage = model.encoder.taxon_stages[s_idx]
        axes[0, s_idx].plot(entropy_per_stage[s_idx], marker="o", ms=3)
        axes[0, s_idx].set_title(f"Stage {s_idx+1} entropy\n(depth={stage.n_taxonomy_layers})")
        axes[0, s_idx].set_xlabel("batch")
        axes[1, s_idx].plot(dkl_per_stage[s_idx], marker="o", ms=3, color="tab:orange")
        axes[1, s_idx].set_title(f"Stage {s_idx+1} DKL\n(depth={stage.n_taxonomy_layers})")
        axes[1, s_idx].set_xlabel("batch")

    plt.suptitle("Per-stage Taxonomy Regularization Over Batches", fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(dist_dir, "taxonomy_regularization.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    np.savez(
        os.path.join(dist_dir, "taxonomy_stats.npz"),
        **{f"stage{i+1}_entropy": np.array(entropy_per_stage[i]) for i in range(n_stages)},
        **{f"stage{i+1}_dkl":    np.array(dkl_per_stage[i])    for i in range(n_stages)},
    )
    print(f"Taxonomy distributions saved to {dist_dir}")


# ── 2b. Path probability plots per stage ─────────────────────────────────────

def _get_encoder_details(model: TaxonAutoencoder, images: torch.Tensor) -> List[dict]:
    """Run encoder with return_details=True and return the stages list."""
    with torch.no_grad():
        _, enc_details = model.encoder(images, return_details=True)
    return enc_details["stages"]


def visualize_taxonomy_path_probs(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_patches: int = 20,
    sample_index: int = 0,
    alpha: float = 0.6,
) -> None:
    """For each encoder stage, plot path probability distributions over patches.

    A separate plot per stage mirrors the notebook's ``plot_patch_path_probabilities``
    but saves to disk.  The left panel shows each patch's probability curve across
    all flattened tree channels (depth-separated by dashed boundaries).  The right
    panel shows the per-depth path-sum heat-map (should be ~1 everywhere).
    """
    prob_dir = os.path.join(save_dir, "path_probability_plots")
    os.makedirs(prob_dir, exist_ok=True)

    model.eval()
    images, _ = next(iter(data_loader))
    images = images.to(device)

    stages = _get_encoder_details(model, images)

    for stage_info in stages:
        name = stage_info["name"]
        logp_all = stage_info["logp"]          # (B, total_C, H, W)
        layer_channels = stage_info["layer_channels"]

        # Take one sample, convert to prob, split by depth
        logp_s = logp_all[sample_index:sample_index + 1]   # (1, total_C, H, W)
        prob_splits = [torch.exp(t) for t in torch.split(logp_s, layer_channels, dim=1)]

        # Flatten into (n_patches, total_C) and layer sum (n_patches, n_depths)
        flattened = torch.cat(prob_splits, dim=1)           # (1, total_C, H, W)
        layer_sums = torch.stack(
            [p.sum(dim=1) for p in prob_splits], dim=1
        )                                                    # (1, n_depths, H, W)

        patch_probs = flattened[0].permute(1, 2, 0).reshape(-1, flattened.shape[1])   # (n_patches, C)
        depth_sums  = layer_sums[0].permute(1, 2, 0).reshape(-1, len(layer_channels)) # (n_patches, L)

        n_show = min(num_patches, patch_probs.shape[0])
        patch_probs = patch_probs[:n_show].detach().cpu().numpy()
        depth_sums  = depth_sums[:n_show].detach().cpu().numpy()

        total_C = patch_probs.shape[1]
        x_full = np.arange(total_C)
        boundaries = (np.cumsum(layer_channels)[:-1] - 0.5).tolist()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                 gridspec_kw={"width_ratios": [2.4, 1.2]})

        ax = axes[0]
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

        ax.set_xlabel("Flattened tree channel index")
        ax.set_ylabel("Path probability")
        ax.set_title(f"{name} path distributions (first {n_show} patches)")

        ax2 = axes[1]
        im = ax2.imshow(depth_sums.T, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax2.set_xlabel("Patch index")
        ax2.set_ylabel("Layer (depth)")
        ax2.set_yticks(np.arange(len(layer_channels)))
        ax2.set_yticklabels([f"L{i+1}" for i in range(len(layer_channels))])
        ax2.set_title("Per-layer path sum (should be 1)")
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        err = np.abs(depth_sums - 1.0).max()
        ax2.set_xlabel(f"Patch index  [max|sum-1|={err:.2e}]")

        plt.tight_layout()
        out_path = os.path.join(prob_dir, f"{name}_path_probs.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {name}: path prob plot -> {out_path}  (max depth-sum err={err:.2e})")

    print(f"Path probability plots saved to {prob_dir}")


# ── 2c. Hierarchical tree activation visualization ────────────────────────────

def visualize_taxonomy_tree(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 2,
    max_tree_depths: int = 3,
) -> None:
    """Draw the binary-tree activation structure for each encoder stage.

    For each (image, stage) pair, a matplotlib figure is produced showing the
    binary tree of taxonomy nodes up to ``max_tree_depths`` (default 3 → 14
    nodes: 2 + 4 + 8).  Each node shows:
    - Background colour: mean activation magnitude of that depth/channel at the
      selected spatial patch (centre of the feature map).
    - Text: the node's channel index within its depth and its mean |activation|.

    Edges are drawn from parent to its two children; edge width is proportional
    to the conditional routing probability of selecting that child (computed from
    the joint log-probs: P(child | parent) ≈ joint_prob[child] / joint_prob[parent]).
    """
    tree_dir = os.path.join(save_dir, "taxonomy_tree_activations")
    os.makedirs(tree_dir, exist_ok=True)

    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)

    stages = _get_encoder_details(model, images)

    for img_idx in range(num_images):
        for stage_info in stages:
            name = stage_info["name"]
            output_all = stage_info["output"]        # (B, total_C, H, W)
            logp_all   = stage_info["logp"]          # (B, total_C, H, W)
            layer_channels: List[int] = stage_info["layer_channels"]

            n_depths_show = min(max_tree_depths, len(layer_channels))
            shown_channels = layer_channels[:n_depths_show]

            # Pick centre spatial location
            H, W = output_all.shape[2], output_all.shape[3]
            cy, cx = H // 2, W // 2

            # Split output and logp by depth, then select centre patch
            out_splits  = list(torch.split(output_all[img_idx], layer_channels, dim=0))  # per depth: (C_d, H, W)
            logp_splits = list(torch.split(logp_all[img_idx],   layer_channels, dim=0))

            # Centre-patch activation and prob per depth
            acts  = [out_splits[d][:, cy, cx].detach().cpu().numpy()  for d in range(n_depths_show)]
            probs = [logp_splits[d][:, cy, cx].exp().detach().cpu().numpy() for d in range(n_depths_show)]

            # Compute max act for colour normalisation
            all_acts = np.concatenate(acts)
            vmax = max(np.abs(all_acts).max(), 1e-8)

            # Tree layout: node i at depth d → x in (0,1], y = -d
            def node_pos(depth: int, idx: int) -> Tuple[float, float]:
                n = layer_channels[depth]
                return (idx + 0.5) / n, -depth

            fig, ax = plt.subplots(1, 1, figsize=(max(8, 2 * shown_channels[-1]), (n_depths_show + 1) * 2.5 + 1))
            ax.set_xlim(0, 1)
            ax.set_ylim(-n_depths_show, 1.5)
            ax.set_aspect("auto")
            ax.axis("off")
            ax.set_title(
                f"{name}  —  image {img_idx+1}  —  centre patch ({cy},{cx})\n"
                f"Node colour = activation magnitude, edge width ∝ conditional prob",
                fontsize=9,
            )

            cmap = plt.cm.RdYlGn
            node_radius = 0.012

            # Virtual root node sits above depth 0 (depth-0 has 2 channels; we
            # add an implicit root so the diagram reads as one connected tree).
            vroot_x, vroot_y = 0.5, 1.0
            ax.add_patch(plt.Circle((vroot_x, vroot_y), node_radius * 1.8,
                                    color="lightgray", zorder=2,
                                    linewidth=0.5, edgecolor="black"))
            ax.text(vroot_x, vroot_y - node_radius * 2.8, "root",
                    ha="center", va="top", fontsize=5.5, zorder=3, color="dimgray")

            # Edges from virtual root to each depth-0 node
            for i in range(layer_channels[0]):
                x_c, y_c = node_pos(0, i)
                ax.plot([vroot_x, x_c], [vroot_y, y_c],
                        color="steelblue", linewidth=1.0, alpha=0.5, zorder=1)

            # Draw remaining edges (depth ≥ 1 → parent at previous depth)
            for d in range(1, n_depths_show):
                n_nodes = layer_channels[d]
                for i in range(n_nodes):
                    parent_i = i // 2
                    x_c, y_c = node_pos(d, i)
                    x_p, y_p = node_pos(d - 1, parent_i)

                    # Conditional prob = joint_prob[child] / joint_prob[parent]
                    joint_child  = float(probs[d][i])
                    joint_parent = float(probs[d - 1][parent_i])
                    cond_prob = joint_child / max(joint_parent, 1e-8)
                    cond_prob = min(cond_prob, 1.0)

                    ax.plot(
                        [x_p, x_c], [y_p, y_c],
                        color="steelblue",
                        linewidth=0.5 + 3.5 * cond_prob,
                        alpha=0.4 + 0.5 * cond_prob,
                        zorder=1,
                    )

            # Draw nodes
            for d in range(n_depths_show):
                n_nodes = layer_channels[d]
                for i in range(n_nodes):
                    x_n, y_n = node_pos(d, i)
                    act_val = float(acts[d][i])
                    norm_val = (act_val + vmax) / (2 * vmax)    # [-vmax, vmax] → [0,1]
                    node_col = cmap(norm_val)

                    circle = plt.Circle(
                        (x_n, y_n), node_radius * (1.8 if n_nodes <= 4 else 1.0),
                        color=node_col, zorder=2, linewidth=0.5,
                        edgecolor="black",
                    )
                    ax.add_patch(circle)

                    prob_val = float(probs[d][i])
                    ax.text(
                        x_n, y_n - node_radius * 2.8,
                        f"c{i}\n{act_val:.2f}\np={prob_val:.2f}",
                        ha="center", va="top", fontsize=5.5, zorder=3,
                    )

                # Depth label on left
                ax.text(
                    0.002, -d, f"L{d+1}\n({n_nodes}ch)",
                    va="center", ha="left", fontsize=7, color="dimgray",
                )

            # Colour bar
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                        norm=plt.Normalize(vmin=-vmax, vmax=vmax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.015, pad=0.01,
                         label="Activation value")

            plt.tight_layout()
            out_path = os.path.join(
                tree_dir, f"img{img_idx+1:02d}_{name}_tree.png"
            )
            plt.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close()
            print(f"  img{img_idx+1} {name}: tree -> {out_path}")

    print(f"Taxonomy tree activations saved to {tree_dir}")


# ── 2d. Parse tree analysis and visualization ───────────────────────────────

def extract_parse_tree(
    model: TaxonAutoencoder,
    images: torch.Tensor,
    device: torch.device,
) -> List[list]:
    """Extract taxonomy parse-tree decisions for each sample in ``images``.

    For each stage and each depth, this records spatially pooled joint path
    probabilities, the selected node, decoded path bits/labels, and the
    per-depth conditional decision probability versus sibling.
    """
    enc = getattr(model, "encoder", None)
    if enc is None:
        raise ValueError("model has no .encoder attribute")

    model.eval()
    images = images.to(device)
    batch_size = images.shape[0]

    stage_data = []
    with torch.no_grad():
        x = enc.stem(images)
        for stage_idx, stage in enumerate(enc.taxon_stages):
            out, logp_all, _ = stage(x)
            depths = []
            for depth_idx, (n_ch, logp_chunk) in enumerate(
                zip(stage.layer_channels, torch.split(logp_all, stage.layer_channels, dim=1))
            ):
                prob = logp_chunk.exp()
                prob_pooled = prob.mean(dim=(2, 3)).cpu().numpy()
                depths.append((depth_idx + 1, n_ch, prob_pooled))
            stage_data.append((
                stage_idx + 1,
                stage.n_taxonomy_layers,
                stage.total_out_channels,
                depths,
            ))
            x = out

    per_sample: List[list] = []
    for sample_idx in range(batch_size):
        sample_stages = []
        for stage_num, n_depths, total_ch, depths in stage_data:
            depths_out = []
            for depth_num, n_nodes, prob_pooled in depths:
                probs = prob_pooled[sample_idx]
                selected = int(probs.argmax())
                bits = [
                    (selected >> (depth_num - 1 - b)) & 1
                    for b in range(depth_num)
                ]
                path_str = "-".join("R" if b else "L" for b in bits)
                sibling = selected ^ 1
                pair_total = float(probs[selected]) + float(probs[sibling])
                cond_prob = float(probs[selected]) / max(pair_total, 1e-9)
                depths_out.append({
                    "depth": depth_num,
                    "n_nodes": n_nodes,
                    "probs": probs,
                    "selected": selected,
                    "path_bits": bits,
                    "path_str": path_str,
                    "cond_prob": cond_prob,
                })

            sample_stages.append({
                "stage": stage_num,
                "n_depths": n_depths,
                "total_channels": total_ch,
                "depths": depths_out,
            })
        per_sample.append(sample_stages)

    return per_sample


def print_parse_tree(per_sample: List[list], max_samples: int = 4) -> None:
    """Print a concise parse-tree summary for up to ``max_samples`` samples."""
    for sample_idx, stages in enumerate(per_sample[:max_samples]):
        print(f"\n{'='*60}")
        print(f"  Parse tree — Sample {sample_idx}")
        print(f"{'='*60}")
        for stage_info in stages:
            print(
                f"\n  Stage {stage_info['stage']} "
                f"({stage_info['n_depths']} depths, {stage_info['total_channels']} latent channels)"
            )
            for d_info in stage_info["depths"]:
                depth = d_info["depth"]
                bits = d_info["path_bits"]
                direction = "R" if bits[-1] else "L"
                direction_opp = "L" if bits[-1] else "R"
                cond = d_info["cond_prob"]
                leaf_p = float(d_info["probs"][d_info["selected"]])
                indent = "  " * (depth + 1)
                print(
                    f"  {indent}depth {depth:2d}: "
                    f"[{direction} p={cond:.3f} | {direction_opp} p={1-cond:.3f}] "
                    f"-> path={d_info['path_str']} joint_p={leaf_p:.5f}"
                )

            leaf = stage_info["depths"][-1]
            leaf_node = leaf["selected"]
            print(
                f"\n  Stage {stage_info['stage']} leaf: node {leaf_node}/{leaf['n_nodes'] - 1} "
                f"({leaf['path_str']}) joint_p={float(leaf['probs'][leaf_node]):.5f}"
            )


def save_parse_tree_json(per_sample: List[list], path: str) -> None:
    """Save parse-tree summary to JSON without raw probability arrays."""
    serialisable = []
    for stages in per_sample:
        stage_list = []
        for stage_info in stages:
            depth_list = []
            for depth in stage_info["depths"]:
                depth_list.append({
                    "depth": depth["depth"],
                    "n_nodes": depth["n_nodes"],
                    "selected": depth["selected"],
                    "path_bits": depth["path_bits"],
                    "path_str": depth["path_str"],
                    "cond_prob": round(float(depth["cond_prob"]), 6),
                    "joint_prob": round(float(depth["probs"][depth["selected"]]), 6),
                })
            stage_list.append({
                "stage": stage_info["stage"],
                "n_depths": stage_info["n_depths"],
                "total_channels": stage_info["total_channels"],
                "depths": depth_list,
            })
        serialisable.append(stage_list)

    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Parse tree JSON saved to {path}")


def visualize_parse_tree(
    per_sample: List[list],
    images: torch.Tensor,
    save_dir: str,
    max_samples: int = 4,
) -> None:
    """Render one parse-tree PNG per sample with one panel per stage."""
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches

    os.makedirs(save_dir, exist_ok=True)

    for sample_idx, stages in enumerate(per_sample[:max_samples]):
        n_stages = len(stages)
        fig_w = 3.5 + 4.5 * n_stages
        fig_h = 6
        fig, axes = plt.subplots(1, n_stages + 1, figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#1a1a2e")
        for ax in axes:
            ax.set_facecolor("#1a1a2e")

        ax_img = axes[0]
        img_np = images[sample_idx].cpu().float().numpy()
        img_np = np.clip(img_np * 0.5 + 0.5, 0, 1)
        if img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)
        elif img_np.shape[0] == 1:
            img_np = img_np[0]
        ax_img.imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
        ax_img.set_title(f"Sample {sample_idx}", color="white", fontsize=10, pad=6)
        ax_img.axis("off")

        cmap = cm.get_cmap("YlOrRd")
        for col, stage_info in enumerate(stages):
            ax = axes[col + 1]
            ax.set_xlim(-1, 1)
            depths = stage_info["depths"]
            n_depths = len(depths)
            ax.set_ylim(-0.5, n_depths + 0.5)
            ax.set_title(
                f"Stage {stage_info['stage']}\n({n_depths} depths)",
                color="white",
                fontsize=9,
                pad=6,
            )
            ax.axis("off")

            selected_per_depth = [d["selected"] for d in depths]

            for d_idx, d_info in enumerate(depths):
                y = n_depths - d_idx
                n_nodes = d_info["n_nodes"]
                probs = d_info["probs"]
                selected = d_info["selected"]
                xs = np.linspace(-0.9, 0.9, n_nodes)

                if d_idx + 1 < n_depths:
                    y_child = n_depths - (d_idx + 1)
                    n_children = n_nodes * 2
                    xs_child = np.linspace(-0.9, 0.9, n_children)
                    for node_i, x_parent in enumerate(xs):
                        lc = node_i * 2
                        rc = node_i * 2 + 1
                        for child_i in (lc, rc):
                            on_path = (
                                selected_per_depth[d_idx] == node_i
                                and selected_per_depth[d_idx + 1] == child_i
                            )
                            ax.plot(
                                [x_parent, xs_child[child_i]],
                                [y, y_child],
                                color="#e63946" if on_path else "#444466",
                                lw=2.5 if on_path else 0.7,
                                zorder=1,
                            )

                for node_i, (x_node, prob_val) in enumerate(zip(xs, probs)):
                    colour = cmap(float(prob_val) / max(probs.max(), 1e-9))
                    is_selected = node_i == selected
                    circle = plt.Circle(
                        (x_node, y),
                        radius=0.07 if n_nodes <= 32 else 0.04,
                        color=colour,
                        ec="#e63946" if is_selected else "#888899",
                        lw=2.0 if is_selected else 0.5,
                        zorder=2,
                    )
                    ax.add_patch(circle)

            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=0, vmax=float(depths[-1]["probs"].max())),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, orientation="vertical")
            cbar.ax.yaxis.set_tick_params(color="white")
            cbar.outline.set_edgecolor("white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white", fontsize=6)
            cbar.set_label("joint prob", color="white", fontsize=7)

        sel_patch = mpatches.Patch(color="#e63946", label="selected path")
        fig.legend(
            handles=[sel_patch],
            loc="lower center",
            ncol=1,
            facecolor="#1a1a2e",
            edgecolor="white",
            labelcolor="white",
            fontsize=8,
        )

        plt.suptitle(
            f"Taxonomy Parse Tree - Sample {sample_idx}",
            color="white",
            fontsize=12,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"parse_tree_sample_{sample_idx:03d}.png")
        plt.savefig(
            out_path,
            dpi=150,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        plt.close()
        print(f"  parse tree figure saved: {out_path}")


def analyze_parse_tree(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 4,
) -> None:
    """Run parse-tree analysis and save outputs to ``save_dir/parse_tree_viz``."""
    parse_dir = os.path.join(save_dir, "parse_tree_viz")
    os.makedirs(parse_dir, exist_ok=True)

    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images]

    trees = extract_parse_tree(model, images, device)
    print_parse_tree(trees, max_samples=num_images)
    save_parse_tree_json(trees, os.path.join(parse_dir, "parse_tree_summary.json"))
    visualize_parse_tree(trees, images, save_dir=parse_dir, max_samples=num_images)
    print(f"Parse tree analysis saved to {parse_dir}")


# ── 2e. Hierarchical activation maps ─────────────────────────────────────────

def extract_hierarchical_activations(
    model: TaxonAutoencoder,
    images: torch.Tensor,
    device: torch.device,
) -> List[dict]:
    """Run the encoder and return per-depth gated activations for every stage.

    For each encoder stage and each taxonomy depth we capture::

        gated_acts  (B, 2^d, H, W)  – logits * joint path probability
        raw_logits  (B, 2^d, H, W)  – pre-gating logits from the residual blocks
        probs       (B, 2^d, H, W)  – joint path probability (root-to-node)

    Returns a list of dicts, one per stage::

        {"stage": int, "n_depths": int, "depths": [{"depth", "n_nodes",
          "gated_acts", "raw_logits", "probs"}, ...]}
    """
    enc = getattr(model, "encoder", None)
    if enc is None:
        raise ValueError("model has no .encoder attribute")

    model.eval()
    images = images.to(device)
    all_stage_data = []

    with torch.no_grad():
        x = enc.stem(images)
        for stage_idx, stage in enumerate(enc.taxon_stages):
            logits_per_depth = stage._taxon_logits_per_depth(x)
            depths = []
            prev_logp = None
            stage_outputs = []

            for d_idx, logits in enumerate(logits_per_depth):
                log_cond = stage._pairwise_log_softmax(logits)
                logp = (
                    log_cond
                    if prev_logp is None
                    else log_cond + prev_logp.repeat_interleave(2, dim=1)
                )
                prob = logp.exp()
                out = logits * prob
                stage_outputs.append(out)

                depths.append({
                    "depth": d_idx + 1,
                    "n_nodes": logits.shape[1],
                    "gated_acts": out.cpu().float(),
                    "raw_logits": logits.cpu().float(),
                    "probs": prob.cpu().float(),
                })
                prev_logp = logp

            x = torch.cat(stage_outputs, dim=1)
            all_stage_data.append({
                "stage": stage_idx + 1,
                "n_depths": stage.n_taxonomy_layers,
                "depths": depths,
            })

    return all_stage_data


def _render_hierarchical_activations(
    s_idx: int,
    stage_info: dict,
    max_depth: int,
    max_cols: int,
    use_nms: bool,
    nms_threshold: float,
    save_dir: str,
) -> None:
    """Render one (sample, stage) PNG for gated hierarchical activations."""
    import matplotlib
    matplotlib.use("Agg")

    stage_num = stage_info["stage"]
    depths = stage_info["depths"][:max_depth]
    n_rows = len(depths)
    max_cols_here = min(depths[-1]["n_nodes"], max_cols)

    cell = 1.6
    fig_w = max(6.0, cell * max_cols_here)
    fig_h = cell * n_rows + 0.6

    fig, axes = plt.subplots(n_rows, max_cols_here, figsize=(fig_w, fig_h),
                             squeeze=False, facecolor="#0f0f1c")
    fig.subplots_adjust(hspace=0.55, wspace=0.12,
                        left=0.06, right=0.98, top=0.90, bottom=0.04)

    for ax in axes.flat:
        ax.set_visible(False)

    if use_nms:
        _winner_maps, _winner_probs, _conf_masks = [], [], []
        for d_info in depths:
            pm = d_info["probs"][s_idx]             # (n_nodes, H, W)
            w = pm.argmax(dim=0).numpy()
            wp = pm.max(dim=0).values.numpy()
            cm_ = wp > nms_threshold
            _winner_maps.append(w)
            _winner_probs.append(wp)
            _conf_masks.append(cm_)

    for row, d_info in enumerate(depths):
        d = d_info["depth"]
        n_nodes = d_info["n_nodes"]
        gated = d_info["gated_acts"][s_idx]         # (n_nodes, H, W)
        prob_maps = d_info["probs"][s_idx]           # (n_nodes, H, W)
        prob_vals = prob_maps.mean(dim=(-1, -2)).numpy()
        selected = int(prob_vals.argmax())

        if use_nms:
            _wmap = _winner_maps[row]
            _cmask = _conf_masks[row]

        if n_nodes <= max_cols_here:
            show_nodes = list(range(n_nodes))
            col_offset = (max_cols_here - n_nodes) // 2
        else:
            half = max_cols_here // 2
            start = max(0, min(selected - half, n_nodes - max_cols_here))
            show_nodes = list(range(start, start + max_cols_here))
            col_offset = 0

        for col_idx, node_i in enumerate(show_nodes):
            col = col_offset + col_idx
            if col >= max_cols_here:
                break
            ax = axes[row, col]
            ax.set_visible(True)
            ax.set_facecolor("#0f0f1c")

            act = gated[node_i].numpy()
            if use_nms:
                nms_mask = (_wmap == node_i) & _cmask
                act = act * nms_mask.astype(act.dtype)

            vmax = float(abs(act).max()) or 1e-6
            ax.imshow(act, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      aspect="auto", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            is_sel = node_i == selected
            ec = "#e63946" if is_sel else "#2a2a4a"
            lw = 2.2 if is_sel else 0.5
            for spine in ax.spines.values():
                spine.set_edgecolor(ec)
                spine.set_linewidth(lw)

            p = float(prob_vals[node_i])
            label = (f"*{node_i}" if is_sel else f"{node_i}") + f"\n{p:.3f}"
            ax.set_title(label, fontsize=5.5,
                         color="#e63946" if is_sel else "#7070a0",
                         pad=1.5)

        left_col = col_offset if n_nodes <= max_cols_here else 0
        axes[row, left_col].set_ylabel(
            f"depth {d}", color="#aaaacc", fontsize=6.5, rotation=90, labelpad=3,
        )

    nms_tag = " [NMS]" if use_nms else ""
    fig.suptitle(
        f"Hierarchical Activations{nms_tag}  *  Sample {s_idx}  *  Stage {stage_num}",
        color="white", fontsize=10, fontweight="bold",
    )

    suffix = "_nms" if use_nms else ""
    out_path = os.path.join(save_dir, f"hier_acts_s{s_idx:03d}_stage{stage_num}{suffix}.png")
    plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved {out_path}")

    # winner-map overview (NMS only)
    if use_nms:
        import matplotlib.cm as _cm
        from matplotlib.colors import BoundaryNorm, ListedColormap

        wfig, waxes = plt.subplots(
            1, len(depths),
            figsize=(3.5 * len(depths), 3.5),
            facecolor="#0f0f1c",
            squeeze=False,
        )
        wfig.subplots_adjust(wspace=0.15, left=0.04, right=0.96, top=0.82, bottom=0.06)

        for col, (d_info, wmap, wprob, cmask) in enumerate(
            zip(depths, _winner_maps, _winner_probs, _conf_masks)
        ):
            ax = waxes[0, col]
            ax.set_facecolor("#0f0f1c")
            n_nodes = d_info["n_nodes"]

            disp = np.where(cmask, wmap, -1).astype(float)
            cmap_nodes = _cm.get_cmap("tab20", n_nodes)
            colors = ["#0f0f1c"] + [cmap_nodes(i / max(n_nodes - 1, 1)) for i in range(n_nodes)]
            lmap = ListedColormap(colors)
            lnorm = BoundaryNorm([-1.5] + [i - 0.5 for i in range(n_nodes + 1)], len(colors))

            ax.imshow(disp, cmap=lmap, norm=lnorm, aspect="auto", interpolation="nearest")
            alpha_map = np.clip(wprob * cmask.astype(float), 0, 1)
            ax.imshow(alpha_map, cmap="gray", alpha=0.25, aspect="auto", interpolation="nearest")

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"depth {d_info['depth']}  ({n_nodes} nodes)",
                         color="#aaaacc", fontsize=8, pad=3)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

        thresh_str = f"  (thr={nms_threshold:.2f})" if nms_threshold > 0 else ""
        wfig.suptitle(
            f"Winner Map{thresh_str}  *  Sample {s_idx}  *  Stage {stage_num}",
            color="white", fontsize=10, fontweight="bold",
        )
        wout = os.path.join(save_dir, f"winner_map_s{s_idx:03d}_stage{stage_num}.png")
        wfig.savefig(wout, dpi=140, facecolor=wfig.get_facecolor())
        plt.close(wfig)
        print(f"  saved {wout}")


def analyze_hierarchical_activations(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 4,
    max_depth: int = 4,
    max_cols: int = 16,
    use_nms: bool = False,
    nms_threshold: float = 0.0,
) -> None:
    """Extract and visualise hierarchical gated activations, saving to
    ``save_dir/hierarchical_activations``."""
    hier_dir = os.path.join(save_dir, "hierarchical_activations")
    os.makedirs(hier_dir, exist_ok=True)

    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images]

    print("  extracting hierarchical activations...")
    stage_data = extract_hierarchical_activations(model, images, device)

    for s_idx in range(len(images)):
        for stage_info in stage_data:
            _render_hierarchical_activations(
                s_idx, stage_info,
                max_depth=max_depth,
                max_cols=max_cols,
                use_nms=use_nms,
                nms_threshold=nms_threshold,
                save_dir=hier_dir,
            )

    print(f"Hierarchical activation maps saved to {hier_dir}")


# ── 2f. Binary split maps ─────────────────────────────────────────────────────

def analyze_binary_split_maps(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 4,
    max_depth: int = 4,
    max_pairs: int = 8,
) -> None:
    """Visualise per–sibling-pair conditional split maps, saving to
    ``save_dir/binary_split_maps``.

    At each depth and for each sibling pair ``(2k, 2k+1)`` a spatial map is
    computed::

        cond_left(h, w) = P_joint(2k, h, w) / (P_joint(2k, h, w) + P_joint(2k+1, h, w))

    Colour (RdBu_r): blue = right wins, red = left wins, white = uncertain.
    Alpha is proportional to the parent joint probability so only unambiguous
    routing regions are opaque.
    """
    import matplotlib.cm as cm

    split_dir = os.path.join(save_dir, "binary_split_maps")
    os.makedirs(split_dir, exist_ok=True)

    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images]

    print("  extracting activations for binary split maps...")
    stage_data = extract_hierarchical_activations(model, images, device)

    rdbu = cm.get_cmap("RdBu_r")
    bg_colour = np.array([0.06, 0.06, 0.11, 1.0])

    for s_idx in range(len(images)):
        for stage_info in stage_data:
            stage_num = stage_info["stage"]
            depths = stage_info["depths"][:max_depth]
            n_rows = len(depths)

            max_pairs_here = min(depths[-1]["n_nodes"] // 2, max_pairs)
            max_pairs_here = max(max_pairs_here, 1)

            cell_w, cell_h = 2.0, 2.0
            fig_w = max(5.0, cell_w * max_pairs_here + 1.0)
            fig_h = cell_h * n_rows + 0.9

            fig, axes = plt.subplots(n_rows, max_pairs_here, figsize=(fig_w, fig_h),
                                     squeeze=False, facecolor="#0f0f1c")
            fig.subplots_adjust(hspace=0.65, wspace=0.08,
                                left=0.07, right=0.97, top=0.88, bottom=0.05)

            for ax in axes.flat:
                ax.set_visible(False)

            for row, d_info in enumerate(depths):
                d = d_info["depth"]
                n_nodes = d_info["n_nodes"]
                n_pairs = n_nodes // 2

                probs = d_info["probs"][s_idx]           # (n_nodes, H, W) tensor
                prob_vals = probs.mean(dim=(-1, -2)).numpy()
                selected_node = int(prob_vals.argmax())
                selected_pair = selected_node // 2

                if n_pairs <= max_pairs_here:
                    show_pairs = list(range(n_pairs))
                    col_offset = (max_pairs_here - n_pairs) // 2
                else:
                    half = max_pairs_here // 2
                    start = max(0, min(selected_pair - half,
                                      n_pairs - max_pairs_here))
                    show_pairs = list(range(start, start + max_pairs_here))
                    col_offset = 0

                p_parent_all = float((probs[0::2] + probs[1::2]).max())
                p_parent_all = max(p_parent_all, 1e-9)

                for col_idx, pair_k in enumerate(show_pairs):
                    col = col_offset + col_idx
                    if col >= max_pairs_here:
                        break
                    ax = axes[row, col]
                    ax.set_visible(True)
                    ax.set_facecolor("#0f0f1c")

                    p_l = probs[2 * pair_k].numpy()
                    p_r = probs[2 * pair_k + 1].numpy()
                    p_parent = p_l + p_r
                    cond_left = p_l / np.maximum(p_parent, 1e-9)
                    alpha = np.clip(p_parent / p_parent_all * 4.0, 0.0, 1.0)

                    rgba = rdbu(cond_left).astype(np.float64)
                    rgba[..., 3] = alpha
                    H_img, W_img = cond_left.shape
                    bg = np.ones((H_img, W_img, 4)) * bg_colour
                    ax.imshow(bg, aspect="auto", interpolation="nearest")
                    ax.imshow(rgba, aspect="auto", interpolation="nearest")

                    ax.set_xticks([])
                    ax.set_yticks([])

                    is_sel = pair_k == selected_pair
                    ec = "#e63946" if is_sel else "#1e1e3a"
                    lw = 2.5 if is_sel else 0.6
                    for spine in ax.spines.values():
                        spine.set_edgecolor(ec)
                        spine.set_linewidth(lw)

                    mean_cond = float(cond_left.mean())
                    arrow = "<- L" if mean_cond > 0.5 else "R ->"
                    conf = abs(mean_cond - 0.5) * 2.0
                    title_col = "#e63946" if is_sel else "#8888aa"
                    ax.set_title(
                        f"{'* ' if is_sel else ''}L{2*pair_k}|R{2*pair_k+1}\n"
                        f"{arrow}  {conf:.2f}",
                        fontsize=5.5, color=title_col, pad=1.5,
                    )

                left_col = col_offset if n_pairs <= max_pairs_here else 0
                axes[row, left_col].set_ylabel(
                    f"depth {d}", color="#aaaacc", fontsize=7, rotation=90, labelpad=3,
                )

            sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[:, -1], fraction=0.05, pad=0.02,
                                orientation="vertical")
            cbar.set_ticks([0.0, 0.5, 1.0])
            cbar.set_ticklabels(["Right\n(P=1)", "Uncertain", "Left\n(P=1)"], fontsize=5.5)
            cbar.ax.yaxis.set_tick_params(color="white")
            cbar.outline.set_edgecolor("white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
            cbar.set_label("P(left | parent)", color="white", fontsize=7)

            fig.suptitle(
                f"Binary Split Maps  *  Sample {s_idx}  *  Stage {stage_num}\n"
                f"(alpha proportional to parent probability)",
                color="white", fontsize=9, fontweight="bold",
            )
            out_path = os.path.join(split_dir,
                                    f"split_map_s{s_idx:03d}_stage{stage_num}.png")
            plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
            plt.close()
            print(f"  saved {out_path}")

    print(f"Binary split maps saved to {split_dir}")


# ── 3. Latent space sparsity ──────────────────────────────────────────────────

def analyze_latent_sparsity(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_batches: int = 50,
    sparsity_threshold: float = 0.1,
) -> None:
    """Sparsity and norm statistics of the latent space."""
    model.eval()
    all_latents: List[np.ndarray] = []

    print(f"Encoding {num_batches} batches for latent space analysis...")
    with torch.no_grad():
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

    mean_act = np.abs(Z).mean(axis=0)
    std_act  = Z.std(axis=0)
    sparsity = np.mean(np.abs(Z) < sparsity_threshold, axis=1)
    l0 = np.sum(np.abs(Z) > sparsity_threshold, axis=1)
    l1 = np.abs(Z).sum(axis=1)
    l2 = np.sqrt((Z ** 2).sum(axis=1))

    print(
        f"  mean_sparsity={sparsity.mean():.4f}  mean_l0={l0.mean():.1f}  "
        f"mean_l1={l1.mean():.4f}  mean_l2={l2.mean():.4f}"
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].bar(range(min(D, 256)), mean_act[:256])
    axes[0, 0].set_title("Mean |activation| per dim (first 256)")

    axes[0, 1].bar(range(min(D, 256)), std_act[:256], color="darkorange")
    axes[0, 1].set_title("Std activation per dim (first 256)")

    axes[0, 2].hist(sparsity, bins=50, edgecolor="black")
    axes[0, 2].axvline(sparsity.mean(), color="red", linestyle="--",
                       label=f"Mean={sparsity.mean():.3f}")
    axes[0, 2].set_title("Sparsity per sample")
    axes[0, 2].legend()

    axes[1, 0].hist(l0, bins=50, edgecolor="black")
    axes[1, 0].axvline(l0.mean(), color="red", linestyle="--",
                       label=f"Mean={l0.mean():.1f}")
    axes[1, 0].set_title("L0 norm per sample")
    axes[1, 0].legend()

    axes[1, 1].hist(l1, bins=50, edgecolor="black")
    axes[1, 1].axvline(l1.mean(), color="red", linestyle="--",
                       label=f"Mean={l1.mean():.2f}")
    axes[1, 1].set_title("L1 norm per sample")
    axes[1, 1].legend()

    axes[1, 2].hist(l2, bins=50, edgecolor="black")
    axes[1, 2].axvline(l2.mean(), color="red", linestyle="--",
                       label=f"Mean={l2.mean():.2f}")
    axes[1, 2].set_title("L2 norm per sample")
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "latent_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    np.savez(
        os.path.join(save_dir, "latent_statistics.npz"),
        mean_activation=mean_act, std_activation=std_act,
        sparsity_per_sample=sparsity, l0_norm=l0, l1_norm=l1, l2_norm=l2,
        mean_sparsity=float(sparsity.mean()), mean_l0=float(l0.mean()),
        mean_l1=float(l1.mean()), mean_l2=float(l2.mean()),
        latent_dim=D, num_samples=N,
    )
    print(f"Latent space analysis saved to {save_dir}")


# ── 4. Reconstruction quality ─────────────────────────────────────────────────

def analyze_reconstruction_quality(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_batches: int = 20,
) -> None:
    """Per-sample MSE and MAE distributions."""
    model.eval()
    mse_list: List[float] = []
    mae_list: List[float] = []

    print(f"Computing reconstruction metrics on {num_batches} batches...")
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(data_loader, desc="Recon quality")):
            if i >= num_batches:
                break
            images = images.to(device)
            recon, *_ = model(images)
            mse_list.extend(((images - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
            mae_list.extend(torch.abs(images - recon).mean(dim=(1, 2, 3)).cpu().numpy())

    mse = np.array(mse_list)
    mae = np.array(mae_list)
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
    print(f"Reconstruction metrics saved to {save_dir}")


# ── 5. Multiple reconstructions ──────────────────────────────────────────────

def visualize_multiple_reconstructions(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 8,
    num_sets: int = 8,
    normalized: bool = True,
) -> None:
    """Save side-by-side original/reconstruction grids."""
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
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2.5, 5))

        with torch.no_grad():
            recon, *_ = model(images)

        for i in range(num_images):
            axes[0, i].imshow(_to_display(images[i:i+1], normalized=normalized))
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Original", fontweight="bold")
            axes[1, i].imshow(_to_display(recon[i:i+1], normalized=normalized))
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_title("Reconstruction", fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(recon_dir, f"set_{set_idx+1:02d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Set {set_idx+1}/{num_sets} saved.")

    print(f"Reconstructions saved to {recon_dir}")


# ── 6. Partonomy sparsity suite ───────────────────────────────────────────────

def analyze_partonomy_sparsity(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_batches: int = 30,
    sparsity_threshold: float = 0.1,
    ablation_images: int = 16,
    n_clusters: int = 8,
) -> None:
    """Five-part partonomy sparsity suite."""
    prt_dir = os.path.join(save_dir, "partonomy_sparsity")
    os.makedirs(prt_dir, exist_ok=True)

    model.eval()
    all_latents: List[np.ndarray] = []
    hook_data: Dict[int, List[torch.Tensor]] = {}
    ablation_imgs: Optional[torch.Tensor] = None

    hooks = []
    for i, stage in enumerate(model.encoder.taxon_stages):
        def _make_hook(idx):
            def _hook(m, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                hook_data.setdefault(idx, []).append(t.detach().cpu())
            return _hook
        hooks.append(stage.register_forward_hook(_make_hook(i)))

    print(f"Collecting activations over {num_batches} batches...")
    with torch.no_grad():
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
    print("  [1/5] Jaccard activation overlap...")
    rng = np.random.default_rng(0)
    n_pairs = min(3000, N * (N - 1) // 2)
    ia = rng.integers(0, N, n_pairs)
    ib = rng.integers(0, N, n_pairs)
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
    axes[0].axvline(jaccard.mean(), color="red", linestyle="--",
                    label=f"Mean={jaccard.mean():.3f}")
    axes[0].set_title("Pairwise Activation Overlap"); axes[0].legend()
    axes[1].bar(range(min(D, 200)), sorted(lifetime_sp, reverse=True)[:200], color="darkorange")
    axes[1].set_title("Lifetime Sparsity per Dim (sorted)")
    axes[2].bar(["Dead\n(<1%)", "Selective\n(1-20%)", "Moderate\n(20-50%)", "Dense\n(>50%)"],
                [v * 100 for v in [dead, selective, moderate, dense]],
                color=["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"])
    axes[2].set_ylabel("% Features"); axes[2].set_title("Feature Activity Categories")
    plt.suptitle("1. Activation Overlap & Specialisation", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(prt_dir, "01_jaccard_overlap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    np.savez(os.path.join(prt_dir, "01_jaccard_stats.npz"), jaccard=jaccard,
             lifetime_sparsity=lifetime_sp, dead_frac=dead, selective_frac=selective,
             dense_frac=dense)
    print(f"    Jaccard mean={jaccard.mean():.4f}  dead={dead*100:.1f}%  "
          f"selective={selective*100:.1f}%  dense={dense*100:.1f}%")

    # 2. Selectivity
    print("  [2/5] Feature selectivity and entropy...")
    p = lifetime_sp.clip(1e-6, 1 - 1e-6)
    per_dim_ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    try:
        from scipy.stats import kurtosis as _sp_kurt
        per_dim_kurt = np.array([_sp_kurt(np.abs(Z[:, d]), fisher=True) for d in range(D)])
    except Exception:
        per_dim_kurt = np.zeros(D)
    per_dim_max  = np.abs(Z).max(axis=0)
    per_dim_mean = np.abs(Z).mean(axis=0)
    sel_idx = (per_dim_max - per_dim_mean) / (per_dim_max + per_dim_mean + 1e-8)
    poly_frac = float((np.abs(p - 0.5) < 0.15).mean())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(per_dim_ent, bins=50, edgecolor="black", color="purple")
    axes[0, 0].axvline(per_dim_ent.mean(), color="red", linestyle="--",
                       label=f"Mean={per_dim_ent.mean():.3f} bits")
    axes[0, 0].set_title("Per-Dim Activation Entropy"); axes[0, 0].legend()
    axes[0, 1].hist(per_dim_kurt.clip(-10, 50), bins=50, edgecolor="black", color="teal")
    axes[0, 1].axvline(per_dim_kurt.mean(), color="red", linestyle="--",
                       label=f"Mean={per_dim_kurt.mean():.2f}")
    axes[0, 1].set_title("Per-Dim Kurtosis"); axes[0, 1].legend()
    axes[1, 0].hist(sel_idx, bins=50, edgecolor="black", color="darkgreen")
    axes[1, 0].axvline(sel_idx.mean(), color="red", linestyle="--",
                       label=f"Mean={sel_idx.mean():.3f}")
    axes[1, 0].set_title("Feature Selectivity Index"); axes[1, 0].legend()
    axes[1, 1].scatter(lifetime_sp, per_dim_ent, alpha=0.3, s=8, c="navy")
    axes[1, 1].set_title(f"Sparsity vs Entropy  (poly~{poly_frac*100:.1f}%)")
    plt.suptitle("2. Feature Selectivity & Entropy", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(prt_dir, "02_feature_selectivity.png"), dpi=150, bbox_inches="tight")
    plt.close()
    np.savez(os.path.join(prt_dir, "02_selectivity_stats.npz"),
             per_dim_entropy=per_dim_ent, per_dim_kurtosis=per_dim_kurt,
             selectivity_idx=sel_idx, polysemantic_frac=poly_frac)

    # 3. Ablations
    print("  [3/5] Causal ablation analysis...")
    if ablation_imgs is not None:
        ablation_imgs = ablation_imgs.to(device)
        with torch.no_grad():
            z_orig, _ = model.encode(ablation_imgs)
            spatial = (z_orig.ndim == 4)
            n_units = z_orig.shape[1]
            base_recon, *_ = model(ablation_imgs)
            if base_recon.shape[2:] != ablation_imgs.shape[2:]:
                base_recon = F.interpolate(base_recon, size=ablation_imgs.shape[2:],
                                           mode="bilinear", align_corners=False)
            base_mse = ((ablation_imgs - base_recon) ** 2).mean(dim=(1, 2, 3))

            if spatial:
                unit_active = (z_orig.abs().mean(dim=(2, 3)) > sparsity_threshold).cpu().numpy()
            else:
                unit_active = (z_orig.abs() > sparsity_threshold).cpu().numpy()
            candidates = np.where(unit_active.any(axis=0))[0]
            if len(candidates) > 256:
                ma = (z_orig.abs().mean(dim=(2, 3)) if spatial else z_orig.abs()).mean(0).cpu().numpy()
                candidates = candidates[np.argsort(-ma[candidates])[:256]]

            print(f"    Ablating {len(candidates)}/{n_units} units...")
            importance = np.zeros(n_units)
            for d in candidates:
                z_abl = z_orig.clone()
                z_abl[:, d] = 0.0
                recon_abl, _ = model.decode(z_abl)
                if recon_abl.shape[2:] != ablation_imgs.shape[2:]:
                    recon_abl = F.interpolate(recon_abl, size=ablation_imgs.shape[2:],
                                              mode="bilinear", align_corners=False)
                delta = ((ablation_imgs - recon_abl) ** 2).mean(dim=(1, 2, 3))
                importance[d] = (delta - base_mse).clamp(min=0).mean().item()

        sorted_imp = np.sort(importance[importance > 0])[::-1]
        total_imp = sorted_imp.sum()
        cumulative = np.cumsum(sorted_imp) / (total_imp + 1e-12)
        p50 = int(np.searchsorted(cumulative, 0.5)) + 1 if total_imp > 0 else 0
        p90 = int(np.searchsorted(cumulative, 0.9)) + 1 if total_imp > 0 else 0

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        show_n = min(50, len(sorted_imp))
        axes[0].bar(range(show_n), sorted_imp[:show_n], color="firebrick")
        axes[0].set_title(f"Top-{show_n} Units by Ablation Importance")
        axes[1].hist(sorted_imp, bins=40, edgecolor="black", color="salmon")
        if len(sorted_imp):
            axes[1].axvline(sorted_imp.mean(), color="blue", linestyle="--",
                            label=f"Mean={sorted_imp.mean():.5f}")
        axes[1].set_title("Ablation Importance Distribution"); axes[1].legend()
        if len(sorted_imp):
            axes[2].plot(range(1, len(sorted_imp) + 1), cumulative * 100, color="darkblue")
            axes[2].axvline(p50, color="orange", linestyle="--", label=f"50% in top-{p50}")
            axes[2].axvline(p90, color="red",    linestyle="--", label=f"90% in top-{p90}")
        axes[2].set_title("Cumulative Importance"); axes[2].legend()
        plt.suptitle("3. Causal Ablation: Unit Indispensability", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(prt_dir, "03_causal_ablations.png"), dpi=150, bbox_inches="tight")
        plt.close()
        np.savez(os.path.join(prt_dir, "03_ablation_importance.npz"),
                 unit_importance=importance, p50_units=p50, p90_units=p90)
        print(f"    50% importance in top-{p50};  90% in top-{p90}")

    # 4. Cross-stage dependency
    print("  [4/5] Cross-stage sparsity dependency...")
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
            dep_sp = float((M < 0.1).mean())
            dep_sp_list.append(dep_sp)
            ax = axes[0, pi]
            im = ax.imshow(M, aspect="auto", cmap="hot", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel(f"Stage {s2+1} channels")
            ax.set_ylabel(f"Stage {s1+1} channels")
            ax.set_title(f"Stage {s1+1}->{s2+1}  dep={dep_sp*100:.1f}%")
        plt.suptitle("4. Cross-Stage Sparsity Dependency", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(prt_dir, "04_cross_stage_dependency.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        np.savez(os.path.join(prt_dir, "04_dependency_stats.npz"),
                 dep_sparsity_per_pair=np.array(dep_sp_list))
        print(f"    Mean dep. sparsity: {np.mean(dep_sp_list)*100:.1f}%")
    else:
        print("    Skipping (<2 stages captured).")

    # 5. Clustering
    print("  [5/5] Clustering latent dimensions...")
    try:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA as _PCA
        from sklearn.metrics import silhouette_score
        _sklearn_ok = True
    except ImportError:
        _sklearn_ok = False
        print("    sklearn not available; skipping.")

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
        axes[0].set_title(f"Cluster Sizes (Silhouette={sil:.3f})")
        im = axes[1].imshow(cluster_mean[:, top_dims], aspect="auto", cmap="viridis")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].set_title(f"Cluster x Feature (top-{n_show})")
        axes[2].hist(exclusivity, bins=40, edgecolor="black", color="darkorange")
        axes[2].axvline(exclusivity.mean(), color="red", linestyle="--",
                        label=f"Mean={exclusivity.mean():.3f}")
        axes[2].set_title("Per-Dim Cluster Exclusivity"); axes[2].legend()
        plt.suptitle("5. Latent Dimension Clustering", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(prt_dir, "05_dimension_clustering.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        np.savez(os.path.join(prt_dir, "05_cluster_stats.npz"),
                 labels=labels, cluster_sizes=cluster_sizes,
                 silhouette=sil, exclusivity=exclusivity)
        print(f"    Silhouette={sil:.4f}  mean exclusivity={exclusivity.mean():.4f}")

    print(f"Partonomy sparsity analysis saved to {prt_dir}")


# ── 7. Stage activation maps ──────────────────────────────────────────────────

def visualize_stage_activations(
    model: TaxonAutoencoder,
    data_loader,
    device: torch.device,
    save_dir: str,
    num_images: int = 3,
    normalized: bool = True,
) -> None:
    """For a few images, save feature-map grids at each encoder stage."""
    act_dir = os.path.join(save_dir, "stage_activations")
    os.makedirs(act_dir, exist_ok=True)

    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:num_images].to(device)

    with torch.no_grad():
        for img_idx in range(num_images):
            img_dir = os.path.join(act_dir, f"image_{img_idx+1:02d}")
            os.makedirs(img_dir, exist_ok=True)
            plt.imsave(os.path.join(img_dir, "original.png"),
                       _to_display(images[img_idx:img_idx+1], normalized=normalized))

            img_t = images[img_idx:img_idx+1]
            x = model.encoder.stem(img_t)

            for s_idx, stage in enumerate(model.encoder.taxon_stages, start=1):
                x_out, _, _ = stage(x)
                maps = x_out[0].detach().cpu()
                n_show = min(maps.shape[0], 16)
                nrow = 4
                fig, axes = plt.subplots(nrow, nrow, figsize=(8, 8))
                axes = np.array(axes).flatten()
                for k in range(n_show):
                    fm = maps[k].numpy()
                    fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
                    axes[k].imshow(fm, cmap="viridis")
                    axes[k].axis("off")
                for ax in axes[n_show:]:
                    ax.axis("off")
                plt.suptitle(
                    f"Stage {s_idx} activations (ch={maps.shape[0]}, showing {n_show})",
                    fontsize=10,
                )
                plt.tight_layout()
                plt.savefig(os.path.join(img_dir, f"stage{s_idx}.png"),
                            dpi=150, bbox_inches="tight")
                plt.close()
                x = x_out

            print(f"  Image {img_idx+1}: saved to {img_dir}")

    print(f"Stage activations saved to {act_dir}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze ResNet Taxon Autoencoder")
    parser.add_argument("--config", type=str, default="configs/taxon_ae_celeba_hq.json",
                        help="Path to JSON config; must set analysis.checkpoint_path")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path from config")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    data_cfg = config.get("data", {})
    batch_size   = args.batch_size  or data_cfg.get("batch_size", 16)
    num_workers  = args.num_workers or data_cfg.get("num_workers", 4)
    data_root    = args.data_root   or data_cfg["data_root"]
    image_size   = data_cfg.get("image_size", 256)
    val_split    = data_cfg.get("val_split", 0.05)

    analysis_cfg = config.get("analysis", {})
    num_latent_batches  = analysis_cfg.get("num_latent_batches", 50)
    num_recon_batches   = analysis_cfg.get("num_reconstruction_batches", 20)
    num_recon_images    = analysis_cfg.get("num_multiple_recon_images", 8)
    num_recon_sets      = analysis_cfg.get("num_reconstructions_per_image", 8)
    num_act_images      = analysis_cfg.get("num_activation_images", 3)
    num_taxon_batches   = analysis_cfg.get("num_taxonomy_batches", 10)
    num_parse_images    = analysis_cfg.get("num_parse_tree_images", 4)
    num_hier_images     = analysis_cfg.get("num_hier_act_images", 4)
    max_hier_depth      = analysis_cfg.get("max_hier_act_depth", 4)
    num_split_images    = analysis_cfg.get("num_split_map_images", 4)
    max_split_pairs     = analysis_cfg.get("max_split_pairs", 8)

    dkl_weight     = config.get("training", {}).get("dkl_weight", None)
    entropy_weight = config.get("training", {}).get("entropy_weight", 0.0)
    temperature    = config.get("model", {}).get("temperature", None)
    hard           = config.get("model", {}).get("hard", False)
    dkl_suffix     = f"_dkl_{dkl_weight:.0e}" if dkl_weight is not None else ""
    temp_str       = f"{temperature:g}".replace(".", "p") if temperature is not None else ""
    temp_suffix    = f"_temp_{temp_str}" if temp_str else ""
    hard_suffix    = "_hard" if hard else ""
    entropy_suffix = f"_ew_{entropy_weight:.0e}" if entropy_weight else ""
    run_suffix     = dkl_suffix + temp_suffix + hard_suffix + entropy_suffix

    save_dir_base = config.get("output", {}).get("analysis_save_dir", "outputs/analysis")
    out_base = config.get("output", {}).get("output_dir", "")
    # Insert the dkl suffix right after the output_dir prefix so the analysis
    # directory lives inside the same dkl-tagged run folder as the checkpoints.
    if out_base and run_suffix and save_dir_base.startswith(out_base):
        save_dir_prefix = out_base + run_suffix + save_dir_base[len(out_base):]
    else:
        save_dir_prefix = save_dir_base.rstrip("/").rstrip("\\") + run_suffix
    experiment_name = config.get("experiment_name", "")
    save_dir = (save_dir_prefix if experiment_name
                else os.path.join(save_dir_prefix, datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(save_dir, exist_ok=True)

    # If checkpoint_path contains the base output_dir, patch in the dkl suffix too.
    checkpoint_path = args.checkpoint or analysis_cfg.get("checkpoint_path")
    if checkpoint_path and run_suffix and out_base and checkpoint_path.startswith(out_base):
        checkpoint_path = out_base + run_suffix + checkpoint_path[len(out_base):]
    if not checkpoint_path:
        raise ValueError(
            "Provide checkpoint_path via --checkpoint or config['analysis']['checkpoint_path']"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("Taxon ResNet Autoencoder — Analysis")
    print("=" * 80)
    print(f"  device={device}  checkpoint={checkpoint_path}")
    print(f"  data_root={data_root}  image_size={image_size}  batch_size={batch_size}")
    print(f"  save_dir={save_dir}")
    print("=" * 80)

    print("\nLoading dataset...")
    if _is_cifar(config):
        loader = CIFAR10Loader(
            batch_size=batch_size,
            root=data_root,
        )
        _, eval_loader = loader.get_loaders()
    else:
        normalize = data_cfg.get("normalize", True)   # default: same normalization as training
        if normalize:
            tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            tf = None  # CelebAHQLoader default (ToTensor only)
        loader = CelebAHQLoader(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            val_split=val_split,
            transform=tf,
        )
        _, val_loader = loader.get_loaders()
        eval_loader = val_loader if val_loader is not None else loader.train_loader

    print("\nLoading model...")
    model, _ = load_model(checkpoint_path, device, config)

    print("\n" + "=" * 80 + "\n1. Stage Filter Visualization\n" + "=" * 80)
    visualize_stage_filters(model, save_dir)

    print("\n" + "=" * 80 + "\n1b. Filter Similarity Analysis\n" + "=" * 80)
    analyze_filter_similarity(model, save_dir)

    vanilla = _has_vanilla_taxonomy(model)

    if vanilla:
        print("\n" + "=" * 80 + "\n2. Taxonomy Probability Distributions\n" + "=" * 80)
        visualize_taxonomy_distributions(model, eval_loader, device, save_dir,
                                         num_batches=num_taxon_batches)

        print("\n" + "=" * 80 + "\n2b. Taxonomy Path Probability Plots\n" + "=" * 80)
        visualize_taxonomy_path_probs(model, eval_loader, device, save_dir)

        print("\n" + "=" * 80 + "\n2c. Taxonomy Tree Activations\n" + "=" * 80)
        visualize_taxonomy_tree(model, eval_loader, device, save_dir,
                                num_images=num_act_images)

        print("\n" + "=" * 80 + "\n2d. Parse Tree Analysis\n" + "=" * 80)
        analyze_parse_tree(model, eval_loader, device, save_dir,
                           num_images=num_parse_images)

        print("\n" + "=" * 80 + "\n2e. Hierarchical Activation Maps\n" + "=" * 80)
        analyze_hierarchical_activations(model, eval_loader, device, save_dir,
                                         num_images=num_hier_images,
                                         max_depth=max_hier_depth)

        print("\n" + "=" * 80 + "\n2f. Binary Split Maps\n" + "=" * 80)
        analyze_binary_split_maps(model, eval_loader, device, save_dir,
                                  num_images=num_split_images,
                                  max_depth=max_hier_depth,
                                  max_pairs=max_split_pairs)
    else:
        print("\n" + "=" * 80)
        print("Skipping taxonomy-specific analyses (2, 2b-2f) — "
              f"model variant ({model.__class__.__name__}) uses non-vanilla routing.")
        print("=" * 80)

    print("\n" + "=" * 80 + "\n3. Latent Space Analysis\n" + "=" * 80)
    analyze_latent_sparsity(model, eval_loader, device, save_dir,
                            num_batches=num_latent_batches)

    print("\n" + "=" * 80 + "\n4. Reconstruction Quality Metrics\n" + "=" * 80)
    analyze_reconstruction_quality(model, eval_loader, device, save_dir,
                                   num_batches=num_recon_batches)

    print("\n" + "=" * 80 + "\n5. Multiple Reconstruction Visualizations\n" + "=" * 80)
    visualize_multiple_reconstructions(model, eval_loader, device, save_dir,
                                       num_images=num_recon_images, num_sets=num_recon_sets,
                                       normalized=normalize)

    print("\n" + "=" * 80 + "\n6. Partonomy Sparsity Suite\n" + "=" * 80)
    analyze_partonomy_sparsity(model, eval_loader, device, save_dir)

    print("\n" + "=" * 80 + "\n7. Encoder Stage Activation Maps\n" + "=" * 80)
    visualize_stage_activations(model, eval_loader, device, save_dir, num_images=num_act_images,
                                normalized=normalize)

    print("\n" + "=" * 80)
    print("Analysis complete! All outputs saved to:", save_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()
