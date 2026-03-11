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
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.taxon_ae import TaxonAutoencoder
from src.utils.dataloader import CelebAHQLoader
from torchvision import transforms


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
    model = TaxonAutoencoder(
        in_channels=mc.get("in_channels", 3),
        resnet_variant=mc.get("resnet_variant", "18"),
        stage_taxonomy_layers=tuple(mc.get("stage_taxonomy_layers", [5, 6, 7, 8])),
        stage_strides=tuple(mc.get("stage_strides", [1, 2, 2, 2])),
        stage_blocks=mc.get("stage_blocks", None),
        temperature=mc.get("temperature", 1.0),
        hard=mc.get("hard", False),
        kernel_size=mc.get("kernel_size", 3),
        use_stem=mc.get("use_stem", True),
        stem_channels=mc.get("stem_channels", 64),
        stem_stride=mc.get("stem_stride", 2),
        use_stem_maxpool=mc.get("use_stem_maxpool", True),
        output_activation=mc.get("output_activation", "none"),
        depth_decay=mc.get("depth_decay", 0.5),
    )

    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")
    if "train_stats" in checkpoint:
        print(f"  train_loss={checkpoint['train_stats'].get('loss', '?'):.6f}")
    if "val_stats" in checkpoint:
        print(f"  val_loss={checkpoint['val_stats'].get('loss', '?'):.6f}")
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
            recon, _, _ = model(images)
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
            recon, _, _ = model(images)

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
            base_recon, _, _ = model(ablation_imgs)
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
    parser.add_argument("--config", type=str, default="configs/celeba_hq.json",
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

    dkl_weight  = config.get("training", {}).get("dkl_weight", None)
    temperature = config.get("model", {}).get("temperature", None)
    hard        = config.get("model", {}).get("hard", False)
    dkl_suffix  = f"_dkl_{dkl_weight:.0e}" if dkl_weight is not None else ""
    temp_str    = f"{temperature:g}".replace(".", "p") if temperature is not None else ""
    temp_suffix = f"_temp_{temp_str}" if temp_str else ""
    hard_suffix = "_hard" if hard else ""
    run_suffix  = dkl_suffix + temp_suffix + hard_suffix

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

    print("\n" + "=" * 80 + "\n2. Taxonomy Probability Distributions\n" + "=" * 80)
    visualize_taxonomy_distributions(model, eval_loader, device, save_dir,
                                     num_batches=num_taxon_batches)

    print("\n" + "=" * 80 + "\n2b. Taxonomy Path Probability Plots\n" + "=" * 80)
    visualize_taxonomy_path_probs(model, eval_loader, device, save_dir)

    print("\n" + "=" * 80 + "\n2c. Taxonomy Tree Activations\n" + "=" * 80)
    visualize_taxonomy_tree(model, eval_loader, device, save_dir,
                            num_images=num_act_images)

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
