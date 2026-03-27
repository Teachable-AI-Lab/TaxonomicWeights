#!/usr/bin/env python3
"""Analyze a trained SparseConvAutoencoder checkpoint.

Computes the same latent and reconstruction metrics written by the taxon and
baseline analysis pipelines so that compare_celeba_hq.py / compare_cifar10.py
can load them from pre-computed npz files:

    outputs/<run>/analysis/latent_statistics.npz
    outputs/<run>/analysis/reconstruction_metrics.npz

Additionally saves SAE-specific sparsity metrics:

    outputs/<run>/analysis/sparsity_metrics.npz

All paths are read from the config JSON (output.output_dir /
output.analysis_save_dir, data.*).  A checkpoint or save-dir override can
be passed explicitly if needed.

Usage:
    # CelebA-HQ
    python src/analyze/analyze_sae.py --config configs/sae_celeba_hq.json

    # CIFAR-10
    python src/analyze/analyze_sae.py --config configs/sae_cifar10.json

    # Custom checkpoint
    python src/analyze/analyze_sae.py \\
        --config     configs/sae_celeba_hq.json \\
        --checkpoint ./outputs/sae_celeba_hq_r18/checkpoints/epoch_50.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.baseline.sae import SparseConvAutoencoder
from src.model.cnn.baseline.topk_sae import TopKSparseConvAutoencoder
from src.model.cnn.baseline.gated_sae import GatedSparseConvAutoencoder
from src.model.cnn.baseline.jumprelu_sae import JumpReLUSparseConvAutoencoder
from src.utils.dataloader import CelebAHQLoader, CIFAR10Loader


# ---------------------------------------------------------------------------
# Output-dir helpers
# ---------------------------------------------------------------------------

def _training_output_dir_suffix(cfg: dict) -> str:
    """Reconstruct the hyperparameter suffix the training script appended.

    Training scripts never write to the bare ``output_dir`` from the config;
    they always append a suffix that encodes the key hyperparameters so that
    multiple runs co-exist.  This function replicates that logic so the
    analyze script can find the correct checkpoint directory.

    Suffix patterns:
      * TopK:  ``_k{topk_k}_sw{sparsity_weight:.0e}``
      * Gated: ``_sw{sparsity_weight:.0e}[_ste]``
      * L1/KL: ``_spw_{sparsity_weight:.0e}_spt_{sparsity_type}``
    """
    mc  = cfg.get("model", {})
    tc  = cfg.get("training", {})
    variant = mc.get("model_variant", "l1")
    sw  = tc.get("sparsity_weight", 1e-3)

    if variant == "topk":
        k = mc.get("topk_k", 64)
        return f"_k{k}_sw{sw:.0e}"
    elif variant == "gated":
        ste_tag = "_ste" if mc.get("use_gate_ste", False) else ""
        return f"_sw{sw:.0e}{ste_tag}"
    elif variant == "jumprelu":
        l0 = int(mc.get("target_l0", 64))
        return f"_l0{l0}_sw{sw:.0e}"
    else:  # l1 / kl
        spt = mc.get("sparsity_type", "l1")
        return f"_spw_{sw:.0e}_spt_{spt}"


def _resolve_output_dir(cfg: dict) -> Path:
    """Return the actual output directory including the training-script suffix.

    Falls back to the bare config path if neither the suffixed nor the bare
    directory exists (e.g. when the user provides an explicit ``--checkpoint``).
    """
    base   = Path(cfg["output"]["output_dir"])
    suffix = _training_output_dir_suffix(cfg)
    full   = Path(str(base) + suffix)
    # Prefer the suffixed path; fall back gracefully if it doesn't exist yet
    # (e.g. dry-run with --checkpoint override).
    if full.exists() or not base.exists():
        return full
    return base


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> nn.Module:
    """Load any SAE-family checkpoint (L1/KL, TopK, or Gated) by inspecting args."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    variant = a.get("model_variant", "l1")

    common = dict(
        in_channels=a.get("in_channels", 3),
        resnet_variant=a.get("resnet_variant", "18"),
        stage_channels=tuple(a.get("stage_channels", [64, 128, 256, 512])),
        stage_strides=tuple(a.get("stage_strides", [1, 2, 2, 2])),
        stage_blocks=a.get("stage_blocks", None),
        kernel_size=a.get("kernel_size", 3),
        use_stem=a.get("use_stem", True),
        stem_channels=a.get("stem_channels", 64),
        stem_stride=a.get("stem_stride", 2),
        use_stem_maxpool=a.get("use_stem_maxpool", True),
        output_activation=a.get("output_activation", "none"),
    )

    if variant == "topk":
        model = TopKSparseConvAutoencoder(
            **common,
            topk_k=a.get("topk_k", 64),
            k_aux=a.get("k_aux", 64),
            use_aux_loss=False,   # disable during analysis
            dead_threshold=a.get("dead_threshold", 1e-3),
        )
        print(f"  variant=topk  topk_k={a.get('topk_k', 64)}")
    elif variant == "gated":
        model = GatedSparseConvAutoencoder(
            **common,
            use_gate_ste=a.get("use_gate_ste", False),
        )
        print(f"  variant=gated  use_gate_ste={a.get('use_gate_ste', False)}")
    elif variant == "jumprelu":
        model = JumpReLUSparseConvAutoencoder(
            **common,
            target_l0=a.get("target_l0", 64.0),
            bandwidth=a.get("bandwidth", 0.001),
            theta_init=a.get("theta_init", 0.1),
        )
        print(f"  variant=jumprelu  target_l0={a.get('target_l0', 64.0)}  "
              f"bandwidth={a.get('bandwidth', 0.001)}")
    else:
        model = SparseConvAutoencoder(
            **common,
            sparsity_type=a.get("sparsity_type", "l1"),
            sparsity_target=a.get("sparsity_target", 0.05),
            latent_activation=a.get("latent_activation", "relu"),
        )
        print(f"  variant=l1/kl  sparsity_type={a.get('sparsity_type','l1')}  "
              f"sparsity_target={a.get('sparsity_target', 0.05)}  "
              f"latent_activation={a.get('latent_activation','relu')}")

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    best  = ckpt.get("best_val", float("nan"))
    print(f"Loaded checkpoint: epoch={epoch}  best_val={best:.6f}")
    return model


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    model: SparseConvAutoencoder,
    loader,
    device: torch.device,
    max_batches: int = 0,
    sparsity_threshold: float = 0.1,
):
    """Compute latent, reconstruction, and sparsity metrics over a data loader.

    Returns a tuple (latent_stats, recon_stats, sae_sparsity_stats).
    """
    all_latents       = []
    all_mse           = []
    all_mae           = []
    all_sparsity_loss = []  # model's own sparsity penalty per-sample estimate

    for b_idx, (imgs, _) in enumerate(tqdm(loader, desc="  Computing metrics")):
        if max_batches > 0 and b_idx >= max_batches:
            break
        imgs = imgs.to(device)

        recon, sp_loss = model(imgs)
        z, _           = model.encode(imgs)

        # Reconstruction metrics (per sample)
        all_mse.extend(((imgs - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
        all_mae.extend(torch.abs(imgs - recon).mean(dim=(1, 2, 3)).cpu().numpy())

        # Sparsity penalty value from model (scalar per batch, replicate for samples)
        all_sparsity_loss.extend(
            [float(sp_loss.detach().cpu())] * imgs.shape[0]
        )

        # Latent (flatten spatial dims)
        zn = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        all_latents.append(zn)

    Z   = np.concatenate(all_latents, axis=0)         # [N, D]
    mse = np.array(all_mse)
    mae = np.array(all_mae)
    sp_loss_arr = np.array(all_sparsity_loss)

    sp  = np.mean(np.abs(Z) < sparsity_threshold, axis=1)     # per-sample sparsity
    l0  = np.sum(np.abs(Z) > sparsity_threshold, axis=1)
    l1  = np.abs(Z).sum(axis=1)
    l2  = np.sqrt((Z ** 2).sum(axis=1))

    lts = np.mean(np.abs(Z) < sparsity_threshold, axis=0)     # lifetime sparsity [D]
    dead      = float((lts > (1 - 0.01)).mean())
    selective = float(((lts > (1 - 0.2)) & (lts <= (1 - 0.01))).mean())
    moderate  = float(((lts > (1 - 0.5)) & (lts <= (1 - 0.2))).mean())
    dense     = float((lts <= (1 - 0.5)).mean())

    act      = 1.0 - lts
    mean_act = Z.mean(axis=0)
    std_act  = Z.std(axis=0)

    latent_stats = {
        "mean_activation":    mean_act.astype(np.float32),
        "std_activation":     std_act.astype(np.float32),
        "sparsity_per_sample":sp.astype(np.float32),
        "l0_norm":            l0.astype(np.float32),
        "l1_norm":            l1.astype(np.float32),
        "l2_norm":            l2.astype(np.float32),
        "mean_sparsity":      np.float32(sp.mean()),
        "mean_l0":            np.float32(l0.mean()),
        "mean_l1":            np.float32(l1.mean()),
        "mean_l2":            np.float32(l2.mean()),
        "latent_dim":         np.int64(Z.shape[1]),
        "num_samples":        np.int64(Z.shape[0]),
        "lifetime_sparsity":  lts.astype(np.float32),
        "dead_frac":          np.float32(dead),
        "selective_frac":     np.float32(selective),
        "moderate_frac":      np.float32(moderate),
        "dense_frac":         np.float32(dense),
        "activation_rate":    act.astype(np.float32),
    }
    recon_stats = {
        "mse": mse.astype(np.float32),
        "mae": mae.astype(np.float32),
    }
    sae_sparsity_stats = {
        "sparsity_loss":      sp_loss_arr.astype(np.float32),
        "mean_sparsity_loss": np.float32(sp_loss_arr.mean()),
        "std_sparsity_loss":  np.float32(sp_loss_arr.std()),
    }
    return latent_stats, recon_stats, sae_sparsity_stats


# ---------------------------------------------------------------------------
# Analysis: Stage filters
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_stage_filters(
    model: SparseConvAutoencoder,
    save_dir: Path,
    n_cols: int = 8,
) -> None:
    """Visualise the learned filters of the first Conv2d in each encoder stage."""
    out = save_dir / "stage_filters"
    out.mkdir(parents=True, exist_ok=True)

    for s_idx, stage in enumerate(model.encoder.stages, start=1):
        conv = next((m for m in stage.modules() if isinstance(m, nn.Conv2d)), None)
        if conv is None:
            continue
        w = conv.weight.detach().cpu()  # [out_c, in_c, kH, kW]
        out_c = w.shape[0]
        n_rows = (out_c + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
        axes = np.array(axes).reshape(n_rows, n_cols)

        for f_idx in range(out_c):
            r, c = divmod(f_idx, n_cols)
            ax = axes[r, c]
            filt = w[f_idx].clone()  # [in_c, kH, kW]
            filt -= filt.min()
            mx = filt.max()
            if mx > 0:
                filt /= mx
            if filt.shape[0] == 3:
                ax.imshow(filt.permute(1, 2, 0).numpy())
            elif filt.shape[0] == 1:
                ax.imshow(filt[0].numpy(), cmap="gray")
            else:
                ax.imshow(filt.mean(0).numpy(), cmap="viridis")
            ax.axis("off")

        for f_idx in range(out_c, n_rows * n_cols):
            r, c = divmod(f_idx, n_cols)
            axes[r, c].axis("off")

        fig.suptitle(
            f"Stage {s_idx} First-Conv Filters  (out_channels={out_c}, "
            f"kernel={tuple(w.shape[2:])})",
            fontsize=9,
        )
        plt.tight_layout()
        fig.savefig(out / f"stage{s_idx}_filters.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  -> stage_filters/ written ({len(model.encoder.stages)} stages)")


# ---------------------------------------------------------------------------
# Analysis: Latent sparsity
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_latent_sparsity(
    model: SparseConvAutoencoder,
    loader,
    device: torch.device,
    save_dir: Path,
    num_batches: int = 50,
    threshold: float = 0.1,
) -> None:
    """Six-panel latent sparsity analysis plot."""
    all_latents: List[np.ndarray] = []
    all_sp_loss: List[float] = []

    for b_idx, (imgs, _) in enumerate(tqdm(loader, desc="  Latent sparsity", leave=False)):
        if b_idx >= num_batches:
            break
        imgs = imgs.to(device)
        z, _ = model.encode(imgs)
        _, sp = model(imgs)
        all_latents.append(z.detach().cpu().numpy().reshape(z.shape[0], -1))
        all_sp_loss.append(float(sp.detach().cpu()))

    Z = np.concatenate(all_latents, axis=0)  # [N, D]

    sparsity_per_dim = np.mean(np.abs(Z) < threshold, axis=0)  # [D] fraction near-zero
    act_rate = 1.0 - sparsity_per_dim
    mean_act = Z.mean(axis=0)
    std_act  = Z.std(axis=0)
    l0 = np.sum(np.abs(Z) > threshold, axis=1).astype(float)
    l1 = np.abs(Z).sum(axis=1)
    sp_arr = np.array(all_sp_loss)

    # Compute feature activity category fractions
    dead_frac     = float((act_rate < 0.01).mean())
    sel_frac      = float(((act_rate >= 0.01) & (act_rate < 0.20)).mean())
    mod_frac      = float(((act_rate >= 0.20) & (act_rate < 0.50)).mean())
    dense_frac    = float((act_rate >= 0.50).mean())

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    # (0,0) Mean activation per dim (sorted desc)
    ax = axes[0, 0]
    idx_m = np.argsort(mean_act)[::-1]
    ax.bar(range(len(mean_act)), mean_act[idx_m], width=1.0, color="steelblue", linewidth=0)
    ax.set_title("Mean Activation per Dim (sorted)")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean activation")

    # (0,1) Std activation per dim (sorted desc)
    ax = axes[0, 1]
    idx_s = np.argsort(std_act)[::-1]
    ax.bar(range(len(std_act)), std_act[idx_s], width=1.0, color="coral", linewidth=0)
    ax.set_title("Std Activation per Dim (sorted)")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Std activation")

    # (0,2) Per-dim activation rate histogram
    ax = axes[0, 2]
    ax.hist(act_rate, bins=50, color="green", edgecolor="none", alpha=0.85)
    ax.axvline(act_rate.mean(), color="red", linestyle="--",
               label=f"mean={act_rate.mean():.3f}")
    ax.set_title(f"Per-Dim Activation Rate  (dead<1%: {dead_frac:.2%})")
    ax.set_xlabel(f"Activation rate  (threshold={threshold})")
    ax.set_ylabel("# Dimensions")
    ax.legend(fontsize=8)

    # (0,3) Feature Activity Categories bar chart
    ax = axes[0, 3]
    cat_labels  = ["Dead\n(<1%)", "Selective\n(1-20%)", "Moderate\n(20-50%)", "Dense\n(>50%)"]
    cat_values  = [dead_frac * 100, sel_frac * 100, mod_frac * 100, dense_frac * 100]
    cat_colours = ["#d62728", "#2ca02c", "#ff7f0e", "#1f77b4"]
    bars = ax.bar(cat_labels, cat_values, color=cat_colours, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, cat_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylim(0, 105)
    ax.set_ylabel("% Features")
    ax.set_title("Feature Activity Categories")
    ax.grid(axis="y", alpha=0.3)

    # (1,0) L0 norm per sample
    ax = axes[1, 0]
    ax.hist(l0, bins=50, color="purple", edgecolor="none", alpha=0.85)
    ax.axvline(l0.mean(), color="red", linestyle="--", label=f"mean={l0.mean():.1f}")
    ax.set_title(f"L0 Norm per Sample  (D={Z.shape[1]})")
    ax.set_xlabel("# Active features")
    ax.set_ylabel("# Samples")
    ax.legend(fontsize=8)

    # (1,1) L1 norm per sample
    ax = axes[1, 1]
    ax.hist(l1, bins=50, color="darkorange", edgecolor="none", alpha=0.85)
    ax.axvline(l1.mean(), color="red", linestyle="--", label=f"mean={l1.mean():.2f}")
    ax.set_title("L1 Norm per Sample")
    ax.set_xlabel("L1 norm")
    ax.set_ylabel("# Samples")
    ax.legend(fontsize=8)

    # (1,2) Sparsity penalty per batch
    ax = axes[1, 2]
    ax.hist(sp_arr, bins=30, color="teal", edgecolor="none", alpha=0.85)
    ax.axvline(sp_arr.mean(), color="red", linestyle="--",
               label=f"mean={sp_arr.mean():.4f}")
    ax.set_title("Sparsity Penalty per Batch")
    ax.set_xlabel("Sparsity penalty scalar")
    ax.set_ylabel("# Batches")
    ax.legend(fontsize=8)

    # (1,3) Lifetime sparsity sorted (fraction near-zero per dim, sorted desc)
    ax = axes[1, 3]
    lts_sorted = np.sort(sparsity_per_dim)[::-1]
    ax.bar(range(len(lts_sorted)), lts_sorted, width=1.0, color="#ff7f0e", linewidth=0)
    ax.set_title("Lifetime Sparsity per Dim (sorted)")
    ax.set_xlabel("Dimension (sorted by sparsity)")
    ax.set_ylabel("Fraction near-zero")
    ax.set_ylim(0, 1)

    overall_sp = float(np.mean(np.abs(Z) < threshold))
    fig.suptitle(
        f"Latent Sparsity Analysis — N={Z.shape[0]}, D={Z.shape[1]}, "
        f"overall fraction near-zero={overall_sp:.3f}  (threshold={threshold})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(save_dir / "latent_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> latent_analysis.png written")


# ---------------------------------------------------------------------------
# Analysis: Reconstruction quality
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_reconstruction_quality(
    model: SparseConvAutoencoder,
    loader,
    device: torch.device,
    save_dir: Path,
    num_batches: int = 20,
) -> None:
    """MSE and MAE distribution plots."""
    all_mse, all_mae = [], []

    for b_idx, (imgs, _) in enumerate(tqdm(loader, desc="  Recon quality", leave=False)):
        if b_idx >= num_batches:
            break
        imgs = imgs.to(device)
        recon, _ = model(imgs)
        all_mse.extend(((imgs - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
        all_mae.extend(torch.abs(imgs - recon).mean(dim=(1, 2, 3)).cpu().numpy())

    mse = np.array(all_mse)
    mae = np.array(all_mae)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.hist(mse, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(mse.mean(), color="red",    linestyle="--", label=f"mean={mse.mean():.4f}")
    ax.axvline(np.percentile(mse, 95), color="orange", linestyle=":",
               label=f"p95={np.percentile(mse, 95):.4f}")
    ax.set_title("Per-Sample MSE")
    ax.set_xlabel("MSE")
    ax.set_ylabel("# Samples")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(mae, bins=50, color="coral", edgecolor="none", alpha=0.85)
    ax.axvline(mae.mean(), color="red",    linestyle="--", label=f"mean={mae.mean():.4f}")
    ax.axvline(np.percentile(mae, 95), color="orange", linestyle=":",
               label=f"p95={np.percentile(mae, 95):.4f}")
    ax.set_title("Per-Sample MAE")
    ax.set_xlabel("MAE")
    ax.set_ylabel("# Samples")
    ax.legend(fontsize=8)

    fig.suptitle(f"Reconstruction Quality  (N={len(mse)})", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_dir / "reconstruction_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> reconstruction_quality.png written")


# ---------------------------------------------------------------------------
# Analysis: Multiple reconstructions
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_multiple_reconstructions(
    model: SparseConvAutoencoder,
    loader,
    device: torch.device,
    save_dir: Path,
    num_images: int = 8,
    num_sets: int = 4,
) -> None:
    """Save grids of original + reconstruction side-by-side."""
    out = save_dir / "multiple_reconstructions"
    out.mkdir(parents=True, exist_ok=True)

    def denorm(t: torch.Tensor) -> np.ndarray:
        return (t.detach().cpu().permute(1, 2, 0).float().numpy() * 0.5 + 0.5).clip(0, 1)

    loader_iter = iter(loader)
    for set_idx in range(num_sets):
        try:
            imgs, _ = next(loader_iter)
        except StopIteration:
            break
        imgs  = imgs[:num_images].to(device)
        recon, _ = model(imgs)
        recon = recon.clamp(-1, 1)
        n = imgs.shape[0]

        fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.5))
        for i in range(n):
            axes[0, i].imshow(denorm(imgs[i]))
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_ylabel("Original", fontsize=7)
            axes[1, i].imshow(denorm(recon[i]))
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_ylabel("Recon", fontsize=7)

        fig.suptitle(f"Reconstructions — set {set_idx + 1}/{num_sets}", fontsize=10)
        plt.tight_layout()
        fig.savefig(out / f"set_{set_idx + 1:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  -> multiple_reconstructions/ written ({num_sets} sets)")


# ---------------------------------------------------------------------------
# Analysis: Sparsity suite (5 sub-analyses)
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_sparsity_suite(
    model: SparseConvAutoencoder,
    loader,
    device: torch.device,
    save_dir: Path,
    num_batches: int = 30,
    threshold: float = 0.1,
    n_jaccard_dims: int = 64,
    n_clusters: int = 8,
) -> None:
    """Five-part sparse representation analysis suite.

    Sub-analyses
    ============
    1. Jaccard co-activation heat-map (top-N most active dims)
    2. Feature selectivity: std, entropy, kurtosis per latent dim
    3. Causal ablations: ablate top-k% features, measure recon loss rise
    4. Cross-stage activation statistics collected via forward hooks
    5. KMeans clustering in PCA-reduced latent space
    """
    suite_dir = save_dir / "sparsity_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)

    n_stages = len(model.encoder.stages)
    stage_acts: Dict[str, List[torch.Tensor]] = {
        f"stage{i + 1}": [] for i in range(n_stages)
    }

    def make_hook(name: str):
        def _hook(m, inp, out):
            stage_acts[name].append(out.detach().cpu())
        return _hook

    hooks = [
        stage.register_forward_hook(make_hook(f"stage{s_idx}"))
        for s_idx, stage in enumerate(model.encoder.stages, start=1)
    ]

    all_latents: List[torch.Tensor] = []
    all_imgs:    List[torch.Tensor] = []

    for b_idx, (imgs, _) in enumerate(
        tqdm(loader, desc="  Sparsity suite", leave=False)
    ):
        if b_idx >= num_batches:
            break
        imgs = imgs.to(device)
        z, _ = model.encode(imgs)
        all_latents.append(z.detach().cpu())
        all_imgs.append(imgs.detach().cpu())

    for h in hooks:
        h.remove()

    Z_t  = torch.cat(all_latents, dim=0)                    # [N, C, H', W']
    Zf   = Z_t.numpy().reshape(Z_t.shape[0], -1)            # [N, D]
    imgs_all = torch.cat(all_imgs, dim=0)                   # [N, C, H, W]

    # ------------------------------------------------------------------
    # Sub-analysis 1: Jaccard co-activation heat-map
    # ------------------------------------------------------------------
    mean_act_per_dim = np.abs(Zf).mean(axis=0)
    top_idx = np.argsort(mean_act_per_dim)[::-1][:n_jaccard_dims]
    Z_top = (np.abs(Zf[:, top_idx]) > threshold).astype(float)  # [N, K] binary

    K = Z_top.shape[1]
    jaccard = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            inter = (Z_top[:, i] * Z_top[:, j]).sum()
            union = ((Z_top[:, i] + Z_top[:, j]) >= 1).sum()
            jaccard[i, j] = inter / (union + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(jaccard, vmin=0, vmax=1, aspect="auto", cmap="hot")
    plt.colorbar(im, ax=ax)
    ax.set_title(
        f"Jaccard Co-activation  (top-{n_jaccard_dims} active dims, threshold={threshold})"
    )
    ax.set_xlabel("Dim index (sorted by mean activation)")
    ax.set_ylabel("Dim index (sorted by mean activation)")
    plt.tight_layout()
    fig.savefig(suite_dir / "01_jaccard_coactivation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 01_jaccard_coactivation.png")

    # ------------------------------------------------------------------
    # Sub-analysis 2: Feature selectivity (std, entropy, kurtosis per dim)
    # ------------------------------------------------------------------
    dim_std  = Zf.std(axis=0)
    # Entropy: normalise activations per dim to a probability distribution
    act_abs = np.abs(Zf)
    col_sum = act_abs.sum(axis=0) + 1e-8
    P = act_abs / col_sum[None, :]                            # [N, D]
    # Clip to avoid log(0)
    P = np.clip(P, 1e-12, None)
    dim_entropy = -(P * np.log(P)).sum(axis=0)               # [D]
    dim_kurt    = scipy_kurtosis(Zf, axis=0, bias=True)      # [D]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.hist(dim_std, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(dim_std.mean(), color="red", linestyle="--",
               label=f"mean={dim_std.mean():.3f}")
    ax.set_title("Per-Dim Activation Std  (selectivity)")
    ax.set_xlabel("Std")
    ax.set_ylabel("# Dims")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(dim_entropy, bins=50, color="mediumseagreen", edgecolor="none", alpha=0.85)
    ax.axvline(dim_entropy.mean(), color="red", linestyle="--",
               label=f"mean={dim_entropy.mean():.2f}")
    ax.set_title("Per-Dim Activation Entropy")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("# Dims")
    ax.legend(fontsize=8)

    ax = axes[2]
    dim_kurt_finite = dim_kurt[np.isfinite(dim_kurt)]
    kurt_data = dim_kurt_finite if len(dim_kurt_finite) > 0 else np.array([0.0])
    ax.hist(kurt_data, bins=50, color="darkorange", edgecolor="none", alpha=0.85)
    kurt_mean = float(dim_kurt_finite.mean()) if len(dim_kurt_finite) > 0 else float("nan")
    if np.isfinite(kurt_mean):
        ax.axvline(kurt_mean, color="red", linestyle="--",
                   label=f"mean={kurt_mean:.2f}")
        ax.legend(fontsize=8)
    ax.set_title("Per-Dim Activation Kurtosis")
    ax.set_xlabel("Kurtosis")
    ax.set_ylabel("# Dims")

    fig.suptitle("Feature Selectivity Analysis", fontsize=11)
    plt.tight_layout()
    fig.savefig(suite_dir / "02_feature_selectivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 02_feature_selectivity.png")

    # ------------------------------------------------------------------
    # Sub-analysis 3: Causal ablations (zero top-k% features → recon ↑)
    # ------------------------------------------------------------------
    D = Zf.shape[1]
    ablation_fracs = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
    # Sort dims by mean abs activation (most → least important)
    sorted_dims = np.argsort(mean_act_per_dim)[::-1].copy()

    abl_mse_mean, abl_mse_std = [], []
    n_abl_samples = min(256, len(imgs_all))

    imgs_sub = imgs_all[:n_abl_samples].to(device)
    Z_sub    = Z_t[:n_abl_samples].to(device)

    for frac in ablation_fracs:
        n_kill = int(frac * D)
        z_abl  = Z_sub.clone()
        if n_kill > 0:
            kill_dims = sorted_dims[:n_kill]
            # Zero those dims across all spatial positions
            z_abl = z_abl.reshape(z_abl.shape[0], D, -1)
            z_abl[:, kill_dims, :] = 0.0
            z_abl = z_abl.reshape(Z_sub.shape)
        recon_abl, _ = model.decode(z_abl, output_size=tuple(imgs_sub.shape[-2:]))
        recon_abl = recon_abl.clamp(-1, 1)
        mse_abl = ((imgs_sub - recon_abl) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        abl_mse_mean.append(mse_abl.mean())
        abl_mse_std.append(mse_abl.std())

    abl_mse_mean = np.array(abl_mse_mean)
    abl_mse_std  = np.array(abl_mse_std)

    fig, ax = plt.subplots(figsize=(7, 4))
    pct = [f * 100 for f in ablation_fracs]
    ax.plot(pct, abl_mse_mean, "o-", color="steelblue", label="Mean MSE")
    ax.fill_between(
        pct,
        abl_mse_mean - abl_mse_std,
        abl_mse_mean + abl_mse_std,
        alpha=0.2, color="steelblue",
    )
    ax.set_title(f"Causal Ablation: MSE vs Fraction of Dims Zeroed\n"
                 f"(dims sorted by mean |activation|, N={n_abl_samples})")
    ax.set_xlabel("% of latent dims zeroed (most active first)")
    ax.set_ylabel("Reconstruction MSE")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(suite_dir / "03_causal_ablations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 03_causal_ablations.png")

    # ------------------------------------------------------------------
    # Sub-analysis 4: Cross-stage activation statistics
    # ------------------------------------------------------------------
    # Build per-stage pooled activation vector for each sample
    stage_pooled = {}
    for sname, acts_list in stage_acts.items():
        acts = torch.cat(acts_list, dim=0)              # [N, C, H, W]
        pooled = acts.mean(dim=(2, 3)).numpy()          # [N, C]  spatial mean
        stage_pooled[sname] = pooled

    stage_names = sorted(stage_pooled.keys())
    n_s = len(stage_names)

    # Correlation of per-stage mean activation vectors (across samples)
    stage_means = np.stack(
        [stage_pooled[s].mean(axis=1) for s in stage_names], axis=1
    )  # [N, n_stages]

    corr = np.corrcoef(stage_means.T)  # [n_stages, n_stages]

    # Sparsity (fraction near-zero) across samples per stage
    stage_sparsity = {
        s: float(np.mean(np.abs(stage_pooled[s]) < threshold))
        for s in stage_names
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n_s))
    ax.set_yticks(range(n_s))
    ax.set_xticklabels(stage_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(stage_names, fontsize=8)
    for i in range(n_s):
        for j in range(n_s):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_title("Cross-Stage Mean Activation Correlation")

    ax = axes[1]
    sp_vals = [stage_sparsity[s] for s in stage_names]
    ch_counts = [stage_pooled[s].shape[1] for s in stage_names]
    bars = ax.bar(stage_names, sp_vals, color="steelblue", alpha=0.8)
    for bar, ch in zip(bars, ch_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"C={ch}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0, 1)
    ax.set_title(f"Feature Sparsity per Stage  (threshold={threshold})")
    ax.set_ylabel("Fraction near-zero")
    ax.set_xlabel("Stage")

    fig.suptitle("Cross-Stage Dependency Analysis", fontsize=11)
    plt.tight_layout()
    fig.savefig(suite_dir / "04_cross_stage_dependency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 04_cross_stage_dependency.png")

    # ------------------------------------------------------------------
    # Sub-analysis 5: KMeans clustering in PCA-reduced latent space
    # ------------------------------------------------------------------
    pca = PCA(n_components=min(50, Zf.shape[1], Zf.shape[0] - 1))
    Zf_pca = pca.fit_transform(Zf)                          # [N, 50]

    sil_scores = []
    ks = [4, 6, 8, 10, 12, 16]
    for k in ks:
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = km.fit_predict(Zf_pca)
        try:
            sil = float(silhouette_score(Zf_pca, labels, sample_size=min(2000, len(Zf_pca))))
        except Exception:
            sil = float("nan")
        sil_scores.append(sil)

    best_k = ks[int(np.nanargmax(sil_scores))]
    km_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels_best = km_best.fit_predict(Zf_pca)

    # 2-D PCA for scatter
    Z2d = PCA(n_components=2).fit_transform(Zf)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(ks, sil_scores, "o-", color="steelblue")
    ax.axvline(best_k, color="red", linestyle="--", label=f"best k={best_k}")
    ax.set_title("KMeans Silhouette Score vs k")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Silhouette score")
    ax.legend(fontsize=8)

    ax = axes[1]
    cmap = plt.get_cmap("tab20")
    for cl in range(best_k):
        mask = labels_best == cl
        ax.scatter(
            Z2d[mask, 0], Z2d[mask, 1],
            s=4, alpha=0.5,
            color=cmap(cl / best_k),
            label=f"C{cl} (n={mask.sum()})",
        )
    ax.set_title(f"2-D PCA of Latent Codes  (k={best_k}, sil={sil_scores[ks.index(best_k)]:.3f})")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=6, ncol=2, markerscale=2)

    fig.suptitle(
        f"KMeans Clustering of SAE Latent Space\n"
        f"(PCA var explained @ 50 PCs: {pca.explained_variance_ratio_.sum():.2%})",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(suite_dir / "05_dimension_clustering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> 05_dimension_clustering.png")


# ---------------------------------------------------------------------------
# Analysis: Stage activation maps
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_stage_activations(
    model: SparseConvAutoencoder,
    loader,
    device: torch.device,
    save_dir: Path,
    num_images: int = 3,
    max_channels_per_stage: int = 64,
) -> None:
    """Feature-map grids and composite summary figures at each encoder stage.

    Per image produces:
    - ``original.png``            — the input image
    - ``summary.png``             — one composite figure: original | per-stage mean
                                    & top-channel maps | reconstruction
    - ``stage{N}.png``            — full channel grid (up to max_channels_per_stage)
    """
    out = save_dir / "stage_activations"
    out.mkdir(parents=True, exist_ok=True)

    imgs, _ = next(iter(loader))
    imgs = imgs[:num_images].to(device)

    def denorm(t: torch.Tensor) -> np.ndarray:
        """[-1,1] tensor → (H,W,3) float32 in [0,1]."""
        return (t.detach().cpu().float() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

    def norm_map(fmap: np.ndarray) -> np.ndarray:
        """Min-max normalise a single 2-D feature map to [0,1]."""
        lo, hi = fmap.min(), fmap.max()
        return (fmap - lo) / (hi - lo + 1e-8)

    n_stages = len(model.encoder.stages)

    for img_idx in range(imgs.shape[0]):
        img_dir = out / f"image_{img_idx + 1:02d}"
        img_dir.mkdir(exist_ok=True)

        img_t = imgs[img_idx : img_idx + 1]  # [1, C, H, W]

        # ── save original ────────────────────────────────────────────────
        orig_np = denorm(imgs[img_idx])
        plt.imsave(str(img_dir / "original.png"), orig_np)

        # ── collect stage feature maps ───────────────────────────────────
        stage_maps: List[np.ndarray] = []   # list of [C, H', W'] numpy arrays
        x = model.encoder.stem(img_t)
        for stage in model.encoder.stages:
            x = stage(x)
            stage_maps.append(x[0].detach().cpu().numpy())  # [C, H', W']

        # ── reconstruction ───────────────────────────────────────────────
        with torch.no_grad():
            recon, _ = model(img_t)
        recon_np = denorm(recon[0])

        # ── composite summary figure ─────────────────────────────────────
        # Layout (2 rows × (n_stages + 2) cols):
        #   Row 0: Original | stage1 top-channel | ... | stageN top-channel | Recon
        #   Row 1: (blank)  | stage1 mean map    | ... | stageN mean map    | (blank)
        n_cols_fig = n_stages + 2
        fig_s, axes_s = plt.subplots(
            2, n_cols_fig,
            figsize=(n_cols_fig * 2.0, 4.5),
            gridspec_kw={"hspace": 0.05, "wspace": 0.05},
        )

        # col 0 row 0 — original
        axes_s[0, 0].imshow(orig_np)
        axes_s[0, 0].set_title("Input", fontsize=8)
        axes_s[0, 0].axis("off")
        axes_s[1, 0].axis("off")

        for s_idx, smaps in enumerate(stage_maps):
            col = s_idx + 1
            # mean across channels
            mean_map = norm_map(smaps.mean(axis=0))
            # top-activated channel (highest mean absolute value)
            top_ch   = int(np.abs(smaps).mean(axis=(1, 2)).argmax())
            top_map  = norm_map(smaps[top_ch])

            axes_s[0, col].imshow(top_map, cmap="viridis")
            axes_s[0, col].set_title(
                f"S{s_idx + 1} top-ch (#{top_ch})\n"
                f"{smaps.shape[0]}ch {smaps.shape[1]}×{smaps.shape[2]}",
                fontsize=7,
            )
            axes_s[0, col].axis("off")

            axes_s[1, col].imshow(mean_map, cmap="viridis")
            axes_s[1, col].set_title(f"S{s_idx + 1} mean", fontsize=7)
            axes_s[1, col].axis("off")

        # last col — reconstruction
        axes_s[0, -1].imshow(recon_np)
        axes_s[0, -1].set_title("Recon", fontsize=8)
        axes_s[0, -1].axis("off")
        axes_s[1, -1].axis("off")

        fig_s.suptitle(
            f"Image {img_idx + 1} — encoder activation summary",
            fontsize=9, y=1.01,
        )
        fig_s.savefig(img_dir / "summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig_s)

        # ── per-stage full channel grids ─────────────────────────────────
        for s_idx, smaps in enumerate(stage_maps, start=1):
            n_ch   = min(smaps.shape[0], max_channels_per_stage)
            n_cols = 8
            n_rows = (n_ch + n_cols - 1) // n_cols

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(n_cols * 1.2, n_rows * 1.2),
            )
            axes = np.array(axes).reshape(n_rows, n_cols)

            for c in range(n_ch):
                r, cc = divmod(c, n_cols)
                axes[r, cc].imshow(norm_map(smaps[c]), cmap="viridis")
                axes[r, cc].axis("off")
            for c in range(n_ch, n_rows * n_cols):
                r, cc = divmod(c, n_cols)
                axes[r, cc].axis("off")

            fig.suptitle(
                f"Stage {s_idx} — {n_ch}/{smaps.shape[0]} channels  "
                f"spatial={smaps.shape[1]}×{smaps.shape[2]}",
                fontsize=9,
            )
            plt.tight_layout()
            fig.savefig(img_dir / f"stage{s_idx}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)

        print(f"  Image {img_idx + 1}: saved to {img_dir}")

    print(f"  -> stage_activations/ written ({num_images} images × {n_stages} stages)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _is_cifar(cfg: dict) -> bool:
    """Infer dataset type from the config (image_size==32 → CIFAR-10)."""
    return cfg.get("data", {}).get("image_size", 256) == 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a trained SparseConvAutoencoder")
    p.add_argument("--config",      type=str, required=True,
                   help="Path to the experiment JSON config (e.g. configs/sae_celeba_hq.json)")
    p.add_argument("--checkpoint",  type=str, default="",
                   help="Override checkpoint path (default: output_dir/checkpoints/best.pt)")
    p.add_argument("--save-dir",    type=str, default="",
                   help="Override analysis save dir (default: from config output.analysis_save_dir)")
    p.add_argument("--max-batches", type=int, default=0,
                   help="Limit eval to N batches (0 = all)")
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    with open(args.config) as f:
        cfg = json.load(f)

    output_dir = _resolve_output_dir(cfg)
    ckpt_path  = Path(args.checkpoint) if args.checkpoint else output_dir / "checkpoints" / "best.pt"
    # Derive analysis save dir from the resolved (suffixed) output_dir so results
    # land alongside the checkpoint rather than in the bare base directory.
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = output_dir / "analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Save dir:   {save_dir}")

    model = load_model(ckpt_path, device)

    dc = cfg.get("data", {})
    if not _is_cifar(cfg):
        image_size = dc.get("image_size", 256)
        tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        loader_obj = CelebAHQLoader(
            data_root=dc["data_root"],
            batch_size=dc.get("batch_size", 32),
            num_workers=dc.get("num_workers", 4),
            image_size=image_size,
            val_split=dc.get("val_split", 0.05),
            seed=42,
            pin_memory=(device.type == "cuda"),
            transform=tf,
        )
        _, eval_loader = loader_obj.get_loaders()
        dataset_name = "celeba"
    else:
        loader_obj = CIFAR10Loader(
            batch_size=dc.get("batch_size", 128),
            root=dc["data_root"],
            num_workers=dc.get("num_workers", 4),
        )
        _, eval_loader = loader_obj.get_loaders()
        dataset_name = "cifar10"

    print(f"Computing metrics on {dataset_name} (device={device}) ...")
    latent_stats, recon_stats, sae_sparsity_stats = compute_metrics(
        model, eval_loader, device, max_batches=args.max_batches
    )

    np.savez_compressed(save_dir / "latent_statistics.npz",      **latent_stats)
    np.savez_compressed(save_dir / "reconstruction_metrics.npz", **recon_stats)
    np.savez_compressed(save_dir / "sparsity_metrics.npz",       **sae_sparsity_stats)

    print(f"\nSaved npz files to {save_dir}/")
    print(f"  mean_mse={recon_stats['mse'].mean():.6f}  mean_mae={recon_stats['mae'].mean():.6f}")
    print(f"  mean_sparsity={float(latent_stats['mean_sparsity']):.4f}  mean_l0={float(latent_stats['mean_l0']):.1f}")
    print(f"  dead={float(latent_stats['dead_frac']):.3f}  selective={float(latent_stats['selective_frac']):.3f}  dense={float(latent_stats['dense_frac']):.3f}")
    print(f"  mean_sparsity_loss={float(sae_sparsity_stats['mean_sparsity_loss']):.6f}  "
          f"std={float(sae_sparsity_stats['std_sparsity_loss']):.6f}")

    # ------------------------------------------------------------------
    # Rich visual analyses
    # ------------------------------------------------------------------
    print("\nRunning visual analyses ...")

    print("[1/6] Stage filter visualisation")
    visualize_stage_filters(model, save_dir)

    print("[2/6] Latent sparsity analysis")
    analyze_latent_sparsity(model, eval_loader, device, save_dir,
                            num_batches=min(50, args.max_batches or 50))

    print("[3/6] Reconstruction quality plots")
    analyze_reconstruction_quality(model, eval_loader, device, save_dir,
                                   num_batches=min(20, args.max_batches or 20))

    print("[4/6] Multiple reconstruction grids")
    visualize_multiple_reconstructions(model, eval_loader, device, save_dir)

    print("[5/6] Sparsity suite (Jaccard, selectivity, ablations, cross-stage, clustering)")
    analyze_sparsity_suite(model, eval_loader, device, save_dir,
                           num_batches=min(30, args.max_batches or 30))

    print("[6/6] Stage activation maps")
    visualize_stage_activations(model, eval_loader, device, save_dir)

    print(f"\nAll analyses written to {save_dir}/")


if __name__ == "__main__":
    main()
