#!/usr/bin/env python3
"""Final comparative analysis: all CelebA-HQ model runs.

Discovers every run directory under ``outputs/`` that looks like a CelebA-HQ
experiment (name contains ``celeba``), loads its best checkpoint and any
pre-computed analysis artifacts (npz files from ``analyze_celeba_hq_ae.py``),
and produces a single multi-page summary comparing:

  Page 1 — Reconstruction Quality
    • Training curves (val-recon per epoch)   for every run with a history.json
    • Bar chart: best-val MSE per model
    • Sample reconstructions: one row of 8 images per model (original + recon)

  Page 2 — Sparsity Analysis
    • Bar chart: mean latent sparsity (fraction of near-zero activations)
    • Bar chart: mean L0 norm (active features count)
    • Bar chart: dead / selective / dense feature fractions per model
    • Lifetime-sparsity histogram comparison across all models

  Page 3 — Feature Quality
    • Bar chart: mean pairwise Jaccard overlap
    • Bar chart: feature selectivity index
    • Bar chart: kurtosis (peakedness of feature distributions)
    • Bar chart: p50 / p90 units from causal ablation (concentration of importance)

For any run whose pre-computed npz files are missing, the relevant metrics are
computed live (requires data access from ``--data-root``).

Output: ``outputs/comparison_celeba_hq/comparison_celeba_hq.png``
         (plus per-page PNGs and a summary CSV).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.taxon_ae import TaxonAutoencoder
from src.model.sae import SparseConvAutoencoder
from src.model.baseline_ae import BaselineConvAutoencoder
from src.utils.dataloader import CelebAHQLoader
from torchvision import transforms  # noqa: F401 (used in transform construction)


# ─── colour palette ────────────────────────────────────────────────────────────
# taxon = blue family, sae = orange family, baseline = green family.
_TAXON_PALETTE = [
    "#1f77b4", "#4e9fc7", "#1a5a8a", "#6baed6",
    "#08519c", "#2171b5", "#4292c6", "#74add1",
    "#9ecae1",
]
_SAE_PALETTE = [
    "#ff7f0e", "#e55e00", "#ffa64d", "#cc4a00",
    "#ff9933", "#cc7a00", "#ffb366",
]
_BASELINE_PALETTE = [
    "#2ca02c", "#1a7a1a", "#5fd35f", "#3d8c3d",
]


def model_colour(run_name: str, model_type: str, idx_within_type: int) -> str:
    if model_type == "taxon":
        return _TAXON_PALETTE[idx_within_type % len(_TAXON_PALETTE)]
    if model_type == "sae":
        return _SAE_PALETTE[idx_within_type % len(_SAE_PALETTE)]
    return _BASELINE_PALETTE[idx_within_type % len(_BASELINE_PALETTE)]


# ─── helpers ───────────────────────────────────────────────────────────────────

def _short_name(run_dir: str) -> str:
    """Make a readable short name for plot labels."""
    n = run_dir
    for prefix in ("taxon_ae_celeba_hq_r18_", "sae_celeba_hq_r18_",
                   "baseline_ae_celeba_hq_r18", "baseline_ae_celeba_hq"):
        n = n.replace(prefix, "").strip("_")
    n = n.replace("_", " ").strip()
    return n or "baseline"


def _model_type(run_dir: str) -> str:
    if run_dir.startswith("sae_"):
        return "sae"
    if run_dir.startswith("baseline_"):
        return "baseline"
    return "taxon"


def _to_display(t: torch.Tensor) -> np.ndarray:
    """Convert [-1,1] or [0,1] tensor (1,C,H,W) -> (H,W,C) displayable."""
    img = t.detach().cpu().squeeze(0).permute(1, 2, 0)
    img = img * 0.5 + 0.5   # assume [-1,1] normalisation
    return img.clamp(0, 1).numpy()


# ─── discovery ────────────────────────────────────────────────────────────────

def discover_runs(outputs_dir: Path) -> List[Dict]:
    """Return a list of run-info dicts, sorted by type then name."""
    runs = []
    for name in sorted(os.listdir(outputs_dir)):
        if "celeba" not in name.lower():
            continue
        run_path = outputs_dir / name
        if not run_path.is_dir():
            continue
        best_ckpt = run_path / "checkpoints" / "best.pt"
        if not best_ckpt.exists():
            continue
        runs.append({
            "name":       name,
            "short":      _short_name(name),
            "type":       _model_type(name),
            "path":       run_path,
            "best_ckpt":  best_ckpt,
            "history":    run_path / "training_history.json",
            "analysis":   run_path / "analysis",
        })
    # sort: taxon first, then sae, then baseline
    _order = {"taxon": 0, "sae": 1, "baseline": 2}
    runs.sort(key=lambda r: (_order.get(r["type"], 9), r["name"]))
    taxon_i = sae_i = baseline_i = 0
    for r in runs:
        if r["type"] == "taxon":
            r["colour"] = model_colour(r["name"], "taxon",    taxon_i);    taxon_i    += 1
        elif r["type"] == "sae":
            r["colour"] = model_colour(r["name"], "sae",      sae_i);      sae_i      += 1
        else:
            r["colour"] = model_colour(r["name"], "baseline", baseline_i); baseline_i += 1
    return runs


# ─── model loading ─────────────────────────────────────────────────────────────

def load_taxon_model(ckpt_path: Path, device: torch.device) -> Tuple[TaxonAutoencoder, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = TaxonAutoencoder(
        in_channels=a.get("in_channels", 3),
        resnet_variant=a.get("resnet_variant", "18"),
        stage_taxonomy_layers=tuple(a.get("stage_taxonomy_layers", [5, 6, 7, 8])),
        stage_strides=tuple(a.get("stage_strides", [1, 2, 2, 2])),
        stage_blocks=a.get("stage_blocks", None),
        temperature=a.get("temperature", 1.0),
        hard=a.get("hard", False),
        kernel_size=a.get("kernel_size", 3),
        use_stem=a.get("use_stem", True),
        stem_channels=a.get("stem_channels", 64),
        stem_stride=a.get("stem_stride", 2),
        use_stem_maxpool=a.get("use_stem_maxpool", True),
        output_activation=a.get("output_activation", "none"),
        depth_decay=a.get("depth_decay", 0.5),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return model, ckpt


def load_sae_model(ckpt_path: Path, device: torch.device) -> Tuple[SparseConvAutoencoder, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = SparseConvAutoencoder(
        in_channels=a.get("in_channels", 3),
        resnet_variant=a.get("resnet_variant", "18"),
        stage_channels=tuple(a.get("stage_channels", [64, 128, 256, 512])),
        stage_strides=tuple(a.get("stage_strides", [1, 2, 2, 2])),
        stage_blocks=a.get("stage_blocks", None),
        sparsity_type=a.get("sparsity_type", "l1"),
        sparsity_target=a.get("sparsity_target", 0.05),
        latent_activation=a.get("latent_activation", "relu"),
        kernel_size=a.get("kernel_size", 3),
        use_stem=a.get("use_stem", True),
        stem_channels=a.get("stem_channels", 64),
        stem_stride=a.get("stem_stride", 2),
        use_stem_maxpool=a.get("use_stem_maxpool", True),
        output_activation=a.get("output_activation", "none"),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return model, ckpt


def load_baseline_model(ckpt_path: Path, device: torch.device) -> Tuple[BaselineConvAutoencoder, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})
    model = BaselineConvAutoencoder(
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
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device).eval()
    return model, ckpt


def load_model(run: Dict, device: torch.device):
    if run["type"] == "taxon":
        return load_taxon_model(run["best_ckpt"], device)
    if run["type"] == "sae":
        return load_sae_model(run["best_ckpt"], device)
    return load_baseline_model(run["best_ckpt"], device)


# ─── live metric computation (for runs without pre-computed npz) ───────────────

@torch.no_grad()
def compute_live_metrics(
    run: Dict,
    device: torch.device,
    val_loader,
    n_batches_latent: int = 20,
    n_batches_recon:  int = 10,
    sparsity_threshold: float = 0.1,
) -> Dict:
    """Compute reconstruction MSE and latent sparsity from scratch."""
    model, ckpt = load_model(run, device)
    model.eval()

    mse_list, mae_list = [], []
    latents = []

    for b_idx, (imgs, _) in enumerate(tqdm(val_loader, desc=f"  live eval {run['short']}", leave=False)):
        imgs = imgs.to(device)
        if run["type"] == "taxon":
            recon, _, _ = model(imgs)
        else:
            recon, _    = model(imgs)
        z, _ = model.encode(imgs)

        if b_idx < n_batches_recon:
            mse_list.extend(((imgs - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
            mae_list.extend(torch.abs(imgs - recon).mean(dim=(1, 2, 3)).cpu().numpy())

        if b_idx < n_batches_latent:
            zn = z.detach().cpu().numpy()
            latents.append(zn.reshape(zn.shape[0], -1))

        if b_idx >= max(n_batches_latent, n_batches_recon):
            break

    mse  = np.array(mse_list)
    mae  = np.array(mae_list)
    Z    = np.concatenate(latents, axis=0)
    sp   = np.mean(np.abs(Z) < sparsity_threshold, axis=1)
    l0   = np.sum(np.abs(Z) > sparsity_threshold, axis=1)
    l1   = np.abs(Z).sum(axis=1)
    l2   = np.sqrt((Z ** 2).sum(axis=1))
    lts  = np.mean(np.abs(Z) < sparsity_threshold, axis=0)  # lifetime sparsity per dim
    dead     = float((lts > (1 - 0.01)).mean())
    selective= float(((lts <= (1 - 0.01)) & (lts > (1 - 0.2))).mean())
    dense    = float((lts < (1 - 0.5)).mean())
    moderate = max(0.0, 1.0 - dead - selective - dense)

    return {
        "mean_mse":       float(mse.mean()),
        "mean_mae":       float(mae.mean()),
        "mean_sparsity":  float(sp.mean()),
        "mean_l0":        float(l0.mean()),
        "mean_l1":        float(l1.mean()),
        "mean_l2":        float(l2.mean()),
        "dead_frac":      dead,
        "selective_frac": selective,
        "moderate_frac":  moderate,
        "dense_frac":     dense,
        "lifetime_sparsity": lts,
        "jaccard":        None,   # not computed live (expensive)
        "selectivity_idx":None,
        "mean_kurtosis":  None,
        "p50_units":      None,
        "p90_units":      None,
    }


def load_precomputed_metrics(run: Dict) -> Optional[Dict]:
    """Load metrics from pre-computed npz files if they exist."""
    adir = run["analysis"]
    lat_f   = adir / "latent_statistics.npz"
    recon_f = adir / "reconstruction_metrics.npz"

    if not (lat_f.exists() and recon_f.exists()):
        return None

    ls = np.load(lat_f, allow_pickle=True)
    rm = np.load(recon_f, allow_pickle=True)

    result = {
        "mean_mse":       float(rm["mse"].mean()),
        "mean_mae":       float(rm["mae"].mean()),
        "mean_sparsity":  float(ls["mean_sparsity"]),
        "mean_l0":        float(ls["mean_l0"]),
        "mean_l1":        float(ls["mean_l1"]),
        "mean_l2":        float(ls["mean_l2"]),
        "lifetime_sparsity": None,
        "dead_frac":      None,
        "selective_frac": None,
        "moderate_frac":  None,
        "dense_frac":     None,
        "jaccard":        None,
        "selectivity_idx":None,
        "mean_kurtosis":  None,
        "p50_units":      None,
        "p90_units":      None,
    }

    # Partonomy sparsity npz files
    pdir = adir / "partonomy_sparsity"
    jac_f  = pdir / "01_jaccard_stats.npz"
    sel_f  = pdir / "02_selectivity_stats.npz"
    abl_f  = pdir / "03_ablation_importance.npz"
    if jac_f.exists():
        jd = np.load(jac_f, allow_pickle=True)
        result["jaccard"]        = float(jd["jaccard"].mean())
        result["dead_frac"]      = float(jd["dead_frac"])
        result["selective_frac"] = float(jd["selective_frac"])
        result["dense_frac"]     = float(jd["dense_frac"])
        result["moderate_frac"]  = max(0.0, 1.0 - result["dead_frac"]
                                       - result["selective_frac"] - result["dense_frac"])
        result["lifetime_sparsity"] = jd["lifetime_sparsity"]
    if sel_f.exists():
        sd = np.load(sel_f, allow_pickle=True)
        result["selectivity_idx"] = float(sd["selectivity_idx"].mean())
        result["mean_kurtosis"]   = float(sd["per_dim_kurtosis"].mean())
    if abl_f.exists():
        ad = np.load(abl_f, allow_pickle=True)
        result["p50_units"] = int(ad["p50_units"])
        result["p90_units"] = int(ad["p90_units"])

    return result


@torch.no_grad()
def collect_reconstructions(
    run: Dict,
    device: torch.device,
    val_loader,
    n_images: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (originals, reconstructions) arrays, each shape (n_images,H,W,3)."""
    model, _ = load_model(run, device)
    model.eval()
    imgs, _ = next(iter(val_loader))
    imgs = imgs[:n_images].to(device)

    with torch.no_grad():
        if run["type"] == "taxon":
            recon, _, _ = model(imgs)
        else:
            recon, _ = model(imgs)

    origs  = np.stack([_to_display(imgs[i:i+1])  for i in range(n_images)])
    recons = np.stack([_to_display(recon[i:i+1]) for i in range(n_images)])
    return origs, recons


# ─── plotting helpers ──────────────────────────────────────────────────────────

def _bar_chart(
    ax,
    values: List[Optional[float]],
    labels: List[str],
    colours: List[str],
    title: str,
    ylabel: str,
    lower_better: bool = False,
) -> None:
    valid = [(v, l, c) for v, l, c in zip(values, labels, colours) if v is not None]
    if not valid:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title); return
    vals, lbls, cols = zip(*valid)
    bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="black", linewidth=0.5)
    best_i = int(np.argmin(vals) if lower_better else np.argmax(vals))
    bars[best_i].set_edgecolor("red"); bars[best_i].set_linewidth(2.0)
    ax.set_xticks(range(len(lbls)))
    ax.set_xticklabels(lbls, rotation=35, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _add_legend(fig, runs: List[Dict]) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=r["colour"], edgecolor="black", linewidth=0.5)
        for r in runs
    ]
    labels = [r["short"] for r in runs]
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(runs), 4), fontsize=7,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9)


# ─── per-page figures ──────────────────────────────────────────────────────────

def make_page1_reconstruction(
    runs: List[Dict],
    all_metrics: List[Dict],
    save_dir: Path,
    device: torch.device,
    val_loader,
    n_recon_images: int = 6,
) -> None:
    """Page 1: Training curves + MSE/MAE bar charts + sample reconstructions."""

    n_runs = len(runs)
    labels  = [r["short"] for r in runs]
    colours = [r["colour"] for r in runs]

    # ---- training curves ----
    runs_with_history = [r for r in runs if r["history"].exists()]

    fig_curves, ax_c = plt.subplots(1, 1, figsize=(14, 5))

    Y_CAP = 0.05  # clip below this; anything above is annotated as text
    y_top = Y_CAP

    curve_data = []
    for r in runs_with_history:
        with open(r["history"]) as f:
            h = json.load(f)
        ys = np.array(h["val_recon"], dtype=float)
        xs = np.array(h["epochs"])
        curve_data.append((r, xs, ys))

    for r, xs, ys in curve_data:
        # Clip values to the visible window for plotting
        ys_clipped = np.where(ys <= y_top, ys, np.nan)
        ax_c.plot(xs, ys_clipped,
                  label=r["short"], color=r["colour"], linewidth=1.8)

        # Annotate each out-of-range point with its actual value as text
        oob_mask = np.isfinite(ys) & (ys > y_top)
        for xi, yi in zip(xs[oob_mask], ys[oob_mask]):
            ax_c.annotate(
                f"{yi:.2e}",
                xy=(xi, y_top),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=6,
                color=r["colour"],
                arrowprops=dict(arrowstyle="->", color=r["colour"], lw=0.8),
            )

    ax_c.set_ylim(0, y_top)
    ax_c.set_title("Validation Reconstruction Loss (MSE) Over Epochs",
                   fontsize=12, fontweight="bold")
    ax_c.set_xlabel("Epoch"); ax_c.set_ylabel("Val MSE")
    ax_c.legend(fontsize=7, ncol=2)
    ax_c.grid(alpha=0.3)
    fig_curves.tight_layout()
    fig_curves.savefig(save_dir / "p1_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig_curves)
    print("  Training curves saved.")

    # ---- best-val MSE / MAE bar charts ----
    best_mses = [m["mean_mse"] for m in all_metrics]
    best_maes = [m["mean_mae"] for m in all_metrics]
    ckpt_mses = []
    for r in runs:
        ckpt = torch.load(r["best_ckpt"], map_location="cpu", weights_only=False)
        ckpt_mses.append(ckpt.get("val_stats", {}).get("recon", None))

    fig_bar, axes = plt.subplots(1, 3, figsize=(18, 5))
    _bar_chart(axes[0], ckpt_mses, labels, colours,
               "Best Val MSE (from checkpoint)", "Val MSE", lower_better=True)
    _bar_chart(axes[1], best_mses, labels, colours,
               "Val MSE (computed on val set)", "MSE", lower_better=True)
    _bar_chart(axes[2], best_maes, labels, colours,
               "Val MAE (computed on val set)", "MAE", lower_better=True)
    plt.suptitle("Reconstruction Quality — CelebA-HQ", fontsize=13, fontweight="bold")
    _add_legend(fig_bar, runs)
    fig_bar.tight_layout(rect=[0, 0.08, 1, 1])
    fig_bar.savefig(save_dir / "p1_reconstruction_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig_bar)
    print("  Reconstruction bar charts saved.")

    # ---- sample reconstructions (recon only, one row per model) ----
    left_margin_inch = 1.8
    img_w = 2.2
    img_h = 2.2
    fig_width  = left_margin_inch + n_recon_images * img_w
    fig_height = n_runs * img_h
    fig_recon = plt.figure(figsize=(fig_width, fig_height))

    left_frac = left_margin_inch / fig_width
    gs = fig_recon.add_gridspec(
        n_runs, n_recon_images,
        left=left_frac + 0.01, right=0.99,
        top=0.95, bottom=0.01,
        hspace=0.06, wspace=0.04,
    )
    axes_r = gs.subplots()
    if n_runs == 1:
        axes_r = axes_r.reshape(1, n_recon_images)

    for ri, run in enumerate(tqdm(runs, desc="  Collecting reconstructions")):
        _, recons = collect_reconstructions(run, device, val_loader, n_recon_images)
        for ci in range(n_recon_images):
            axes_r[ri, ci].imshow(recons[ci])
            axes_r[ri, ci].axis("off")
            for spine in axes_r[ri, ci].spines.values():
                spine.set_edgecolor(run["colour"])
                spine.set_linewidth(2.0)

        # Run directory name in the left margin
        pos = axes_r[ri, 0].get_position()
        y_mid   = (pos.y0 + pos.y1) / 2
        x_label = left_frac * 0.5
        fig_recon.text(
            x_label, y_mid,
            run["name"],
            ha="center", va="center",
            fontsize=6.5, fontfamily="monospace",
            color=run["colour"], fontweight="bold",
            wrap=False,
        )

    fig_recon.suptitle("Sample Reconstructions — CelebA-HQ",
                 fontsize=12, fontweight="bold", y=0.99)
    fig_recon.savefig(save_dir / "p1_sample_reconstructions.png",
                      dpi=120, bbox_inches="tight")
    plt.close(fig_recon)
    print("  Sample reconstructions saved.")


def make_page2_sparsity(
    runs: List[Dict],
    all_metrics: List[Dict],
    save_dir: Path,
) -> None:
    """Page 2: Sparsity analysis comparison."""

    labels  = [r["short"] for r in runs]
    colours = [r["colour"] for r in runs]

    mean_sparsity = [m["mean_sparsity"] for m in all_metrics]
    mean_l0       = [m["mean_l0"]       for m in all_metrics]
    dead_fracs    = [m.get("dead_frac")      for m in all_metrics]
    sel_fracs     = [m.get("selective_frac") for m in all_metrics]
    mod_fracs     = [m.get("moderate_frac")  for m in all_metrics]
    dns_fracs     = [m.get("dense_frac")     for m in all_metrics]

    # ---- bar charts ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    _bar_chart(axes[0, 0], mean_sparsity, labels, colours,
               "Mean Latent Sparsity\n(fraction near-zero features)",
               "Fraction zeros", lower_better=False)
    _bar_chart(axes[0, 1], mean_l0, labels, colours,
               "Mean L0 Norm\n(active feature count per sample)",
               "Active features", lower_better=True)

    # Stacked bar: dead / selective / moderate / dense
    valid_mask = [all(v is not None for v in (d, s, mo, dn))
                  for d, s, mo, dn in zip(dead_fracs, sel_fracs, mod_fracs, dns_fracs)]
    valid_runs    = [r for r, v in zip(runs, valid_mask) if v]
    valid_labels  = [r["short"] for r in valid_runs]
    valid_colours = [r["colour"] for r in valid_runs]
    if valid_runs:
        x = np.arange(len(valid_runs))
        d_vals = [dead_fracs[i]  * 100 for i, v in enumerate(valid_mask) if v]
        s_vals = [sel_fracs[i]   * 100 for i, v in enumerate(valid_mask) if v]
        m_vals = [mod_fracs[i]   * 100 for i, v in enumerate(valid_mask) if v]
        dn_vals= [dns_fracs[i]   * 100 for i, v in enumerate(valid_mask) if v]
        ax = axes[1, 0]
        b0 = ax.bar(x, d_vals,  label="Dead (<1%)",          color="#d62728",  edgecolor="black", linewidth=0.4)
        b1 = ax.bar(x, s_vals,  bottom=d_vals,               label="Selective (1-20%)",  color="#2ca02c",  edgecolor="black", linewidth=0.4)
        b2 = ax.bar(x, m_vals,  bottom=[a+b for a,b in zip(d_vals, s_vals)],
                    label="Moderate (20-50%)", color="#ff7f0e", edgecolor="black", linewidth=0.4)
        b3 = ax.bar(x, dn_vals, bottom=[a+b+c for a,b,c in zip(d_vals, s_vals, m_vals)],
                    label="Dense (>50%)",      color="#1f77b4", edgecolor="black", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(valid_labels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("% of Features"); ax.set_title("Feature Activity Categories", fontsize=9)
        ax.legend(fontsize=7, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, "Partonomy data\nnot available",
                        ha="center", va="center", transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Feature Activity Categories")

    # Lifetime sparsity histograms
    ax_h = axes[1, 1]
    for r, m in zip(runs, all_metrics):
        lts = m.get("lifetime_sparsity")
        if lts is not None:
            # lts[d] = fraction of samples where dim d is near-zero
            # "active_fraction" = 1 - near_zero_fraction
            active_frac = 1 - lts if isinstance(lts, np.ndarray) else None
            if active_frac is not None:
                ax_h.hist(active_frac, bins=60, alpha=0.5,
                          color=r["colour"], label=r["short"], density=True)
    ax_h.set_title("Latent Feature Activation Rate\n(histogram across all dims)", fontsize=9)
    ax_h.set_xlabel("Fraction of samples where feature is active")
    ax_h.set_ylabel("Density")
    ax_h.legend(fontsize=6, ncol=1); ax_h.grid(alpha=0.3)

    plt.suptitle("Sparsity Analysis — CelebA-HQ", fontsize=13, fontweight="bold")
    _add_legend(fig, runs)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(save_dir / "p2_sparsity_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Sparsity analysis saved.")


def make_page3_feature_quality(
    runs: List[Dict],
    all_metrics: List[Dict],
    save_dir: Path,
) -> None:
    """Page 3: Feature quality metrics."""

    labels  = [r["short"] for r in runs]
    colours = [r["colour"] for r in runs]

    jaccard    = [m.get("jaccard")        for m in all_metrics]
    sel_idx    = [m.get("selectivity_idx")for m in all_metrics]
    kurtosis   = [m.get("mean_kurtosis")  for m in all_metrics]
    p50        = [m.get("p50_units")      for m in all_metrics]
    p90        = [m.get("p90_units")      for m in all_metrics]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    _bar_chart(axes[0, 0], jaccard, labels, colours,
               "Mean Pairwise Activation Overlap\n(Jaccard — lower = more diverse/sparse)",
               "Mean Jaccard", lower_better=True)

    _bar_chart(axes[0, 1], sel_idx, labels, colours,
               "Feature Selectivity Index\n(higher = more monosemantic)",
               "Selectivity", lower_better=False)

    _bar_chart(axes[0, 2], kurtosis, labels, colours,
               "Mean Feature Kurtosis\n(higher = more peaked/sparse distributions)",
               "Kurtosis", lower_better=False)

    _bar_chart(axes[1, 0], p50, labels, colours,
               "Units for 50% Ablation Importance\n(lower = more concentrated)",
               "# Units", lower_better=True)

    _bar_chart(axes[1, 1], p90, labels, colours,
               "Units for 90% Ablation Importance\n(lower = more concentrated)",
               "# Units", lower_better=True)

    # Combined reconstruction vs sparsity scatter
    ax_sc = axes[1, 2]
    for r, m in zip(runs, all_metrics):
        mse = m.get("mean_mse"); sp = m.get("mean_sparsity")
        if mse is not None and sp is not None:
            if r["type"] == "taxon":
                marker = "o"
            elif r["type"] == "sae":
                marker = "s"
            else:
                marker = "^"
            ax_sc.scatter(sp, mse, color=r["colour"], s=80,
                          marker=marker, edgecolors="black", linewidths=0.8,
                          label=r["short"], zorder=3)
            ax_sc.annotate(r["short"], (sp, mse), textcoords="offset points",
                           xytext=(4, 4), fontsize=5.5)
    ax_sc.set_xlabel("Mean Latent Sparsity (fraction zero)")
    ax_sc.set_ylabel("Val MSE (lower is better)")
    ax_sc.set_title("Sparsity vs Reconstruction Trade-off\n○=Taxon  □=SAE  △=Baseline", fontsize=9)
    ax_sc.grid(alpha=0.3)

    plt.suptitle("Feature Quality & Sparsity–Reconstruction Trade-off — CelebA-HQ",
                 fontsize=12, fontweight="bold")
    _add_legend(fig, runs)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save_dir / "p3_feature_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Feature quality saved.")


def make_summary_csv(
    runs: List[Dict],
    all_metrics: List[Dict],
    save_dir: Path,
) -> None:
    """Dump all metrics to a CSV for easy inspection."""
    import csv
    fields = [
        "run", "type", "epoch",
        "ckpt_val_recon",
        "mean_mse", "mean_mae",
        "mean_sparsity", "mean_l0", "mean_l1",
        "dead_frac", "selective_frac", "moderate_frac", "dense_frac",
        "jaccard", "selectivity_idx", "mean_kurtosis",
        "p50_units", "p90_units",
    ]
    rows = []
    for r, m in zip(runs, all_metrics):
        ckpt = torch.load(r["best_ckpt"], map_location="cpu", weights_only=False)
        row = {
            "run":  r["name"],
            "type": r["type"],
            "epoch": ckpt.get("epoch", "?"),
            "ckpt_val_recon": ckpt.get("val_stats", {}).get("recon", ""),
        }
        row.update({k: m.get(k, "") for k in fields if k not in row})
        rows.append(row)

    csv_path = save_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Summary CSV saved to {csv_path}")


# ─── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparative analysis of all CelebA-HQ Taxon-AE and SAE runs"
    )
    parser.add_argument("--outputs-dir",  type=str, default="./outputs")
    parser.add_argument("--save-dir",     type=str, default="./outputs/comparison_celeba_hq")
    parser.add_argument("--data-root",    type=str, default="./data/celeba_hq",
                        help="CelebA-HQ data root (needed for live metric computation).")
    parser.add_argument("--image-size",   type=int, default=256)
    parser.add_argument("--batch-size",   type=int, default=16)
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--val-split",    type=float, default=0.05)
    parser.add_argument("--n-latent-batches", type=int, default=20,
                        help="Batches for live latent metric computation.")
    parser.add_argument("--n-recon-batches",  type=int, default=10)
    parser.add_argument("--n-recon-images",   type=int, default=6,
                        help="Number of images shown in the reconstruction panel.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    outputs_dir = Path(args.outputs_dir)
    save_dir    = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── discover runs ──────────────────────────────────────────────────────
    runs = discover_runs(outputs_dir)
    if not runs:
        print("No CelebA-HQ runs with a best.pt found under", outputs_dir)
        return

    print(f"Found {len(runs)} CelebA-HQ run(s):")
    for r in runs:
        print(f"  [{r['type']:5s}] {r['name']}")

    # ── build val loader (used for live metric computation + reconstructions) ──
    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    celeba = CelebAHQLoader(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        val_split=args.val_split,
        seed=42,
        pin_memory=(device.type == "cuda"),
        transform=tf,
    )
    train_loader, val_loader = celeba.get_loaders()
    if val_loader is None:
        val_loader = train_loader   # fallback: use train loader

    # ── gather metrics ─────────────────────────────────────────────────────
    all_metrics: List[Dict] = []
    print("\nGathering metrics...")
    for run in runs:
        m = load_precomputed_metrics(run)
        if m is not None:
            print(f"  [{run['type']:5s}] {run['short']}: loaded from pre-computed npz")
        else:
            print(f"  [{run['type']:5s}] {run['short']}: computing live...")
            m = compute_live_metrics(
                run, device, val_loader,
                n_batches_latent=args.n_latent_batches,
                n_batches_recon=args.n_recon_batches,
            )
        all_metrics.append(m)

    # ── produce figures ────────────────────────────────────────────────────
    print("\nGenerating figures...")
    make_page1_reconstruction(
        runs, all_metrics, save_dir, device, val_loader, args.n_recon_images
    )
    make_page2_sparsity(runs, all_metrics, save_dir)
    make_page3_feature_quality(runs, all_metrics, save_dir)
    make_summary_csv(runs, all_metrics, save_dir)

    # ── combined 3-page overview ───────────────────────────────────────────
    from PIL import Image as _PIL_Image
    pages = [
        save_dir / "p1_training_curves.png",
        save_dir / "p1_reconstruction_bars.png",
        save_dir / "p1_sample_reconstructions.png",
        save_dir / "p2_sparsity_analysis.png",
        save_dir / "p3_feature_quality.png",
    ]
    existing = [p for p in pages if p.exists()]
    if existing:
        imgs_pil = [_PIL_Image.open(p) for p in existing]
        widths  = [im.width  for im in imgs_pil]
        heights = [im.height for im in imgs_pil]
        max_w   = max(widths)
        total_h = sum(heights)
        combined = _PIL_Image.new("RGB", (max_w, total_h), (255, 255, 255))
        y_off = 0
        for im in imgs_pil:
            combined.paste(im, (0, y_off)); y_off += im.height
        combined.save(save_dir / "comparison_celeba_hq_overview.png")
        print(f"\nCombined overview saved to {save_dir}/comparison_celeba_hq_overview.png")

    print(f"\nAll outputs saved to {save_dir}")


if __name__ == "__main__":
    main()
