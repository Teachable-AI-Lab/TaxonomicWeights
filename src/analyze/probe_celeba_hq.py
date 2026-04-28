#!/usr/bin/env python3
"""Linear probing, KNN classification, and sparse interpretability analysis
for all CelebA-HQ runs in both ``outputs/celeba_hq/main/`` and
``outputs/celeba_hq/`` (root-level runs).

Metrics computed per model
--------------------------
Linear probing (sklearn LogisticRegression)
  • Per-attribute accuracy and ROC-AUC on each of the 40 CelebA binary
    attributes, using the global-average-pooled encoder latent as input.
  • Mean accuracy and mean AUC across all 40 attributes.

KNN classification (k=1,5,20)
  • Per-attribute AUC at each k-value.
  • Macro-average across attributes.

Sparse interpretability
  • L0 norm of the latent (fraction of non-zero spatial positions per channel).
  • Dead feature fraction (channels that never fire across the full val set).
  • Feature activation frequency histogram.
  • Mean Jaccard overlap between per-image binarised activation vectors.
  • Feature selectivity index: std(channel_mean_per_image) / mean activation.

Taxonomic vs non-taxonomic comparisons (sparsity-matched)
  • For each taxon model, the non-taxon model with the nearest L0 is found.
  • Scatter plots of performance and sparsity metrics vs L0 for both families.
  • Side-by-side bar charts of matched pairs (taxon vs nearest baseline).
  • Per-attribute AUC heatmap with taxon and matched-baseline rows interleaved.

Outputs
-------
``<save_dir>/probe_celeba_hq_results.csv``         — wide per-model metric table
``<save_dir>/probe_celeba_hq_linear.png``          — bar charts: mean linear AUC
``<save_dir>/probe_celeba_hq_knn.png``             — bar charts: mean KNN AUC (k=5)
``<save_dir>/probe_celeba_hq_sparsity.png``        — bar charts: L0, dead, selectivity
``<save_dir>/probe_celeba_hq_attr_detail.png``     — per-attribute heatmap across models
``<save_dir>/probe_celeba_hq_taxon_scatter.png``   — perf vs L0 scatter, taxon vs non-taxon
``<save_dir>/probe_celeba_hq_taxon_matched.png``   — matched-pair bar charts
``<save_dir>/probe_celeba_hq_taxon_attr_compare.png`` — per-attr heatmap, matched pairs
``<save_dir>/tsne/probe_celeba_hq_tsne_<name>.png``   — per-model t-SNE coloured by all attrs
``<save_dir>/tsne/probe_celeba_hq_tsne_compare.png``  — cross-model t-SNE for headline attrs
``<save_dir>/<run_name>/latents.npz``              — cached latents + attr labels
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── lazy sklearn imports (heavy) ────────────────────────────────────────────
def _import_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.preprocessing import StandardScaler
    return LogisticRegression, KNeighborsClassifier, roc_auc_score, accuracy_score, StandardScaler

# ── CelebA-HQ attribute names ────────────────────────────────────────────────
CELEBA_ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace",
    "Wearing_Necktie", "Young",
]

# ── re-use model loading from compare script ─────────────────────────────────
sys.path.insert(0, str(ROOT / "src" / "compare"))
from compare_celeba_hq import (   # type: ignore[import]
    load_model,
    discover_runs,
    _model_type,
    _short_name,
    _detect_version,
)


# ─── data loading ─────────────────────────────────────────────────────────────

def _load_celeba_attr_dataset(data_root: str, image_size: int = 256,
                               val_split: float = 0.05, seed: int = 42,
                               max_samples: int = 5000):
    """Return a DataLoader that yields (img_tensor, attr_tensor[40]) pairs.

    Tries celeba_hq first; falls back to the standard CelebA dataset in
    ``data_root/celeba_fallback``.  The fallback path is guaranteed to have
    attribute labels.

    ``max_samples`` caps the number of images used to keep inference fast.
    """
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader, Subset, Dataset
    import random

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Try CelebA fallback which always has attributes
    celeba_fallback = Path(data_root) / "celeba_fallback"
    if celeba_fallback.exists():
        print(f"  Loading CelebA attributes from: {celeba_fallback}")
        try:
            ds = datasets.CelebA(
                root=str(celeba_fallback),
                split="test",         # held-out split for unbiased probing
                target_type="attr",
                transform=transform,
                download=False,
            )
        except Exception:
            ds = datasets.CelebA(
                root=str(celeba_fallback),
                split="train",
                target_type="attr",
                transform=transform,
                download=False,
            )
    else:
        # Try downloading CelebA
        print(f"  CelebA fallback not found at {celeba_fallback}, downloading...")
        celeba_fallback.mkdir(parents=True, exist_ok=True)
        ds = datasets.CelebA(
            root=str(celeba_fallback),
            split="test",
            target_type="attr",
            transform=transform,
            download=True,
        )

    # Cap size for speed
    n = min(max_samples, len(ds))
    rng = np.random.RandomState(seed)
    idxs = rng.choice(len(ds), n, replace=False)
    sub = Subset(ds, idxs.tolist())

    loader = DataLoader(sub, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    print(f"  Attribute dataset: {n} images, 40 binary attributes")
    return loader


# ─── latent extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_latents(
    model: torch.nn.Module,
    model_type: str,
    loader,
    device: torch.device,
    max_batches: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract global-avg-pooled encoder latent vectors for all batches.

    Returns
    -------
    latents  : (N, D) float32 numpy array
    attrs    : (N, 40) int8 numpy array  (+1 / -1 / 0 → binarised as ≥0)
    """
    model.eval()

    # Hook the encoder output (first tensor returned by model.encoder forward)
    captured: Dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            captured["z"] = out.detach()
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            first = out[0]
            if isinstance(first, torch.Tensor):
                captured["z"] = first.detach()
            elif isinstance(first, (list, tuple)) and len(first) > 0:
                # intermediate_topk_sae: encoder returns (List[Tensor], info)
                # Use the last (deepest) stage latent as the representation
                last = first[-1]
                if isinstance(last, torch.Tensor):
                    captured["z"] = last.detach()

    # Attach hook to model.encoder
    hook_handle = model.encoder.register_forward_hook(_hook)

    all_latents: List[np.ndarray] = []
    all_attrs:   List[np.ndarray] = []

    try:
        for batch_idx, (imgs, attrs) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)

            # Forward pass triggers the hook
            try:
                model(imgs)
            except Exception:
                pass  # hook already fired before any error

            if "z" not in captured:
                continue

            z = captured.pop("z")  # (B, C, H, W) or (B, C)
            if z.dim() == 4:
                z = F.adaptive_avg_pool2d(z, 1).flatten(1)  # → (B, C)
            elif z.dim() == 3:
                z = z.mean(dim=1)  # sequence → avg

            all_latents.append(z.cpu().float().numpy())
            # attrs: CelebA returns {0,1} or {-1,1}; convert to {0,1}
            a = attrs.cpu().numpy() if isinstance(attrs, torch.Tensor) else attrs
            a = (a >= 1).astype(np.int8)   # -1→0, 0→0, 1→1
            all_attrs.append(a)

    finally:
        hook_handle.remove()

    if not all_latents:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 40), dtype=np.int8)

    return np.concatenate(all_latents, axis=0), np.concatenate(all_attrs, axis=0)


# ─── linear probing ───────────────────────────────────────────────────────────

def run_linear_probe(
    latents: np.ndarray,
    attrs: np.ndarray,
    attr_names: List[str],
    n_train_frac: float = 0.8,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Train a LogisticRegression probe on each of the 40 CelebA attributes.

    Returns dict with keys:
      ``accuracy``  (40,) per-attribute accuracy on held-out set
      ``auc``       (40,) per-attribute ROC-AUC on held-out set
    """
    LogisticRegression, _, roc_auc_score, accuracy_score, StandardScaler = _import_sklearn()

    N = len(latents)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_train = int(N * n_train_frac)
    tr, te = idx[:n_train], idx[n_train:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(latents[tr])
    X_te = scaler.transform(latents[te])

    acc_list, auc_list = [], []
    for ai, name in enumerate(attr_names):
        y_tr, y_te = attrs[tr, ai], attrs[te, ai]
        if len(np.unique(y_tr)) < 2:
            acc_list.append(float("nan"))
            auc_list.append(float("nan"))
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1] if len(clf.classes_) == 2 else None
        acc_list.append(accuracy_score(y_te, y_pred))
        try:
            auc_list.append(roc_auc_score(y_te, y_prob) if y_prob is not None else float("nan"))
        except Exception:
            auc_list.append(float("nan"))

    return {
        "accuracy": np.array(acc_list, dtype=np.float32),
        "auc":      np.array(auc_list,  dtype=np.float32),
    }


# ─── KNN classification ───────────────────────────────────────────────────────

def run_knn(
    latents: np.ndarray,
    attrs: np.ndarray,
    attr_names: List[str],
    k_values: Tuple[int, ...] = (1, 5, 20),
    n_train_frac: float = 0.8,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """KNN probing per attribute at each k.

    Returns dict ``k{k}_auc`` → (40,) per-attribute AUC arrays.
    """
    _, KNeighborsClassifier, roc_auc_score, _, StandardScaler = _import_sklearn()

    N = len(latents)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_train = int(N * n_train_frac)
    tr, te = idx[:n_train], idx[n_train:]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(latents[tr])
    X_te = scaler.transform(latents[te])

    results: Dict[str, np.ndarray] = {}
    for k in k_values:
        auc_list = []
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=4, metric="cosine")
        # Fit once per k (all attrs at once for label matching)
        for ai, name in enumerate(attr_names):
            y_tr, y_te = attrs[tr, ai], attrs[te, ai]
            if len(np.unique(y_tr)) < 2:
                auc_list.append(float("nan"))
                continue
            knn.fit(X_tr, y_tr)
            y_prob = knn.predict_proba(X_te)
            # select prob for class=1 if available
            cls1_idx = list(knn.classes_).index(1) if 1 in knn.classes_ else -1
            if cls1_idx >= 0:
                proba1 = y_prob[:, cls1_idx]
            else:
                proba1 = y_prob[:, 0]
            try:
                auc_list.append(roc_auc_score(y_te, proba1))
            except Exception:
                auc_list.append(float("nan"))
        results[f"k{k}_auc"] = np.array(auc_list, dtype=np.float32)

    return results


# ─── sparse interpretability ──────────────────────────────────────────────────

@torch.no_grad()
def compute_sparsity_stats(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float = 1e-6,
    max_batches: int = 60,
) -> Dict[str, float]:
    """Compute L0 norm, dead feature fraction, and selectivity on the val set.

    Returns
    -------
    dict with float scalars:
      mean_l0        : mean fraction of active (> threshold) latent positions (flattened)
      dead_frac      : fraction of latent dimensions that never fire across all samples
      l0_abs         : mean count of active latent positions per image (channels × spatial)
      selectivity    : mean channel std-dev / mean activation (higher = more selective)
      mean_jaccard   : mean pairwise Jaccard on binarised activation vectors
                       (estimated from a random 512-sample subset)
    """
    model.eval()
    captured: Dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            captured["z"] = out.detach()
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            first = out[0]
            if isinstance(first, torch.Tensor):
                captured["z"] = first.detach()
            elif isinstance(first, (list, tuple)) and len(first) > 0:
                last = first[-1]
                if isinstance(last, torch.Tensor):
                    captured["z"] = last.detach()

    hook_handle = model.encoder.register_forward_hook(_hook)

    all_acts: List[np.ndarray] = []  # pooled activations (N, C)

    try:
        for batch_idx, (imgs, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            imgs = imgs.to(device, non_blocking=True)
            try:
                model(imgs)
            except Exception:
                pass

            if "z" not in captured:
                continue
            z = captured.pop("z")
            if z.dim() == 4:
                z = z.reshape(z.shape[0], -1)
            z = z.abs().cpu().float().numpy()
            all_acts.append(z)
    finally:
        hook_handle.remove()

    if not all_acts:
        return {k: float("nan") for k in
                ("mean_l0", "dead_frac", "l0_abs", "selectivity", "mean_jaccard")}

    acts = np.concatenate(all_acts, axis=0)   # (N, C)
    N, C = acts.shape

    binary = (acts > threshold).astype(np.float32)

    mean_l0 = binary.mean()                         # fraction active per entry
    l0_abs  = binary.sum(axis=1).mean()              # mean active channels per image
    dead    = (acts.max(axis=0) <= threshold)        # channel never fires
    dead_frac = dead.mean()

    # Selectivity: std of channel means / global mean
    ch_mean = acts.mean(axis=0)                     # (C,)
    global_mean = ch_mean.mean() + 1e-8
    selectivity = ch_mean.std() / global_mean

    # Mean Jaccard on a random 512-sample subset
    n_sub = min(512, N)
    rng = np.random.RandomState(0)
    sub_idx = rng.choice(N, n_sub, replace=False)
    B = binary[sub_idx]                              # (n_sub, C)
    inter = (B @ B.T)                                # (n_sub, n_sub)
    row_sum = B.sum(axis=1, keepdims=True)
    union = row_sum + row_sum.T - inter + 1e-8
    jac = inter / union
    # Off-diagonal mean
    mask = ~np.eye(n_sub, dtype=bool)
    mean_jaccard = float(jac[mask].mean())

    return {
        "mean_l0":      float(mean_l0),
        "dead_frac":    float(dead_frac),
        "l0_abs":       float(l0_abs),
        "selectivity":  float(selectivity),
        "mean_jaccard": float(mean_jaccard),
    }


def compute_monosemanticity(
    latents: np.ndarray,
    attrs: np.ndarray,
    top_k: int = 50,
    min_active_frac: float = 0.005,
) -> float:
    """Compute a monosemanticity score averaged over active features.

    For each feature *i* that fires on at least ``min_active_frac`` of samples:

    1. Collect the indices of the top-K images most strongly activating feature i.
    2. For each CelebA attribute j compute the *lift*:
       ``lift_ij = freq(attr_j | top-K) / (base_freq(attr_j) + eps)``
    3. The per-feature score is ``max_j(lift_ij)``.

    The model-level score is the mean over all qualifying features.
    A perfectly monosemantic feature that fires only for images with attribute j
    yields lift ≈ 1/base_freq_j; a random feature yields lift ≈ 1.

    Parameters
    ----------
    latents : np.ndarray, shape (N, C)
        Pooled encoder activations.
    attrs   : np.ndarray, shape (N, 40)
        Binary CelebA attribute labels (float32, 0/1).
    top_k   : int
        Number of top activating images to examine per feature.
    min_active_frac : float
        Skip a feature if its max activation across all samples is below this
        quantile of the activation distribution (proxy for "dead").

    Returns
    -------
    float  — mean max-lift monosemanticity score (NaN if no features qualify).
    """
    N, C = latents.shape
    k = min(top_k, N)
    attrs_f = attrs.astype(np.float32)              # (N, 40)
    base_freq = attrs_f.mean(axis=0) + 1e-6         # (40,)  global attribute rates

    # Consider only features that fire on at least min_active_frac of images
    ch_max = latents.max(axis=0)                    # (C,)
    thresh = np.quantile(latents[latents > 0], min_active_frac) if (latents > 0).any() else 0.0
    active_mask = ch_max > thresh                   # (C,) bool

    scores = []
    for i in range(C):
        if not active_mask[i]:
            continue
        # Indices of top-K activating images for feature i
        top_idx = np.argpartition(latents[:, i], -k)[-k:]
        top_attrs = attrs_f[top_idx]                # (k, 40)
        freq_top = top_attrs.mean(axis=0)           # (40,)
        lift = freq_top / base_freq                 # (40,)
        scores.append(float(lift.max()))

    return float(np.mean(scores)) if scores else float("nan")


# ─── plotting ─────────────────────────────────────────────────────────────────

_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#000080", "#1f3a93", "#3b5998", "#5b7bbf", "#082567",
    "#003f5c", "#2f4b7c", "#1a3a5c", "#bc6c25", "#dda15e",
]


def _bar(ax, vals, labels, colours, title, ylabel, lower_better=False):
    valid = [(v, l, c) for v, l, c in zip(vals, labels, colours)
             if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    vs, ls, cs = zip(*valid)
    bars = ax.bar(range(len(vs)), vs, color=cs, edgecolor="black", linewidth=0.4)
    best = int(np.argmin(vs) if lower_better else np.argmax(vs))
    bars[best].set_edgecolor("red")
    bars[best].set_linewidth(2.0)
    ax.set_xticks(range(len(ls)))
    ax.set_xticklabels(ls, rotation=40, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_linear_results(all_results, save_dir: Path):
    names   = [r["short"] for r in all_results]
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(all_results))]

    mean_aucs = [np.nanmean(r["linear"]["auc"]) if r.get("linear") else float("nan")
                 for r in all_results]
    mean_accs = [np.nanmean(r["linear"]["accuracy"]) if r.get("linear") else float("nan")
                 for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(names) * 0.9), 5))
    _bar(axes[0], mean_aucs, names, colours, "Mean Linear Probe AUC (40 attrs)", "ROC-AUC")
    _bar(axes[1], mean_accs, names, colours, "Mean Linear Probe Accuracy", "Accuracy")
    plt.suptitle("Linear Probing — CelebA-HQ", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "probe_celeba_hq_linear.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: probe_celeba_hq_linear.png")


def plot_knn_results(all_results, save_dir: Path):
    names   = [r["short"] for r in all_results]
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(all_results))]

    for k_key in ("k1_auc", "k5_auc", "k20_auc"):
        k = k_key.split("_")[0]
        vals = [np.nanmean(r["knn"][k_key]) if r.get("knn") and k_key in r["knn"]
                else float("nan") for r in all_results]
        fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 4))
        _bar(ax, vals, names, colours, f"Mean KNN AUC ({k}) — 40 CelebA attrs", "ROC-AUC")
        plt.tight_layout()
        fig.savefig(save_dir / f"probe_celeba_hq_knn_{k}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("  Saved: probe_celeba_hq_knn_{k1,k5,k20}.png")


def plot_sparsity_results(all_results, save_dir: Path):
    names   = [r["short"] for r in all_results]
    colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(all_results))]
    sp      = [r.get("sparsity", {}) for r in all_results]

    fig, axes = plt.subplots(1, 5, figsize=(max(20, len(names) * 1.2), 5))
    _bar(axes[0], [s.get("mean_l0")      for s in sp], names, colours,
         "Mean L0 (frac active)",      "Fraction",    lower_better=False)
    _bar(axes[1], [s.get("dead_frac")    for s in sp], names, colours,
         "Dead Feature Fraction",      "Fraction",    lower_better=True)
    _bar(axes[2], [s.get("l0_abs")       for s in sp], names, colours,
         "Mean Active Channels",       "Count",       lower_better=False)
    _bar(axes[3], [s.get("selectivity")  for s in sp], names, colours,
         "Feature Selectivity Index",  "Selectivity", lower_better=False)
    _bar(axes[4], [s.get("mean_jaccard") for s in sp], names, colours,
         "Mean Jaccard (binarised act)", "Jaccard",   lower_better=True)
    plt.suptitle("Sparse Interpretability — CelebA-HQ", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "probe_celeba_hq_sparsity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: probe_celeba_hq_sparsity.png")


def plot_attr_heatmap(all_results, save_dir: Path):
    """Heatmap: rows = models, columns = 40 CelebA attributes, value = AUC."""
    rows = [r for r in all_results if r.get("linear")]
    if not rows:
        return
    names = [r["short"] for r in rows]
    matrix = np.stack([r["linear"]["auc"] for r in rows], axis=0)  # (M, 40)
    fig, ax = plt.subplots(figsize=(max(16, len(CELEBA_ATTR_NAMES) * 0.45),
                                    max(4, len(rows) * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(CELEBA_ATTR_NAMES)))
    ax.set_xticklabels(CELEBA_ATTR_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Per-Attribute Linear Probe AUC (rows=models, cols=attrs)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    plt.tight_layout()
    fig.savefig(save_dir / "probe_celeba_hq_attr_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: probe_celeba_hq_attr_detail.png")


# ─── taxon vs. non-taxon sparsity-matched comparisons ─────────────────────────

# Model types that use a taxonomic hierarchy in their latent space.
_TAXON_TYPES = {
    "taxon", "multi_taxon",
    "topk_taxon", "topk_multi_taxon",
    "bias_taxon", "bias_multi_taxon",
    "bottleneck_taxon", "bottleneck_multi_taxon",
    "bottleneck_topk_taxon", "bottleneck_topk_multi_taxon",
}

_TAXON_COLOR  = "#1f77b4"   # blue for taxonomic models
_BASE_COLOR   = "#d62728"   # red for non-taxonomic baselines


def _is_taxon(r: dict) -> bool:
    return r.get("type", "") in _TAXON_TYPES


def _sparsity_l0(r: dict) -> Optional[float]:
    """Return the mean active channel count (L0 abs) for a result, or None."""
    sp = r.get("sparsity", {})
    v = sp.get("l0_abs")
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    return float(v)


def find_sparsity_matched_pairs(
    all_results: List[dict],
    max_l0_ratio: float = 3.0,
) -> List[Tuple[dict, dict]]:
    """For each taxon model find the non-taxon model with the nearest L0 (log-scale).

    Pairs whose L0 ratio exceeds ``max_l0_ratio`` are excluded as too dissimilar.
    Returns a list of *(taxon_result, baseline_result)* tuples, one per taxon model.
    The same baseline may appear in multiple pairs.
    """
    taxon_rs    = [r for r in all_results if _is_taxon(r) and _sparsity_l0(r) is not None]
    baseline_rs = [r for r in all_results if not _is_taxon(r) and _sparsity_l0(r) is not None]

    if not taxon_rs or not baseline_rs:
        return []

    pairs: List[Tuple[dict, dict]] = []
    for tr in taxon_rs:
        l0_t = _sparsity_l0(tr)
        best, best_dist = None, float("inf")
        for br in baseline_rs:
            l0_b = _sparsity_l0(br)
            dist = abs(np.log(l0_t + 1) - np.log(l0_b + 1))
            if dist < best_dist:
                best_dist, best = dist, br
        if best is not None:
            l0_b = _sparsity_l0(best)
            ratio = max(l0_t, l0_b) / (min(l0_t, l0_b) + 1e-8)
            if ratio <= max_l0_ratio:
                pairs.append((tr, best))

    return pairs


def plot_taxon_scatter(all_results: List[dict], save_dir: Path):
    """Scatter: L0 (mean active channels) vs performance/sparsity metrics.

    Taxonomic models are shown as filled circles (blue); non-taxonomic models
    as crosses (red).  A dashed reference line is drawn at the median L0 of
    taxonomic models to aid visual comparison.
    """
    taxon_r    = [r for r in all_results if _is_taxon(r)]
    baseline_r = [r for r in all_results if not _is_taxon(r)]

    def _collect(results, metric_fn):
        xs, ys, labels = [], [], []
        for r in results:
            l0 = _sparsity_l0(r)
            y  = metric_fn(r)
            if l0 is not None and y is not None and not np.isnan(float(y)):
                xs.append(l0)
                ys.append(float(y))
                labels.append(r["short"])
        return xs, ys, labels

    metrics = [
        ("Linear Probe Mean AUC",
         lambda r: float(np.nanmean(r["linear"]["auc"])) if r.get("linear") else None),
        ("KNN k=5 Mean AUC",
         lambda r: float(np.nanmean(r["knn"]["k5_auc"])) if r.get("knn") and "k5_auc" in r["knn"] else None),
        ("Feature Selectivity",
         lambda r: r.get("sparsity", {}).get("selectivity")),
        ("Dead Feature Fraction",
         lambda r: r.get("sparsity", {}).get("dead_frac")),
    ]

    # Drop metrics where neither family has data
    active_metrics = []
    for title, fn in metrics:
        tx, ty, _ = _collect(taxon_r, fn)
        bx, by, _ = _collect(baseline_r, fn)
        if tx or bx:
            active_metrics.append((title, fn))

    if not active_metrics:
        print("  No data for taxon scatter plot, skipping.")
        return

    n = len(active_metrics)

    def _draw_panel(ax, tx, ty, tlabels, bx, by, blabels, title, x_cap=None):
        def _cx(x):
            return min(x, x_cap) if x_cap is not None else x
        ax.scatter([_cx(x) for x in bx], by, marker="X", s=90,
                   color=_BASE_COLOR, alpha=0.8, label="Non-taxon",
                   zorder=3, edgecolors="black", linewidths=0.4)
        ax.scatter([_cx(x) for x in tx], ty, marker="o", s=90,
                   color=_TAXON_COLOR, alpha=0.85, label="Taxon",
                   zorder=4, edgecolors="black", linewidths=0.4)
        for x, y, lbl in zip(tx, ty, tlabels):
            prefix = "\u2192" if (x_cap is not None and x > x_cap) else ""
            ax.annotate(f"{prefix}{lbl}", (_cx(x), y), textcoords="offset points",
                        xytext=(4, 3), fontsize=6, color=_TAXON_COLOR, alpha=0.85)
        for x, y, lbl in zip(bx, by, blabels):
            prefix = "\u2192" if (x_cap is not None and x > x_cap) else ""
            ax.annotate(f"{prefix}{lbl}", (_cx(x), y), textcoords="offset points",
                        xytext=(4, 3), fontsize=6, color=_BASE_COLOR, alpha=0.7)
        if tx:
            med = float(np.median(tx))
            ax.axvline(_cx(med), color=_TAXON_COLOR, linewidth=0.8, linestyle="--",
                       alpha=0.45, label=f"Taxon median L0={med:.0f}")
        if x_cap is not None:
            ax.set_xlim(-0.02 * x_cap, x_cap * 1.05)
            ax.axvline(x_cap, color="gray", lw=0.7, ls=":", alpha=0.5)
        ax.set_xlabel("Mean Active Features (L0 abs)", fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    # Shared x_cap: 90th percentile of all L0 values
    all_xs_flat = []
    for _, fn in active_metrics:
        tx, _, _ = _collect(taxon_r, fn)
        bx, _, _ = _collect(baseline_r, fn)
        all_xs_flat.extend(tx + bx)
    x_cap = float(np.percentile(all_xs_flat, 90)) if all_xs_flat else None

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (title, fn) in enumerate(active_metrics):
        tx, ty, tlabels = _collect(taxon_r, fn)
        bx, by, blabels = _collect(baseline_r, fn)
        _draw_panel(axes[0, col], tx, ty, tlabels, bx, by, blabels, title)
        axes[0, col].set_title(f"{title} (full)", fontsize=10)
        _draw_panel(axes[1, col], tx, ty, tlabels, bx, by, blabels, title, x_cap=x_cap)
        axes[1, col].set_title(f"{title} (zoomed, L0 \u2264 {x_cap:.0f})", fontsize=10)

    fig.suptitle("Taxonomic vs Non-Taxonomic: Performance vs Sparsity (CelebA-HQ)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "probe_celeba_hq_taxon_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_matched_pairs(all_results: List[dict], save_dir: Path):
    """Side-by-side bar chart: each taxon model vs its nearest-L0 non-taxon model.

    Four metric panels are stacked vertically: linear AUC, KNN k=5 AUC,
    feature selectivity, and dead feature fraction.  An L0 annotation is
    shown below each pair to confirm sparsity similarity.
    """
    pairs = find_sparsity_matched_pairs(all_results)
    if not pairs:
        print("  No sparsity-matched pairs found — skipping matched-pair plot.")
        return

    metric_defs = [
        ("Linear AUC",
         lambda r: float(np.nanmean(r["linear"]["auc"])) if r.get("linear") else None),
        ("KNN k=5 AUC",
         lambda r: float(np.nanmean(r["knn"]["k5_auc"])) if r.get("knn") and "k5_auc" in r["knn"] else None),
        ("Feature Selectivity",
         lambda r: r.get("sparsity", {}).get("selectivity")),
        ("Dead Feature Fraction",
         lambda r: r.get("sparsity", {}).get("dead_frac")),
    ]

    # Keep only panels with at least one non-None value in the pair set
    active_defs = []
    for label, fn in metric_defs:
        tv = [fn(t) for t, _ in pairs]
        bv = [fn(b) for _, b in pairs]
        if any(v is not None for v in tv + bv):
            active_defs.append((label, fn))

    if not active_defs:
        print("  No metrics available for matched-pair plot, skipping.")
        return

    n_pairs   = len(pairs)
    n_metrics = len(active_defs)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(max(12, n_pairs * 1.8), 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    x_pos = np.arange(n_pairs)
    width = 0.35

    for ax, (label, fn) in zip(axes, active_defs):
        tv = [fn(t) for t, _ in pairs]
        bv = [fn(b) for _, b in pairs]
        pair_labels = [f"{t['short']}\nvs\n{b['short']}" for t, b in pairs]
        l0_annots   = [
            f"L0 {_sparsity_l0(t):.0f} / {_sparsity_l0(b):.0f}"
            for t, b in pairs
        ]

        def _safe(v):
            return float(v) if v is not None and not np.isnan(float(v)) else 0.0

        tv_safe = [_safe(v) for v in tv]
        bv_safe = [_safe(v) for v in bv]

        bars_t = ax.bar(x_pos - width / 2, tv_safe, width,
                        color=_TAXON_COLOR, label="Taxon",
                        alpha=0.85, edgecolor="black", linewidth=0.5)
        bars_b = ax.bar(x_pos + width / 2, bv_safe, width,
                        color=_BASE_COLOR, label="Non-taxon (matched)",
                        alpha=0.85, edgecolor="black", linewidth=0.5)

        # Delta annotations on taxon bars
        for i, (tv_i, bv_i) in enumerate(zip(tv_safe, bv_safe)):
            delta = tv_i - bv_i
            if abs(delta) > 1e-6:
                sign = "+" if delta > 0 else ""
                ax.text(x_pos[i] - width / 2, tv_i + 0.002,
                        f"{sign}{delta:.3f}", ha="center", va="bottom",
                        fontsize=6, color=_TAXON_COLOR, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(pair_labels, fontsize=7, rotation=15, ha="right")
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"{label} — Taxon vs Sparsity-Matched Baseline", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        ymin = ax.get_ylim()[0]
        for i, annot in enumerate(l0_annots):
            ax.annotate(annot, xy=(x_pos[i], ymin),
                        xytext=(0, -24), textcoords="offset points",
                        ha="center", fontsize=6, color="gray")

    fig.suptitle("Taxon vs Sparsity-Matched Non-Taxon Baselines (CelebA-HQ)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = save_dir / "probe_celeba_hq_taxon_matched.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


def plot_taxon_attr_compare(all_results: List[dict], save_dir: Path):
    """Per-attribute AUC heatmap interleaving each taxon model with its matched baseline.

    Rows alternate: [Taxon] model row then [Base] matched non-taxon row, one pair
    per taxon model.  Dashed horizontal lines separate pairs.
    """
    pairs = find_sparsity_matched_pairs(all_results)

    # Keep only pairs where the taxon model has linear results
    valid_pairs = [(t, b) for t, b in pairs if t.get("linear")]
    if not valid_pairs:
        print("  No matched pairs with linear results — skipping attr compare plot.")
        return

    row_models: List[dict] = []
    row_labels:  List[str]  = []
    row_colors:  List[str]  = []

    for t, b in valid_pairs:
        row_models.append(t)
        row_labels.append(f"[Taxon] {t['short']}")
        row_colors.append(_TAXON_COLOR)
        if b.get("linear"):
            row_models.append(b)
            row_labels.append(f"[Base]  {b['short']}")
            row_colors.append(_BASE_COLOR)

    if not row_models:
        return

    matrix = np.stack([r["linear"]["auc"] for r in row_models], axis=0)

    fig, ax = plt.subplots(
        figsize=(max(18, len(CELEBA_ATTR_NAMES) * 0.48),
                 max(4, len(row_models) * 0.55))
    )
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(CELEBA_ATTR_NAMES)))
    ax.set_xticklabels(CELEBA_ATTR_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    # Colour y-tick labels by family
    for ytick, color in zip(ax.get_yticklabels(), row_colors):
        ytick.set_color(color)

    # Dashed separator lines between pairs
    cursor = -0.5
    for t, b in valid_pairs:
        cursor += 1.0  # taxon row
        if b.get("linear"):
            cursor += 1.0  # baseline row
        ax.axhline(cursor, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_title(
        "Per-Attribute Linear Probe AUC — Taxon (blue) vs Sparsity-Matched Baseline (red)\n"
        "Pairs separated by dashed lines",
        fontsize=10,
    )
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    plt.tight_layout()
    out = save_dir / "probe_celeba_hq_taxon_attr_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─── save CSV ─────────────────────────────────────────────────────────────────

def save_csv(all_results, save_dir: Path):
    rows = []
    for r in all_results:
        row = {
            "name":       r["name"],
            "short":      r["short"],
            "type":       r["type"],
            "mean_lin_auc":  float(np.nanmean(r["linear"]["auc"])) if r.get("linear") else "",
            "mean_lin_acc":  float(np.nanmean(r["linear"]["accuracy"])) if r.get("linear") else "",
            "mean_knn1_auc": float(np.nanmean(r["knn"]["k1_auc"])) if r.get("knn") and "k1_auc" in r["knn"] else "",
            "mean_knn5_auc": float(np.nanmean(r["knn"]["k5_auc"])) if r.get("knn") and "k5_auc" in r["knn"] else "",
            "mean_knn20_auc": float(np.nanmean(r["knn"]["k20_auc"])) if r.get("knn") and "k20_auc" in r["knn"] else "",
            "mean_l0":       r.get("sparsity", {}).get("mean_l0",    ""),
            "dead_frac":     r.get("sparsity", {}).get("dead_frac",  ""),
            "l0_abs":        r.get("sparsity", {}).get("l0_abs",     ""),
            "selectivity":   r.get("sparsity", {}).get("selectivity",""),
            "mean_jaccard":  r.get("sparsity", {}).get("mean_jaccard",""),
            "monosemanticity": r.get("sparsity", {}).get("monosemanticity",""),
        }
        # Append per-attribute AUC columns
        if r.get("linear"):
            for i, aname in enumerate(CELEBA_ATTR_NAMES):
                row[f"lin_auc_{aname}"] = float(r["linear"]["auc"][i])
        rows.append(row)

    if not rows:
        return

    out_path = save_dir / "probe_celeba_hq_results.csv"
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {out_path}")


# ─── latent cache ────────────────────────────────────────────────────────────

def _cache_path(run_path: Path, cache_dir: Path) -> Path:
    return cache_dir / run_path.name / "latents.npz"


def _load_cached_latents(path: Path):
    if path.exists():
        d = np.load(path)
        return d["latents"], d["attrs"]
    return None, None


def _save_cached_latents(path: Path, latents: np.ndarray, attrs: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, latents=latents, attrs=attrs)


# ─── main ────────────────────────────────────────────────────────────────────

# ─── t-SNE visualisation ──────────────────────────────────────────────────────

# Model types that receive t-SNE plots.
_TSNE_TYPES = {
    "bottleneck_topk_taxon",
    "bottleneck_topk_multi_taxon",
    "topk_taxon",
    "topk_multi_taxon",
    "matryoshka_batch_topk_sae",
    "matryoshka_intermediate_topk_sae",
    "topk_sae",
    "intermediate_topk_sae",
}

# Curated headline attributes for the cross-model comparison figure.
_TSNE_HEADLINE_ATTRS = [
    "Male", "Smiling", "Young", "Bald", "Eyeglasses",
    "Blond_Hair", "Heavy_Makeup", "Wearing_Hat",
    "Mustache", "Pale_Skin", "Attractive", "Wavy_Hair",
]

# All attributes shown in the per-model grid (trimmed label for readability).
_TSNE_ALL_ATTRS = CELEBA_ATTR_NAMES  # all 40


def plot_monosemanticity_scatter(all_results: List[dict], save_dir: Path):
    """Scatter: mean active channels (L0) vs monosemanticity score.

    Each point is one model, coloured by model type.  Models without a
    monosemanticity score (e.g. dead/no latents) are omitted.
    A score of 1.0 means the top-K images for every feature match the global
    attribute base-rate (random).  Higher scores indicate features that fire
    preferentially for images carrying one specific attribute.
    """
    # Group results by type for per-type colour coding
    type_order = sorted({r["type"] for r in all_results if r.get("type")})
    colour_map = {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(type_order)}

    xs, ys, colours, labels = [], [], [], []
    for r in all_results:
        l0 = _sparsity_l0(r)
        mono = r.get("sparsity", {}).get("monosemanticity")
        if l0 is None or mono is None or np.isnan(mono):
            continue
        xs.append(l0)
        ys.append(mono)
        colours.append(colour_map.get(r.get("type", ""), "#888888"))
        labels.append(r["short"])

    if not xs:
        print("  No monosemanticity data, skipping scatter.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(xs, ys, c=colours, s=60, edgecolors="none", alpha=0.85, zorder=3)

    # Annotate each point
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), fontsize=5.5, ha="left", va="bottom",
                    xytext=(3, 2), textcoords="offset points", color="#333333")

    # Reference line at score=1 (random / no enrichment)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.6, label="random (score=1)")

    # Legend for model types
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colour_map[t],
                   markersize=7, label=t)
        for t in type_order if t in colour_map
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right",
              title="Model type", title_fontsize=7, framealpha=0.8)

    ax.set_xlabel("Mean Active Channels (L0 abs)", fontsize=10)
    ax.set_ylabel("Mean Monosemanticity Score\n(max attribute lift, top-50 images)", fontsize=10)
    ax.set_title("Monosemanticity vs. Sparsity — CelebA-HQ", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, lw=0.5)
    plt.tight_layout()

    out = save_dir / "probe_celeba_hq_monosemanticity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─── taxon hierarchy exploration ──────────────────────────────────────────────

_TAXON_HIERARCHY_TYPES = {"bottleneck_topk_taxon", "bottleneck_taxon"}
_HIER_PROB_THRESHOLD   = 0.01   # "active" node: dataset-mean path-prob > 1%


@torch.no_grad()
def compute_taxon_hierarchy_stats(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int = 60,
    prob_threshold: float = _HIER_PROB_THRESHOLD,
) -> Dict[str, object]:
    """Analyse how the binary-tree hierarchy is explored by a bottleneck_topk_taxon model.

    Runs ``model.encoder(x, return_details=True)`` to obtain per-depth log-path
    probabilities (from the taxon stage, *before* TopK masking), then computes:

    ``node_util[d]``       — fraction of the 2^(d+1) nodes at depth d whose
                             dataset-mean path probability exceeds *prob_threshold*.
    ``mean_entropy[d]``    — mean per-spatial-position path entropy at depth d,
                             normalised by log(2^(d+1)) (0 = fully peaked, 1 = uniform).
    ``dead_nodes[d]``      — fraction of nodes that never exceed *prob_threshold*
                             in any single (image, spatial-position) pair.
    ``leaf_utilization``   — fraction of the 2^L leaf nodes that are ever the
                             dominant (argmax) path across the sampled dataset.
    ``leaf_counts``        — int ndarray (2^L,): per-leaf dominance count over
                             all (image, spatial-position) pairs seen.
    """
    encoder   = getattr(model, "encoder", None)
    if encoder is None:
        return {}
    bottleneck = getattr(encoder, "bottleneck_stage", None)
    if bottleneck is None:
        return {}
    layer_channels = getattr(bottleneck, "layer_channels", None)
    if layer_channels is None:
        return {}

    n_layers   = len(layer_channels)
    n_leaves   = layer_channels[-1]
    model.eval()

    depth_prob_mean_acc = [np.zeros(layer_channels[d], dtype=np.float64) for d in range(n_layers)]
    depth_prob_max      = [np.zeros(layer_channels[d], dtype=np.float64) for d in range(n_layers)]
    depth_entropy_acc   = [0.0] * n_layers
    leaf_counts         = np.zeros(n_leaves, dtype=np.int64)
    n_batches           = 0

    for batch_idx, (imgs, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        try:
            _, enc_details = encoder(imgs, return_details=True)
        except Exception:
            continue

        stages = enc_details.get("stages", [])
        if not stages:
            continue
        stage_logp = stages[0].get("logp")
        if stage_logp is None or not isinstance(stage_logp, torch.Tensor):
            continue

        # Split (B, total_channels, H, W) by layer_channels → per-depth logp
        logp_per_depth = torch.split(stage_logp.float(), list(layer_channels), dim=1)

        for d, logp_d in enumerate(logp_per_depth):
            prob_d = logp_d.exp()                                           # (B, 2^(d+1), H, W)
            depth_prob_mean_acc[d] += prob_d.mean(dim=(0, 2, 3)).cpu().numpy()
            depth_prob_max[d]       = np.maximum(
                depth_prob_max[d], prob_d.amax(dim=(0, 2, 3)).cpu().numpy()
            )
            # Per-spatial-position path entropy, normalised
            ent     = -(prob_d * logp_d.clamp(min=-20.0)).sum(dim=1).mean().item()
            max_ent = math.log(max(layer_channels[d], 2))
            depth_entropy_acc[d] += ent / max_ent

        # Dominant leaf path per (image, spatial-position)
        leaf_argmax = logp_per_depth[-1].argmax(dim=1)                      # (B, H, W)
        np.add.at(leaf_counts, leaf_argmax.cpu().numpy().ravel(), 1)
        n_batches += 1

    if n_batches == 0:
        return {}

    node_util    = []
    dead_nodes   = []
    mean_entropy = []
    for d in range(n_layers):
        mean_p = depth_prob_mean_acc[d] / n_batches
        node_util.append(float(np.mean(mean_p > prob_threshold)))
        dead_nodes.append(float(np.mean(depth_prob_max[d] <= prob_threshold)))
        mean_entropy.append(depth_entropy_acc[d] / n_batches)

    return {
        "n_layers":         n_layers,
        "layer_channels":   list(layer_channels),
        "node_util":        node_util,
        "dead_nodes":       dead_nodes,
        "mean_entropy":     mean_entropy,
        "leaf_utilization": float(np.sum(leaf_counts > 0)) / float(n_leaves),
        "leaf_counts":      leaf_counts,
    }


# ─── prefix reconstruction ────────────────────────────────────────────────────

@torch.no_grad()
def compute_prefix_reconstruction(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    save_dir: Path,
    run_name: str,
    max_batches: int = 30,
    max_visual_images: int = 8,
) -> Dict[str, object]:
    """Compute per-depth prefix reconstruction quality using ``forward_matryoshka``.

    At prefix depth d the decoder receives only the first d+1 levels of the
    hierarchy (deeper channels are zeroed), so we can observe how much each
    successive level contributes to reconstruction.

    Saves a visual grid (rows = original + one per depth, cols = sample images)
    to ``<save_dir>/taxon_hierarchy/<run_name>/prefix_recon_grid.png``.

    Returns dict with:
      ``n_layers``       — number of hierarchy depths
      ``mse_per_depth``  — list[float]: mean MSE per prefix depth
      ``psnr_per_depth`` — list[float]: PSNR (dB); pixel range [-1,1] so MAX_VAL=2
    """
    if not hasattr(model, "forward_matryoshka"):
        return {}
    model.eval()

    n_layers: Optional[int]       = None
    mse_sums: Optional[List[float]] = None
    n_batches  = 0
    grid_saved = False

    for batch_idx, (imgs, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        try:
            prefix_recons, _ = model.forward_matryoshka(imgs)
        except Exception as e:
            print(f"    [prefix_recon] forward_matryoshka failed: {e}")
            continue

        if n_layers is None:
            n_layers = len(prefix_recons)
            mse_sums = [0.0] * n_layers

        for d, recon_d in enumerate(prefix_recons):
            mse_sums[d] += float(F.mse_loss(recon_d, imgs).item())

        if not grid_saved:
            safe     = run_name.replace("/", "_").replace(" ", "_")
            grid_dir = save_dir / "taxon_hierarchy" / safe
            grid_dir.mkdir(parents=True, exist_ok=True)
            _save_prefix_recon_grid(
                imgs[:max_visual_images],
                [r[:max_visual_images] for r in prefix_recons],
                grid_dir / "prefix_recon_grid.png",
                run_name,
            )
            grid_saved = True
        n_batches += 1

    if n_batches == 0 or n_layers is None:
        return {}

    mse_per_depth  = [s / n_batches for s in mse_sums]
    psnr_per_depth = [float(10 * math.log10(4.0 / max(m, 1e-10))) for m in mse_per_depth]
    return {
        "n_layers":       n_layers,
        "mse_per_depth":  mse_per_depth,
        "psnr_per_depth": psnr_per_depth,
    }


def _save_prefix_recon_grid(
    imgs: torch.Tensor,
    prefix_recons: List[torch.Tensor],
    out_path: Path,
    run_name: str,
) -> None:
    """Grid: rows = original + each prefix depth; cols = sample images."""
    rows     = [imgs] + prefix_recons
    n_images = imgs.shape[0]
    n_rows   = len(rows)

    fig, axes = plt.subplots(n_rows, n_images, figsize=(n_images * 1.6, n_rows * 1.6))
    if n_rows   == 1: axes = axes[np.newaxis, :]
    if n_images == 1: axes = axes[:, np.newaxis]

    for row_idx, row_batch in enumerate(rows):
        row_label = "Original" if row_idx == 0 else f"Prefix d={row_idx}"
        for col_idx in range(n_images):
            ax      = axes[row_idx, col_idx]
            img_np  = (row_batch[col_idx].cpu().clamp(-1, 1) + 1.0) * 0.5
            ax.imshow(img_np.permute(1, 2, 0).float().numpy(), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=6.5, rotation=0,
                              ha="right", va="center", labelpad=60)

    fig.suptitle(f"Prefix Reconstruction — {run_name}", fontsize=9, fontweight="bold")
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_taxon_hierarchy_depth_stats(result: dict, save_dir: Path) -> None:
    """Four-panel hierarchy exploration chart for one taxon model.

    Panels:
    1. Node utilisation per depth (fraction with dataset-mean prob > threshold)
    2. Normalised path entropy per depth (0=fully peaked, 1=uniform)
    3. Dead node fraction per depth (never exceed threshold in any sample)
    4. Leaf path distribution — sorted dominance histogram over sampled dataset
    """
    hs = result.get("hierarchy_stats")
    if not hs or not hs.get("n_layers"):
        return

    n_layers    = hs["n_layers"]
    depths      = list(range(1, n_layers + 1))
    node_util   = hs["node_util"]
    dead_nodes  = hs["dead_nodes"]
    mean_ent    = hs["mean_entropy"]
    leaf_util   = hs["leaf_utilization"]
    leaf_counts = hs["leaf_counts"]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))

    # 1 — node utilisation
    ax = axes[0]
    ax.bar(depths, node_util, color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.axhline(leaf_util, color="red", lw=1.2, ls="--",
               label=f"leaf util {leaf_util:.1%}")
    ax.set_xlabel("Depth"); ax.set_ylabel("Fraction of nodes used")
    ax.set_title(f"Node Utilisation\n(mean prob > {_HIER_PROB_THRESHOLD:.0%})")
    ax.set_ylim(0, 1.05); ax.set_xticks(depths)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

    # 2 — normalised entropy
    ax = axes[1]
    ax.bar(depths, mean_ent, color="#ff7f0e", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Depth"); ax.set_ylabel("Normalised entropy")
    ax.set_title("Per-depth Path Entropy\n(normalised: 0=peaked / 1=uniform)")
    ax.set_ylim(0, 1.05); ax.set_xticks(depths); ax.grid(axis="y", alpha=0.3)

    # 3 — dead node fraction
    ax = axes[2]
    ax.bar(depths, dead_nodes, color="#d62728", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Depth"); ax.set_ylabel("Fraction never active")
    ax.set_title("Dead Node Fraction per Depth")
    ax.set_ylim(0, 1.05); ax.set_xticks(depths); ax.grid(axis="y", alpha=0.3)

    # 4 — leaf distribution (rank-sorted)
    ax       = axes[3]
    n_leaves = len(leaf_counts)
    sorted_c = np.sort(leaf_counts)[::-1]
    n_used   = int(np.sum(sorted_c > 0))
    ax.bar(np.arange(n_leaves), sorted_c,
           color="#2ca02c", width=1.0, edgecolor="none", linewidth=0)
    ax.set_xlabel("Leaf node (rank-sorted)")
    ax.set_ylabel("Times dominant across dataset")
    ax.set_title(f"Leaf Path Distribution\n({n_used}/{n_leaves} used = {leaf_util:.1%})")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Hierarchy Exploration — {result['short']}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    safe    = result["short"].replace("/", "_").replace(" ", "_")
    out_dir = save_dir / "taxon_hierarchy" / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "depth_stats.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out.relative_to(save_dir)}")


def plot_prefix_recon_curve(result: dict, save_dir: Path) -> None:
    """MSE and PSNR vs prefix depth for one taxon model."""
    pr = result.get("prefix_recon")
    if not pr or not pr.get("n_layers"):
        return

    n_layers       = pr["n_layers"]
    depths         = list(range(1, n_layers + 1))
    mse_per_depth  = pr["mse_per_depth"]
    psnr_per_depth = pr["psnr_per_depth"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(depths, mse_per_depth, marker="o", color="#1f77b4",
                 linewidth=1.8, markersize=5)
    axes[0].set_xlabel("Prefix depth"); axes[0].set_ylabel("MSE")
    axes[0].set_title("Prefix Reconstruction MSE\n(lower = better)")
    axes[0].set_xticks(depths); axes[0].grid(alpha=0.3)

    axes[1].plot(depths, psnr_per_depth, marker="o", color="#ff7f0e",
                 linewidth=1.8, markersize=5)
    axes[1].set_xlabel("Prefix depth"); axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Prefix Reconstruction PSNR\n(higher = better; range [-1, 1])")
    axes[1].set_xticks(depths); axes[1].grid(alpha=0.3)

    fig.suptitle(f"Prefix Reconstruction Quality — {result['short']}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    safe    = result["short"].replace("/", "_").replace(" ", "_")
    out_dir = save_dir / "taxon_hierarchy" / safe
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "prefix_recon_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out.relative_to(save_dir)}")


def _run_tsne(
    latents: np.ndarray,
    max_samples: int = 2000,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    pca_components: int = 50,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run (optional PCA →) t-SNE on latents.

    Returns
    -------
    emb     : (N, 2) 2-D t-SNE embedding
    idx_sub : (N,)   indices of the subsample used (into original latents array)
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(seed)
    N = latents.shape[0]
    n = min(max_samples, N)
    idx_sub = rng.choice(N, n, replace=False)
    X = latents[idx_sub].copy()

    # Standardise
    X = StandardScaler().fit_transform(X)

    # PCA first if dimensionality is high
    if X.shape[1] > pca_components:
        n_comp = min(pca_components, X.shape[0] - 1, X.shape[1])
        X = PCA(n_components=n_comp, random_state=seed).fit_transform(X)

    perp = min(perplexity, n / 4)
    emb = TSNE(
        n_components=2,
        perplexity=perp,
        max_iter=n_iter,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    ).fit_transform(X)

    return emb.astype(np.float32), idx_sub


def _tsne_scatter_ax(
    ax: "plt.Axes",
    emb: np.ndarray,
    labels: np.ndarray,
    title: str,
    size: float = 4.0,
    alpha: float = 0.6,
):
    """Draw a two-class coloured scatter on *ax* (1=present / 0=absent)."""
    pos = labels == 1
    neg = ~pos
    ax.scatter(emb[neg, 0], emb[neg, 1], s=size, c="#aec7e8", alpha=alpha,
               linewidths=0, rasterized=True, label="absent")
    ax.scatter(emb[pos, 0], emb[pos, 1], s=size, c="#1f77b4", alpha=alpha,
               linewidths=0, rasterized=True, label="present")
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def plot_tsne_latents(
    result: dict,
    save_dir: Path,
    max_samples: int = 2000,
    perplexity: float = 30.0,
    n_iter: int = 1000,
):
    """Per-model t-SNE grid: one panel per CelebA attribute, coloured by label.

    The embedding is shared across all 40 panels (computed once), so the
    spatial structure can be compared across attributes directly.
    """
    latents = result.get("_latents")
    attrs   = result.get("_attrs")
    if latents is None or attrs is None or latents.shape[0] < 50:
        print(f"  [t-SNE] No cached latents for {result['short']}, skipping.")
        return

    print(f"  [t-SNE] Running for {result['short']} ({latents.shape[0]} samples)...")
    try:
        emb, idx = _run_tsne(latents, max_samples=max_samples,
                             perplexity=perplexity, n_iter=n_iter)
    except Exception as e:
        print(f"  [t-SNE] Failed: {e}")
        return

    attrs_sub = attrs[idx]  # (N_sub, 40)

    n_attrs = len(_TSNE_ALL_ATTRS)
    ncols = 8
    nrows = (n_attrs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.2, nrows * 2.2))
    axes_flat = axes.flat

    for ai, aname in enumerate(_TSNE_ALL_ATTRS):
        ax = next(axes_flat)
        _tsne_scatter_ax(ax, emb, attrs_sub[:, ai], aname)

    # Hide unused axes
    for ax in axes_flat:
        ax.set_visible(False)

    model_label = f"{result['short']}  ({result['type']})"
    fig.suptitle(
        f"t-SNE of Sparse Latent — {model_label}\n"
        f"Blue = attribute present | Grey = absent   (n={emb.shape[0]})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Safe filename from short name
    safe = result["short"].replace("/", "_").replace(" ", "_").replace(",", "")
    out = save_dir / f"probe_celeba_hq_tsne_{safe}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [t-SNE] Saved: {out.name}")


def plot_tsne_comparison(
    all_results: List[dict],
    save_dir: Path,
    max_samples: int = 2000,
    perplexity: float = 30.0,
    n_iter: int = 1000,
):
    """Cross-model t-SNE comparison for headline attributes.

    Rows = models; columns = headline CelebA attributes.
    Each model's embedding is computed independently (same perplexity/init),
    so inter-model spatial alignment is only qualitative.
    """
    # Filter to models that have latents and are in the t-SNE types
    tsne_results = [
        r for r in all_results
        if r.get("type") in _TSNE_TYPES
        and r.get("_latents") is not None
        and r["_latents"].shape[0] >= 50
    ]
    if not tsne_results:
        print("  [t-SNE compare] No eligible models — skipping.")
        return

    headline_indices = [
        CELEBA_ATTR_NAMES.index(a) for a in _TSNE_HEADLINE_ATTRS
        if a in CELEBA_ATTR_NAMES
    ]
    headline_names = [CELEBA_ATTR_NAMES[i] for i in headline_indices]

    n_models  = len(tsne_results)
    n_cols    = len(headline_names)
    fig, axes = plt.subplots(n_models, n_cols,
                             figsize=(n_cols * 2.0, n_models * 2.0))
    if n_models == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, result in enumerate(tsne_results):
        latents = result["_latents"]
        attrs   = result["_attrs"]
        print(f"  [t-SNE compare] {result['short']} ({latents.shape[0]} samples)...")
        try:
            emb, idx = _run_tsne(latents, max_samples=max_samples,
                                 perplexity=perplexity, n_iter=n_iter)
        except Exception as e:
            print(f"    Failed: {e}")
            for ax in axes[row_idx]:
                ax.set_visible(False)
            continue

        attrs_sub = attrs[idx]

        for col_idx, (ai, aname) in enumerate(zip(headline_indices, headline_names)):
            ax = axes[row_idx, col_idx]
            _tsne_scatter_ax(ax, emb, attrs_sub[:, ai], "")
            if row_idx == 0:
                ax.set_title(aname, fontsize=8, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(result["short"], fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=60)

    fig.suptitle(
        "t-SNE Latent Space Comparison — CelebA-HQ\n"
        "Rows: models | Cols: attributes   (blue = present)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0.08, 0, 1, 0.96])

    out = save_dir / "probe_celeba_hq_tsne_compare.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [t-SNE compare] Saved: {out.name}")


# ─── parse args ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe CelebA-HQ model latents")
    p.add_argument("--outputs-dir", type=str, default="./outputs/celeba_hq",
                   help="Root outputs directory to scan (both / and /main/ subdirs)")
    p.add_argument("--data-root", type=str, default="./data",
                   help="Data root containing celeba_fallback/ for attribute labels")
    p.add_argument("--save-dir", type=str, default="./outputs/analysis_celeba_hq",
                   help="Directory to write analysis outputs")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=5000,
                   help="Max images to use for probing (caps both latent extraction and labels)")
    p.add_argument("--max-batches-sparsity", type=int, default=60,
                   help="Max batches for sparsity analysis (separate from probing)")
    p.add_argument("--n-train-frac", type=float, default=0.8,
                   help="Fraction of samples used for probe training")
    p.add_argument("--knn-k", type=int, nargs="+", default=[1, 5, 20],
                   help="k values for KNN probing")
    p.add_argument("--force-recompute", action="store_true",
                   help="Ignore cached latents and recompute from scratch")
    p.add_argument("--ablations", type=str, default=None, metavar="DIR",
                   help="Path to ablations outputs directory to include alongside "
                        "main runs (e.g. ./outputs/celeba_hq/ablations)")
    p.add_argument("--skip-sparsity", action="store_true",
                   help="Skip sparsity analysis (faster)")
    p.add_argument("--skip-knn", action="store_true",
                   help="Skip KNN probing")
    p.add_argument("--skip-tsne", action="store_true",
                   help="Skip t-SNE visualisation")
    p.add_argument("--tsne-max-samples", type=int, default=2000,
                   help="Max samples for t-SNE (keep ≤3000 for speed)")
    p.add_argument("--tsne-perplexity", type=float, default=30.0,
                   help="t-SNE perplexity")
    p.add_argument("--tsne-n-iter", type=int, default=1000,
                   help="t-SNE number of iterations")
    p.add_argument("--model-filter", type=str, default="",
                   help="Only process runs whose name contains this substring")
    p.add_argument("--skip-hierarchy", action="store_true",
                   help="Skip taxon hierarchy exploration analysis (bottleneck_topk_taxon only)")
    p.add_argument("--skip-prefix-recon", action="store_true",
                   help="Skip prefix reconstruction quality analysis (bottleneck_topk_taxon only)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = Path(args.outputs_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = save_dir / "latent_cache"

    print("=" * 80)
    print("CelebA-HQ Probing Analysis")
    print("=" * 80)
    print(f"  device      = {device}")
    print(f"  outputs_dir = {outputs_dir}")
    print(f"  save_dir    = {save_dir}")
    if args.ablations:
        print(f"  ablations   = {args.ablations}")
    print(f"  max_samples = {args.max_samples}")
    print("=" * 80)

    # ── discover runs ──────────────────────────────────────────────────────────
    runs = discover_runs(outputs_dir, include_ablations=False)
    if args.ablations:
        abl_dir = Path(args.ablations)
        if abl_dir.exists():
            abl_runs = discover_runs(abl_dir, include_ablations=True)
            existing = {r["path"] for r in runs}
            new_abl = [r for r in abl_runs if r["path"] not in existing]
            runs.extend(new_abl)
            print(f"  + {len(new_abl)} ablation run(s) from {abl_dir}")
        else:
            print(f"  WARNING: --ablations directory not found: {abl_dir}")
    if args.model_filter:
        runs = [r for r in runs if args.model_filter.lower() in r["name"].lower()]
    print(f"\nFound {len(runs)} runs to analyze:")
    for r in runs:
        print(f"  [{r['type']:35s}] {r['name']}")

    if not runs:
        print("No runs found. Exiting.")
        return

    # ── load attribute dataset ─────────────────────────────────────────────────
    print("\n[Step 1] Loading CelebA attribute dataset...")
    try:
        attr_loader = _load_celeba_attr_dataset(
            data_root=args.data_root,
            image_size=args.image_size,
            max_samples=args.max_samples,
        )
    except Exception as e:
        print(f"  WARNING: Could not load attribute dataset: {e}")
        print("  Linear probing and KNN will be skipped.")
        attr_loader = None

    # ── process each run ────────────────────────────────────────────────────────
    all_results = []

    for run_idx, run in enumerate(runs):
        print(f"\n[{run_idx + 1}/{len(runs)}] {run['name']}")
        result = {
            "name":     run["name"],
            "short":    run["short"],
            "type":     run["type"],
            "path":     str(run["path"]),
            "linear":   None,
            "knn":      None,
            "sparsity": {},
        }

        # Load model
        try:
            model, _ = load_model(run, device)
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            all_results.append(result)
            continue

        model.eval()

        latents, attrs = None, None  # initialised here so monosemanticity block is always safe

        # ── extract / load cached latents for probing ──────────────────────────
        if attr_loader is not None:
            cpath = _cache_path(run["path"], cache_dir)
            if not args.force_recompute:
                latents, attrs = _load_cached_latents(cpath)
                if latents is not None:
                    print(f"  Loaded cached latents: shape={latents.shape}")

            if latents is None:
                print(f"  Extracting latents ({args.max_samples} samples)...")
                latents, attrs = extract_latents(model, run["type"], attr_loader, device)
                if latents.shape[0] > 0:
                    _save_cached_latents(cpath, latents, attrs)
                    print(f"  Extracted: latents={latents.shape}, attrs={attrs.shape}")
                else:
                    print("  WARNING: No latents extracted, skipping.")
                    latents = None

            if latents is not None and latents.shape[0] >= 50:
                # ── stash latents for t-SNE (only for eligible model types) ────
                if run["type"] in _TSNE_TYPES and not args.skip_tsne:
                    result["_latents"] = latents
                    result["_attrs"]   = attrs

                # ── linear probing ──────────────────────────────────────────────
                print("  Running linear probe...")
                try:
                    result["linear"] = run_linear_probe(
                        latents, attrs, CELEBA_ATTR_NAMES, n_train_frac=args.n_train_frac)
                    m_auc = np.nanmean(result["linear"]["auc"])
                    m_acc = np.nanmean(result["linear"]["accuracy"])
                    print(f"  Linear probe: mean_AUC={m_auc:.4f}, mean_acc={m_acc:.4f}")
                except Exception as e:
                    print(f"  WARNING linear probe failed: {e}")

                # ── KNN ────────────────────────────────────────────────────────
                if not args.skip_knn:
                    print("  Running KNN...")
                    try:
                        result["knn"] = run_knn(
                            latents, attrs, CELEBA_ATTR_NAMES,
                            k_values=tuple(args.knn_k), n_train_frac=args.n_train_frac)
                        m_k5 = np.nanmean(result["knn"].get("k5_auc", [float("nan")]))
                        print(f"  KNN k=5: mean_AUC={m_k5:.4f}")
                    except Exception as e:
                        print(f"  WARNING KNN failed: {e}")

        # ── sparsity stats ──────────────────────────────────────────────────────
        if not args.skip_sparsity and attr_loader is not None:
            print("  Computing sparsity stats...")
            try:
                result["sparsity"] = compute_sparsity_stats(
                    model, attr_loader, device,
                    max_batches=args.max_batches_sparsity)
                sp = result["sparsity"]
                print(f"  Sparsity: L0={sp['mean_l0']:.4f}, dead={sp['dead_frac']:.4f}, "
                      f"selectivity={sp['selectivity']:.3f}, jaccard={sp['mean_jaccard']:.4f}")
            except Exception as e:
                print(f"  WARNING sparsity failed: {e}")

        # ── monosemanticity (requires latents + attrs) ──────────────────────────
        if latents is not None and attrs is not None and latents.shape[0] >= 50:
            try:
                mono = compute_monosemanticity(latents, attrs)
                if "sparsity" not in result or not result["sparsity"]:
                    result["sparsity"] = {}
                result["sparsity"]["monosemanticity"] = mono
                print(f"  Monosemanticity: {mono:.3f}")
            except Exception as e:
                print(f"  WARNING monosemanticity failed: {e}")

        # ── taxon hierarchy exploration ────────────────────────────────────────
        if (not args.skip_hierarchy
                and run["type"] in _TAXON_HIERARCHY_TYPES
                and attr_loader is not None):
            print("  Computing taxon hierarchy stats...")
            try:
                result["hierarchy_stats"] = compute_taxon_hierarchy_stats(
                    model, attr_loader, device,
                    max_batches=args.max_batches_sparsity)
                hs = result["hierarchy_stats"]
                if hs:
                    lu = hs["leaf_utilization"]
                    nu = hs["node_util"][-1]
                    print(f"  Hierarchy: leaf_util={lu:.1%}, node_util[last]={nu:.1%}")
            except Exception as e:
                print(f"  WARNING hierarchy stats failed: {e}")

        # ── prefix reconstruction ───────────────────────────────────────────────
        if (not args.skip_prefix_recon
                and hasattr(model, "forward_matryoshka")
                and attr_loader is not None):
            print("  Computing prefix reconstruction quality...")
            try:
                result["prefix_recon"] = compute_prefix_reconstruction(
                    model, attr_loader, device,
                    save_dir=save_dir, run_name=run["short"],
                    max_batches=30)
                pr = result["prefix_recon"]
                if pr:
                    print(f"  Prefix recon: full MSE={pr['mse_per_depth'][-1]:.4f}, "
                          f"PSNR={pr['psnr_per_depth'][-1]:.2f} dB")
            except Exception as e:
                print(f"  WARNING prefix recon failed: {e}")

        # Free model memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        all_results.append(result)

    if not all_results:
        print("No results to save. Exiting.")
        return

    # ── save outputs ─────────────────────────────────────────────────────────────
    print("\n[Saving results]")
    save_csv(all_results, save_dir)

    has_linear = any(r.get("linear") for r in all_results)
    has_knn    = any(r.get("knn") for r in all_results)
    has_sp     = any(r.get("sparsity") for r in all_results)

    if has_linear:
        plot_linear_results(all_results, save_dir)
        plot_attr_heatmap(all_results, save_dir)
    if has_knn:
        plot_knn_results(all_results, save_dir)
    if has_sp:
        plot_sparsity_results(all_results, save_dir)

    has_mono = any(r.get("sparsity", {}).get("monosemanticity") is not None
                   for r in all_results)
    if has_mono:
        plot_monosemanticity_scatter(all_results, save_dir)

    # ── taxon hierarchy and prefix reconstruction plots ───────────────────────
    hier_results   = [r for r in all_results if r.get("hierarchy_stats")]
    prefix_results = [r for r in all_results if r.get("prefix_recon")]
    if hier_results or prefix_results:
        print(f"\n[Taxon Hierarchy] {len(hier_results)} model(s) | "
              f"[Prefix Recon] {len(prefix_results)} model(s)")
        for r in hier_results:
            plot_taxon_hierarchy_depth_stats(r, save_dir)
        for r in prefix_results:
            plot_prefix_recon_curve(r, save_dir)

    # ── taxon vs. non-taxon sparsity-matched comparisons ─────────────────────
    has_taxon    = any(_is_taxon(r) for r in all_results)
    has_baseline = any(not _is_taxon(r) for r in all_results)
    if has_taxon and has_baseline:
        print("\n[Taxon vs. Baseline Comparisons]")
        if has_sp:
            plot_taxon_scatter(all_results, save_dir)
        plot_matched_pairs(all_results, save_dir)
        if has_linear:
            plot_taxon_attr_compare(all_results, save_dir)

    # ── t-SNE visualisations ──────────────────────────────────────────────────
    if not args.skip_tsne:
        tsne_results = [r for r in all_results if r.get("_latents") is not None]
        if tsne_results:
            print(f"\n[t-SNE] {len(tsne_results)} model(s) eligible")
            tsne_dir = save_dir / "tsne"
            tsne_dir.mkdir(parents=True, exist_ok=True)
            tsne_kw = dict(
                max_samples=args.tsne_max_samples,
                perplexity=args.tsne_perplexity,
                n_iter=args.tsne_n_iter,
            )
            for r in tsne_results:
                plot_tsne_latents(r, tsne_dir, **tsne_kw)
            if len(tsne_results) > 1:
                plot_tsne_comparison(all_results, tsne_dir, **tsne_kw)
        else:
            print("\n[t-SNE] No eligible models with latents — skipping.")

    print(f"\nAll outputs saved to: {save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
