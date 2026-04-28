#!/usr/bin/env python3
"""Publication-quality graphics for the TaxonomicWeights paper — ImageNet (Tiny).

Mirror of paper_graphics_celeba_hq.py adapted for TinyImageNet (200 classes, 64×64).

Output directory structure
──────────────────────────
  paper_graphics_imagenet/
    main/
      fig01_training_curves.png
      fig02_reconstruction_gallery.png
      fig03_sparsity_probing.png       Linear-probe top-1 vs L0
      fig04_sparsity_knn.png           5-NN top-1 vs L0
      fig05_monosemanticity.png        Monosemanticity vs L0
      fig06_metrics_overview.png
    ablation/
      fig07_ablation_depth.png
      fig08_ablation_multitaxon.png
    taxonomy/<model_name>/
      fig09_prefix_recon.png
      fig10_activations.png
      fig11_gradcam.png
    cache/<model_name>/
      metrics.json
      latents.npz

Usage
─────
  python src/paper/paper_graphics_imagenet.py \\
      [--data-root ./data/imagenet] \\
      [--out-dir paper_graphics_imagenet] \\
      [--max-images 3000] \\
      [--batch-size 32] \\
      [--skip-hierarchy] \\
      [--force-recompute]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── reuse existing helpers ────────────────────────────────────────────────────
from src.compare.compare_imagenet import (   # noqa: E402
    load_model as _load_model_from_run_dict,
    _model_type,
    _short_name,
    model_colour,
)
from src.analyze._taxon_hierarchy_viz_lib import (  # noqa: E402
    discover_hierarchies,
    collect_node_activations,
    constrained_path_for_image,
    path_prefix_reconstructions,
    _gradcam_for_image_node_subset,
    _tensor_to_uint8,
    _overlay,
    _depth_offsets,
)
from src.utils.dataloader import TinyImageNetLoader  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Model registry — exactly the models specified for the paper
# ─────────────────────────────────────────────────────────────────────────────

MAIN_CONFIG_RELPATHS: List[str] = [
    "configs/imagenet/baseline_ae_imagenet.json",
    "configs/imagenet/sae_imagenet.json",
    "configs/imagenet/jumprelu_sae_imagenet.json",
    "configs/imagenet/topk_sae_imagenet.json",
    "configs/imagenet/matryoshka_batch_topk_sae_imagenet.json",
    "configs/imagenet/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_imagenet_r18_v1_main_L8.json",
    "configs/celeba_hq/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_celeba_hq_r18_v1_main_L8_dkl1e2.json",
    "configs/imagenet/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_imagenet_r18_v6_main_K4_L6_gate4.json",
    "configs/imagenet/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_imagenet_r18_v6_main_K4_L8_gate1.json",
    "configs/imagenet/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_imagenet_r18_v6_main_K8_L6_gate1.json",
]

ABLATION_TAXON_RELPATHS: List[str] = [
    "configs/imagenet/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_imagenet_r18_v1_main_L4.json",
    "configs/imagenet/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_imagenet_r18_v1_main_L6.json",
    "configs/imagenet/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_imagenet_r18_v1_main_L8.json",
    "configs/imagenet/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_imagenet_r18_v1_main_L10.json",
]

ABLATION_MULTI_RELPATHS: List[str] = [
    "configs/imagenet/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_imagenet_r18_v6_main_K2_L6_gate1.json",
    "configs/imagenet/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_imagenet_r18_v6_main_K4_L4_gate1.json",
    "configs/imagenet/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_imagenet_r18_v6_main_K4_L6_gate1.json",
    "configs/imagenet/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_imagenet_r18_v6_main_K4_L8_gate1.json",
    "configs/imagenet/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_imagenet_r18_v6_main_K8_L4_gate1.json",
    "configs/imagenet/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_imagenet_r18_v6_main_K8_L6_gate1.json",
]

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint resolution  (mirrors paper_graphics_celeba_hq.py)
# ─────────────────────────────────────────────────────────────────────────────

def _training_suffix_for_config(cfg: dict) -> str:
    mc = cfg.get("model", {})
    tc = cfg.get("training", {})
    exp = cfg.get("experiment_name", "")
    variant = mc.get("model_variant", None)

    # ── bottleneck topk taxon ─────────────────────────────────────────────────
    if exp.startswith("bottleneck_topk_taxon_"):
        L = int(mc.get("bottleneck_n_taxonomy_layers", 6))
        return f"_L{L}"
    # ── bottleneck topk multi-taxon ───────────────────────────────────────────
    if exp.startswith("bottleneck_topk_multi_taxon_"):
        K = int(mc.get("n_hierarchies", 4))
        L = int(mc.get("bottleneck_n_taxonomy_layers", 6))
        gate_k_cfg = int(mc.get("gate_k", K))
        use_gate = bool(mc.get("use_inter_hierarchy_gate", False))
        gate_tag = f"_gate{gate_k_cfg}" if (use_gate and gate_k_cfg < K) else ""
        return f"_K{K}_L{L}{gate_tag}"
    # ── bottleneck taxon / multi-taxon (non-topk) ─────────────────────────────
    if exp.startswith("bottleneck_taxon_") or exp.startswith("bottleneck_multi_taxon_"):
        base_name = Path(cfg.get("output", {}).get("output_dir", "")).name
        if base_name and exp.startswith(base_name):
            return exp[len(base_name):]
        return ""

    if variant is not None:
        sw = tc.get("sparsity_weight", 1e-3)
        if variant == "matryoshka_batch_topk":
            k_vals = sorted(mc.get("k_values", [8, 4, 2]), reverse=True)
            k_str = "-".join(str(k) for k in k_vals)
            return f"_k{k_str}_sw{sw:.0e}"
        if variant == "topk":
            k = mc.get("topk_k", 64)
            return f"_k{k}_sw{sw:.0e}"
        if variant == "gated":
            ste = "_ste" if mc.get("use_gate_ste", False) else ""
            return f"_sw{sw:.0e}{ste}"
        if variant == "jumprelu":
            l0 = int(mc.get("target_l0", 64))
            return f"_l0{l0}_sw{sw:.0e}"
        spt = mc.get("sparsity_type", "l1")
        return f"_spw_{sw:.0e}_spt_{spt}"

    if "baseline" in exp:
        return ""
    return ""


def resolve_ckpt(cfg_path: Path) -> Optional[Path]:
    with open(cfg_path) as f:
        cfg = json.load(f)
    base = cfg.get("output", {}).get("output_dir", "")
    if not base:
        return None
    base = Path(base)
    if not base.is_absolute():
        base = ROOT / base
    suffix = _training_suffix_for_config(cfg)
    for candidate in (Path(str(base) + suffix), base):
        ckpt = candidate / "checkpoints" / "best.pt"
        if ckpt.exists():
            return ckpt
    return None


def build_run_dict(cfg_path: Path) -> Optional[Dict]:
    with open(cfg_path) as f:
        cfg = json.load(f)
    exp = cfg.get("experiment_name", cfg_path.stem)
    mtype = _model_type(exp)
    ckpt = resolve_ckpt(cfg_path)
    if ckpt is None:
        print(f"  [SKIP] checkpoint not found for {exp}")
        return None
    run_path = ckpt.parent.parent
    short = _short_name(exp)
    return {
        "name":      exp,
        "short":     short,
        "type":      mtype,
        "path":      run_path,
        "best_ckpt": ckpt,
        "history":   run_path / "training_history.json",
        "cfg":       cfg,
        "cfg_path":  cfg_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def make_imagenet_loaders(
    data_root: str, image_size: int = 64,
    batch_size: int = 64, num_workers: int = 4,
    max_val_images: int = 5000,
) -> Tuple[Optional[object], Optional[object]]:
    """Return (train_loader, val_loader) from TinyImageNetLoader.

    TinyImageNetLoader pulls from HuggingFace (zh-plus/tiny-imagenet),
    caching under data_root.  It has no val_subset parameter so we create
    a subsetted DataLoader manually after construction.
    """
    loader = TinyImageNetLoader(
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        cache_dir=data_root if data_root else None,
    )
    train_loader, val_loader = loader.get_loaders()

    # Optionally restrict val set size
    if max_val_images and max_val_images < len(loader.valset):
        from torch.utils.data import Subset, DataLoader as DL
        sub = Subset(loader.valset, list(range(max_val_images)))
        val_loader = DL(sub, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)


def compute_sparsity_metrics(Z: np.ndarray, threshold: float = 1e-6) -> Dict:
    l0_per = (np.abs(Z) > threshold).sum(axis=1).astype(float)
    max_act = np.abs(Z).max(axis=0)
    dead_frac = float((max_act < threshold).mean())
    return dict(
        l0_mean=float(l0_per.mean()),
        l0_median=float(np.median(l0_per)),
        dead_frac=dead_frac,
        n_features=int(Z.shape[1]),
    )


def compute_monosemanticity_imagenet(
    Z: np.ndarray, labels: np.ndarray,
    n_classes: int = 200, top_k: int = 50, min_active_frac: float = 0.005,
) -> float:
    """Mean max-class lift over active features.

    For each active feature collect the top-K activating images, compute
    lift_j = freq(class_j | top-K) / (base_freq(class_j) + eps) for every
    class, and record the max lift.
    The model score is the mean max lift over qualifying features.

    A random feature gives lift ≈ 1; a perfectly class-selective feature
    gives lift ≈ n_classes (≈ 200 for TinyImageNet).

    Z:      [N, D] float32 latent matrix
    labels: [N]    int class indices (0-199)
    """
    N, D = Z.shape
    k = min(top_k, N)
    labels_oh = np.zeros((N, n_classes), dtype=np.float32)
    for i, c in enumerate(labels):
        if 0 <= int(c) < n_classes:
            labels_oh[i, int(c)] = 1.0
    base_freq = labels_oh.mean(axis=0) + 1e-6  # [n_classes]
    thresh = float(np.quantile(Z[Z > 0], min_active_frac)) if (Z > 0).any() else 0.0
    active_mask = Z.max(axis=0) > thresh
    scores: List[float] = []
    for c in range(D):
        if not active_mask[c]:
            continue
        top_idx = np.argpartition(Z[:, c], -k)[-k:]
        freq_top = labels_oh[top_idx].mean(axis=0)  # [n_classes]
        lift = freq_top / base_freq
        scores.append(float(lift.max()))
    return float(np.mean(scores)) if scores else float("nan")


@torch.no_grad()
def collect_latents_and_recon_safe(
    model, run: Dict, loader, device: torch.device, max_images: int = 0
) -> Dict:
    Zs, labels_out, mses, maes = [], [], [], []
    n_seen = 0
    mtype = run["type"]
    for batch in tqdm(loader, desc=f"    {run['short'][:25]}", leave=False):
        if max_images > 0 and n_seen >= max_images:
            break
        imgs, labels = batch[0].to(device), batch[1]
        take = min(imgs.shape[0], max_images - n_seen) if max_images > 0 else imgs.shape[0]
        imgs = imgs[:take]
        labels = labels[:take]

        try:
            if mtype in ("bottleneck_topk_taxon", "bottleneck_topk_multi_taxon"):
                recon_out, _ = model(imgs)
            elif mtype == "matryoshka_batch_topk_sae":
                recon_list, _ = model(imgs)
                recon_out = recon_list[0] if isinstance(recon_list, (list, tuple)) else recon_list
            elif mtype in ("topk_sae", "jumprelu_sae", "gated_sae", "sae"):
                recon_out, _ = model(imgs)
            elif mtype == "baseline":
                recon_out = model(imgs)
            else:
                out = model(imgs)
                recon_out = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(recon_out, (tuple, list)):
                recon_out = recon_out[0]
            z, *_ = model.encode(imgs)
            z_flat = z.detach().cpu().flatten(start_dim=1)
        except Exception as exc:
            print(f"    [warn] forward error for {run['short']}: {exc}")
            continue

        Zs.append(z_flat.numpy())
        labels_out.append(labels.numpy())
        mses.extend(((imgs - recon_out) ** 2).mean(dim=(1, 2, 3)).cpu().numpy().tolist())
        maes.extend(torch.abs(imgs - recon_out).mean(dim=(1, 2, 3)).cpu().numpy().tolist())
        n_seen += take

    if not Zs:
        return {}
    return dict(
        Z=np.concatenate(Zs),
        labels=np.concatenate(labels_out),
        mse=np.array(mses),
        mae=np.array(maes),
    )


@torch.no_grad()
def extract_probe_features_imagenet(
    model, loader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (Z [N,D], Y [N]) where Y is class index 0-199."""
    Zs, Ys = [], []
    for imgs, labels in tqdm(loader, desc="    probe encode", leave=False):
        imgs = imgs.to(device)
        z, *_ = model.encode(imgs)
        Zs.append(z.cpu().flatten(start_dim=1).numpy())
        Ys.append(labels.numpy())
    if not Zs:
        return np.zeros((0, 1)), np.zeros(0, dtype=np.int64)
    return np.concatenate(Zs), np.concatenate(Ys).astype(int)


def run_probe_imagenet(Z_tr, Y_tr, Z_va, Y_va, device: torch.device,
                       n_classes: int = 200) -> Dict:
    """GPU multi-class linear probe + KNN for TinyImageNet."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier

    scaler = StandardScaler()
    Z_tr_s = scaler.fit_transform(Z_tr)
    Z_va_s = scaler.transform(Z_va)

    # GPU linear probe
    t0 = time.time()
    D = Z_tr_s.shape[1]
    dev = device
    Zt = torch.from_numpy(Z_tr_s).float().to(dev)
    Yt = torch.from_numpy(Y_tr).long().to(dev)
    Zv = torch.from_numpy(Z_va_s).float().to(dev)

    clf = torch.nn.Linear(D, n_classes).to(dev)
    torch.nn.init.zeros_(clf.weight); torch.nn.init.zeros_(clf.bias)
    opt = torch.optim.Adam(clf.parameters(), lr=3e-3, weight_decay=1e-4)
    n, bs = len(Zt), 256
    for ep in range(200):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            opt.zero_grad()
            F.cross_entropy(clf(Zt[idx]), Yt[idx]).backward()
            opt.step()
    with torch.no_grad():
        preds = clf(Zv).argmax(dim=1).cpu().numpy()
    top1_acc = float((preds == Y_va).mean())
    print(f"    probe done  top1={top1_acc:.3f}  t={time.time()-t0:.1f}s")

    # KNN (k=5)
    knn_k5 = float("nan")
    try:
        knn = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1, algorithm="brute")
        knn.fit(Z_tr_s, Y_tr)
        knn_k5 = float((knn.predict(Z_va_s) == Y_va).mean())
    except Exception as e:
        print(f"    [warn] KNN failed: {e}")

    return dict(
        probe_top1_acc=top1_acc,
        probe_mean_auc=float("nan"),   # n/a for 200-class classification
        knn_k5_mean_acc=knn_k5,
    )


def compute_all_metrics(
    run: Dict, val_loader, device: torch.device,
    train_loader_for_probe, max_images: int,
    cache_dir: Path, force: bool,
) -> Dict:
    cache_file = cache_dir / run["name"] / "metrics.json"
    latents_file = cache_dir / run["name"] / "latents.npz"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and latents_file.exists() and not force:
        with open(cache_file) as f:
            return json.load(f)

    print(f"\n  Computing metrics for {run['short']} ...")
    model, _ = _load_model_from_run_dict(run, device)
    model.eval()

    data = collect_latents_and_recon_safe(model, run, val_loader, device, max_images)
    if not data:
        return {}
    Z = data["Z"]
    labels = data.get("labels", np.zeros(len(Z), dtype=np.int64))
    mse_mean = float(data["mse"].mean())
    psnr_mean = _psnr(mse_mean, max_val=2.0)

    sp = compute_sparsity_metrics(Z)
    metrics: Dict = dict(mse_mean=mse_mean, mae_mean=float(data["mae"].mean()),
                         psnr_mean=psnr_mean, **sp)

    np.savez_compressed(str(latents_file), Z=Z, labels=labels)

    # Monosemanticity
    if labels is not None and len(np.unique(labels)) > 1:
        metrics["monosemanticity"] = float(compute_monosemanticity_imagenet(Z, labels))

    # Probe + KNN (uses train_loader for Z_tr, val_loader Z_va already encoded)
    if train_loader_for_probe is not None:
        try:
            Z_tr, Y_tr = extract_probe_features_imagenet(model, train_loader_for_probe, device)
            Z_va, Y_va = Z, labels
            probe_metrics = run_probe_imagenet(Z_tr, Y_tr, Z_va, Y_va, device)
            metrics.update(probe_metrics)
        except Exception as e:
            print(f"    [warn] probe failed: {e}")

    with open(cache_file, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers (reuse patterns from celeba_hq version)
# ─────────────────────────────────────────────────────────────────────────────

_RCPARAMS = {
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
}

def _apply_style():
    plt.rcParams.update(_RCPARAMS)


def _assign_colours(runs: List[Dict]) -> List[str]:
    counts: Dict[str, int] = {}
    colours = []
    for r in runs:
        idx = counts.get(r["type"], 0)
        colours.append(model_colour(r["type"], idx))
        counts[r["type"]] = idx + 1
    return colours


# ─────────────────────────────────────────────────────────────────────────────
# Main figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_training_curves(runs: List[Dict], save_path: Path) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = _assign_colours(runs)
    ax_recon, ax_dead = axes

    for run, col in zip(runs, colours):
        hist_path = run.get("history")
        if hist_path is None or not Path(hist_path).exists():
            continue
        try:
            with open(hist_path) as f:
                h = json.load(f)
        except Exception:
            continue
        recon_key = next((k for k in ("val_recon", "val_recon_k0", "val_loss") if k in h), None)
        if recon_key is None:
            continue
        epochs = h.get("epochs", list(range(1, len(h[recon_key]) + 1)))
        ax_recon.plot(epochs, h[recon_key], label=run["short"], color=col, linewidth=1.5)
        if "val_dead" in h:
            ax_dead.plot(epochs, h["val_dead"], label=run["short"], color=col, linewidth=1.5)

    ax_recon.set_title("Val reconstruction loss"); ax_recon.set_xlabel("Epoch")
    ax_recon.set_ylabel("Recon loss"); ax_recon.grid(alpha=0.25)
    ax_recon.legend(ncol=2, loc="upper right", framealpha=0.8)

    ax_dead.set_title("Val dead-feature fraction"); ax_dead.set_xlabel("Epoch")
    ax_dead.set_ylabel("Dead frac"); ax_dead.grid(alpha=0.25)
    ax_dead.legend(ncol=2, loc="upper right", framealpha=0.8)

    fig.suptitle("Training dynamics — TinyImageNet", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def _scatter_vs_l0(runs, metrics_list, y_key, y_label, title, save_path,
                   higher_is_better=True):
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colours = _assign_colours(runs)
    for run, m, col in zip(runs, metrics_list, colours):
        l0 = m.get("l0_mean", float("nan"))
        y = m.get(y_key, float("nan"))
        if math.isnan(l0) or math.isnan(y):
            continue
        ax.scatter(l0, y, color=col, s=100, zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(run["short"], (l0, y), textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color=col, ha="left")
    ax.set_xlabel("Mean L0 (active features per image)", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.25)
    import matplotlib.patches as mpatches
    seen: Dict[str, str] = {}
    for r, c in zip(runs, colours):
        if r["type"] not in seen:
            seen[r["type"]] = c
    handles = [mpatches.Patch(color=c, label=t.replace("_", " ")) for t, c in seen.items()]
    ax.legend(handles=handles, loc="best", framealpha=0.8, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def fig_metrics_overview(runs, metrics_list, save_path):
    _apply_style()
    metric_keys = [
        ("mse_mean",        "MSE ↓"),
        ("psnr_mean",       "PSNR ↑ (dB)"),
        ("l0_mean",         "Mean L0"),
        ("dead_frac",       "Dead frac"),
        ("probe_top1_acc",  "Linear probe top-1 ↑"),
        ("knn_k5_mean_acc", "5-NN top-1 ↑"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    colours = _assign_colours(runs)
    labels = [r["short"] for r in runs]
    x = np.arange(len(labels))
    for ax, (key, lbl) in zip(axes, metric_keys):
        vals = [m.get(key, float("nan")) for m in metrics_list]
        bars = ax.bar(x, vals, color=colours, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(lbl, fontsize=10); ax.grid(axis="y", alpha=0.25)
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
    fig.suptitle("Metrics Overview — TinyImageNet (main models)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


@torch.no_grad()
def fig_reconstruction_gallery(runs, val_loader, device, save_path, n_images=6):
    _apply_style()
    batch = next(iter(val_loader))
    imgs = batch[0][:n_images].to(device)
    n_rows = len(runs) + 1
    fig, axes = plt.subplots(n_rows, n_images, figsize=(n_images * 1.8, n_rows * 1.8))

    def _show(ax, img_t):
        np_img = (img_t.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        ax.imshow(np_img); ax.axis("off")

    for j in range(n_images):
        _show(axes[0, j], imgs[j])
    axes[0, 0].set_ylabel("Original", rotation=0, ha="right", va="center", fontsize=8)

    counts: Dict[str, int] = {}
    for i, run in enumerate(runs):
        try:
            model, _ = _load_model_from_run_dict(run, device)
            model.eval()
        except Exception as e:
            print(f"  [warn] {run['short']}: {e}")
            continue
        try:
            mtype = run["type"]
            if mtype in ("bottleneck_topk_taxon", "bottleneck_topk_multi_taxon"):
                recon, _ = model(imgs)
            elif mtype == "matryoshka_batch_topk_sae":
                recon_list, _ = model(imgs)
                recon = recon_list[0] if isinstance(recon_list, (list, tuple)) else recon_list
            elif mtype == "baseline":
                recon = model(imgs)
            else:
                out = model(imgs)
                recon = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(recon, (tuple, list)):
                recon = recon[0]
        except Exception as e:
            print(f"  [warn] forward {run['short']}: {e}")
            continue
        row_idx = i + 1
        for j in range(n_images):
            _show(axes[row_idx, j], recon[j])
        idx = counts.get(run["type"], 0); counts[run["type"]] = idx + 1
        axes[row_idx, 0].set_ylabel(run["short"], rotation=0, ha="right", va="center",
                                     fontsize=7, color=model_colour(run["type"], idx))
        del model

    fig.suptitle("Reconstruction Gallery — TinyImageNet", fontsize=13, y=1.0)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation_bar(runs, metrics_list, save_path, title):
    _apply_style()
    metric_keys = [
        ("mse_mean",        "MSE ↓"),
        ("psnr_mean",       "PSNR ↑"),
        ("l0_mean",         "Mean L0"),
        ("dead_frac",       "Dead frac"),
        ("probe_top1_acc",  "Probe top-1 ↑"),
        ("knn_k5_mean_acc", "5-NN top-1 ↑"),
        ("monosemanticity", "Monosemanticity ↑"),
    ]
    valid_keys = [(k, lbl) for k, lbl in metric_keys
                  if any(not math.isnan(m.get(k, float("nan"))) for m in metrics_list)]
    ncols = min(4, len(valid_keys))
    nrows = math.ceil(len(valid_keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes_flat = np.array(axes).reshape(-1)
    colours = _assign_colours(runs)
    labels = [r["short"] for r in runs]
    x = np.arange(len(labels))
    for ax_i, (key, lbl) in enumerate(valid_keys):
        ax = axes_flat[ax_i]
        vals = [m.get(key, float("nan")) for m in metrics_list]
        bars = ax.bar(x, vals, color=colours, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(lbl, fontsize=10); ax.grid(axis="y", alpha=0.25)
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    for ax in axes_flat[len(valid_keys):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Case study: t-SNE
# ─────────────────────────────────────────────────────────────────────────────

_TINY_IMAGENET_CLASS_NAMES: Optional[List[str]] = None

def _get_tiny_imagenet_class_names(data_root: str) -> List[str]:
    """Return list of 200 class names in label-index order.
    Falls back to 'class_N' strings if wnids.txt is not found."""
    global _TINY_IMAGENET_CLASS_NAMES
    if _TINY_IMAGENET_CLASS_NAMES is not None:
        return _TINY_IMAGENET_CLASS_NAMES
    # HuggingFace dataset has a 'label' feature with names
    try:
        from datasets import load_dataset
        ds_info = load_dataset("zh-plus/tiny-imagenet",
                               cache_dir=data_root or None,
                               split="train").features["label"]
        _TINY_IMAGENET_CLASS_NAMES = ds_info.names
        return _TINY_IMAGENET_CLASS_NAMES
    except Exception:
        pass
    _TINY_IMAGENET_CLASS_NAMES = [f"class_{i}" for i in range(200)]
    return _TINY_IMAGENET_CLASS_NAMES


def _run_tsne(Z: np.ndarray, n_components: int = 2, perplexity: float = 40,
              max_samples: int = 3000, seed: int = 42):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    N = min(len(Z), max_samples)
    idx = rng.choice(len(Z), N, replace=False)
    Z_sub = StandardScaler().fit_transform(Z[idx])
    emb = TSNE(n_components=n_components, perplexity=perplexity,
               init="pca", random_state=seed, n_jobs=-1).fit_transform(Z_sub)
    return emb, idx


def fig_tsne_imagenet(
    run: Dict, cache_dir: Path, val_loader,
    device: torch.device, save_path: Path,
    n_classes_shown: int = 9, max_tsne_samples: int = 3000,
    data_root: str = "", force: bool = False,
) -> None:
    """t-SNE coloured by ImageNet class — highlights n most-populated classes."""
    _apply_style()
    tsne_cache = cache_dir / run["name"] / "tsne.npz"
    tsne_cache.parent.mkdir(parents=True, exist_ok=True)

    if tsne_cache.exists() and not force:
        d = np.load(tsne_cache, allow_pickle=True)
        emb, idx, labels = d["emb"], d["idx"], d["labels"]
    else:
        print(f"  [tsne] encoding for {run['short']} ...")
        model, _ = _load_model_from_run_dict(run, device)
        model.eval()
        Zs, Ls = [], []
        n_seen = 0
        for imgs, lbls in tqdm(val_loader, desc="  tsne encode", leave=False):
            if n_seen >= max_tsne_samples:
                break
            take = min(imgs.shape[0], max_tsne_samples - n_seen)
            imgs = imgs[:take].to(device)
            z, *_ = model.encode(imgs)
            Zs.append(z.detach().cpu().flatten(start_dim=1).numpy())
            Ls.append(lbls[:take].numpy())
            n_seen += take
        del model
        Z_all = np.concatenate(Zs)
        L_all = np.concatenate(Ls)
        emb, idx = _run_tsne(Z_all, max_samples=max_tsne_samples)
        labels = L_all[idx]
        np.savez_compressed(str(tsne_cache), emb=emb, idx=idx, labels=labels)

    class_names = _get_tiny_imagenet_class_names(data_root)

    # Pick n_classes_shown most frequent classes in the subsample
    counts = np.bincount(labels, minlength=200)
    top_classes = np.argsort(-counts)[:n_classes_shown]

    ncols = 3
    nrows = math.ceil(n_classes_shown / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.0))
    axes_flat = np.array(axes).reshape(-1)

    cmap = plt.cm.get_cmap("tab20", n_classes_shown)
    for plot_i, cls in enumerate(top_classes):
        ax = axes_flat[plot_i]
        is_cls = (labels == cls)
        # background points
        ax.scatter(emb[~is_cls, 0], emb[~is_cls, 1],
                   c="#cccccc", s=2, alpha=0.4, linewidths=0, rasterized=True)
        # foreground points
        ax.scatter(emb[is_cls, 0], emb[is_cls, 1],
                   color=cmap(plot_i), s=10, alpha=0.8,
                   linewidths=0, rasterized=True, zorder=5)
        name = class_names[cls] if cls < len(class_names) else f"class {cls}"
        ax.set_title(f"{name}\n(n={is_cls.sum()})", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    for ax in axes_flat[n_classes_shown:]:
        ax.axis("off")

    fig.suptitle(f"t-SNE latent space — {run['short']} — TinyImageNet",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Case study: neuron steering
# ─────────────────────────────────────────────────────────────────────────────

def _find_selective_neurons_imagenet(
    Z: np.ndarray, labels: np.ndarray,
    n_per_class: int = 1, min_samples: int = 10, min_active: float = 1e-6,
) -> Dict[int, List[int]]:
    """For each class return the most selective neurons.
    Selectivity = mean_activation_for_class / (global_mean + ε).
    """
    result: Dict[int, List[int]] = {}
    global_mean = Z.mean(axis=0) + 1e-8
    live = (np.abs(Z).max(axis=0) >= min_active)
    for cls in range(200):
        mask = labels == cls
        if mask.sum() < min_samples:
            continue
        mean_cls = Z[mask].mean(axis=0)
        selectivity = mean_cls / global_mean
        selectivity[~live] = 0.0
        top = np.argsort(-selectivity)[:n_per_class]
        result[cls] = top.tolist()
    return result


def fig_neuron_steering_imagenet(
    run: Dict, cache_dir: Path, train_loader, val_loader,
    device: torch.device, save_path: Path,
    data_root: str = "", n_classes: int = 6, n_images: int = 4,
    steer_scale: float = 10.0, force: bool = False,
) -> None:
    """Find class-selective neurons; dial them up on out-of-class images.

    Layout: one row per selected class.
    Columns: [original image] [steer ×0 (recon)] [steer ×1] [steer ×3] [steer ×N]
    """
    _apply_style()
    steer_cache = cache_dir / run["name"] / "steer_neurons.npz"
    steer_cache.parent.mkdir(parents=True, exist_ok=True)

    if steer_cache.exists() and not force:
        sc = np.load(steer_cache, allow_pickle=True)
        sel = {int(k): list(v) for k, v in sc["sel"].item().items()}
    else:
        print(f"  [steer] computing selectivity for {run['short']} ...")
        model, _ = _load_model_from_run_dict(run, device)
        model.eval()
        Zs, Ls = [], []
        n_seen = 0
        max_enc = 5000
        for imgs, lbls in tqdm(train_loader, desc="  steer encode", leave=False):
            if n_seen >= max_enc:
                break
            take = min(imgs.shape[0], max_enc - n_seen)
            imgs = imgs[:take].to(device)
            z, *_ = model.encode(imgs)
            Zs.append(z.detach().cpu().flatten(start_dim=1).numpy())
            Ls.append(lbls[:take].numpy())
            n_seen += take
        del model
        Z_tr = np.concatenate(Zs)
        L_tr = np.concatenate(Ls)
        sel = _find_selective_neurons_imagenet(Z_tr, L_tr)
        np.savez_compressed(str(steer_cache), sel=np.array(sel, dtype=object))

    if not sel:
        print(f"  [steer] no selective neurons found")
        return

    class_names = _get_tiny_imagenet_class_names(data_root)
    scales = [0.0, 1.0, 3.0, steer_scale]
    n_scales = len(scales)
    chosen_classes = sorted(sel.keys())[:n_classes]

    # Collect a large pool so each row can use a different image
    all_imgs, all_labels = [], []
    for imgs, lbls in val_loader:
        all_imgs.append(imgs)
        all_labels.append(lbls.numpy())
        if sum(b.shape[0] for b in all_imgs) >= 600:
            break
    all_imgs_t = torch.cat(all_imgs, 0)
    all_labels_np = np.concatenate(all_labels)

    model, _ = _load_model_from_run_dict(run, device)
    model.eval()

    fig, axes = plt.subplots(
        n_classes, 1 + n_scales,
        figsize=((1 + n_scales) * 2.5, n_classes * 2.8),
    )
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["Original"] + [f"steer ×{s:.0f}" for s in scales]
    for col_j, lbl in enumerate(col_labels):
        axes[0, col_j].set_title(lbl, fontsize=8)

    def _show(ax, img_t):
        np_img = (img_t.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        ax.imshow(np_img); ax.axis("off")

    for row_i, cls in enumerate(chosen_classes):
        neuron_idx = sel[cls][0] if isinstance(sel[cls], list) else int(sel[cls])
        cls_name = (class_names[cls] if cls < len(class_names) else f"class {cls}").replace("_", " ")

        # Images from different classes; rotate offset per row for visual variety
        out_mask = all_labels_np != cls
        out_indices = np.where(out_mask)[0]
        if len(out_indices) == 0:
            continue
        offset = row_i % max(len(out_indices) - n_images + 1, 1)
        face_indices = out_indices[offset : offset + n_images]
        if len(face_indices) < n_images:
            face_indices = out_indices[:n_images]
        imgs_out = all_imgs_t[face_indices].to(device)

        with torch.no_grad():
            z_raw, *_ = model.encode(imgs_out)
        z_shape = z_raw.shape
        z_flat = z_raw.flatten(start_dim=1).clone()
        ref_val = max(float(torch.quantile(z_flat[:, neuron_idx].abs(), 0.95).item()), 1e-3)

        ax_orig = axes[row_i, 0]
        _show(ax_orig, imgs_out[0].cpu())
        ax_orig.text(
            0.03, 0.97, f"{cls_name}\n(n°{neuron_idx})",
            transform=ax_orig.transAxes,
            fontsize=7, color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0),
        )

        for col_j, scale in enumerate(scales):
            z_edit = z_flat.clone()
            z_edit[:, neuron_idx] = scale * ref_val
            z_decode = z_edit.reshape(z_shape).to(device)
            with torch.no_grad():
                recon = model.decode(z_decode)
                if isinstance(recon, (tuple, list)):
                    recon = recon[0]
            _show(axes[row_i, 1 + col_j], recon[0].cpu())

    del model
    fig.suptitle(f"Neuron Steering — {run['short']} — TinyImageNet",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy exploration figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_hierarchy_prefix_recon(model, val_loader, device, save_path, n_images=4, run_short=""):
    """Prefix reconstruction using model.forward_matryoshka (same as probe analysis).

    Layout: rows = (Original + each depth),  cols = sample images.
    Falls back to path_prefix_reconstructions if forward_matryoshka unavailable.
    """
    _apply_style()

    # collect images
    all_imgs: List = []
    for batch in val_loader:
        all_imgs.append(batch[0])
        if sum(b.shape[0] for b in all_imgs) >= n_images:
            break
    if not all_imgs:
        return
    imgs = torch.cat(all_imgs, 0)[:n_images].to(device)

    if hasattr(model, "forward_matryoshka"):
        model.eval()
        with torch.no_grad():
            try:
                prefix_recons, _ = model.forward_matryoshka(imgs)
            except Exception as e:
                print(f"  [prefix_recon] forward_matryoshka failed: {e}")
                return
        rows = [imgs.cpu()] + [r.detach().cpu() for r in prefix_recons]
        n_levels = len(prefix_recons)
    else:
        metas = discover_hierarchies(model)
        if not metas:
            return
        meta = metas[0]
        n_levels = meta.n_levels
        acts, _ = collect_node_activations(model, val_loader, [meta], device, max_samples=500)
        act_mat = acts[meta.label]
        recon_stacks: List = []
        for img_i, img_t in enumerate(imgs):
            acts_row = act_mat[img_i] if img_i < act_mat.shape[0] else np.zeros(meta.n_nodes)
            path = constrained_path_for_image(acts_row, meta)
            recon_stacks.append(
                path_prefix_reconstructions(model, img_t.unsqueeze(0), meta, path, device)
            )
        prefix_recons_t = []
        for d in range(n_levels):
            depth_imgs = []
            for img_i in range(len(imgs)):
                if d < len(recon_stacks[img_i]):
                    r = recon_stacks[img_i][d]
                    t = torch.from_numpy(r).float().permute(2, 0, 1) / 127.5 - 1.0
                else:
                    t = torch.zeros_like(imgs[0].cpu())
                depth_imgs.append(t)
            prefix_recons_t.append(torch.stack(depth_imgs))
        rows = [imgs.cpu()] + prefix_recons_t

    n_rows = 1 + n_levels
    n_cols = len(imgs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 1.7))
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]

    row_labels = ["Original"] + [f"Prefix d={d+1}" for d in range(n_levels)]
    for row_idx, (row_batch, row_lbl) in enumerate(zip(rows, row_labels)):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            img_np = (row_batch[col_idx].clamp(-1, 1) + 1.0) * 0.5
            ax.imshow(img_np.permute(1, 2, 0).float().numpy(), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(row_lbl, fontsize=6.5, rotation=0,
                              ha="right", va="center", labelpad=60)

    title = f"Prefix Reconstruction — {run_short}" if run_short else "Prefix Reconstruction"
    fig.suptitle(title, fontsize=9, fontweight="bold")
    plt.tight_layout(rect=[0.08, 0, 1, 0.97])
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def fig_hierarchy_activations(model, val_loader, device, save_path):
    _apply_style()
    metas = discover_hierarchies(model)
    if not metas:
        return
    acts_dict, _ = collect_node_activations(model, val_loader, metas, device, max_samples=2000)
    n_hier = len(metas)
    fig, axes = plt.subplots(n_hier, 1, figsize=(14, 3 * n_hier), squeeze=False)
    for h_i, meta in enumerate(metas):
        ax = axes[h_i, 0]
        acts = acts_dict[meta.label]
        mean_acts = acts.mean(axis=0)
        offs = _depth_offsets(meta.layer_channels)
        colors_per_level = plt.cm.Blues(np.linspace(0.3, 0.9, meta.n_levels))
        x_offset = 0
        xtick_pos, xtick_lbl = [], []
        for d in range(meta.n_levels):
            n_ch = meta.layer_channels[d]
            start = offs[d]
            vals = mean_acts[start:start + n_ch]
            xs = np.arange(x_offset, x_offset + n_ch)
            ax.bar(xs, vals, color=colors_per_level[d], label=f"Depth {d+1}")
            xtick_pos.append(x_offset + n_ch / 2)
            xtick_lbl.append(f"D{d+1}\n({n_ch} nodes)")
            x_offset += n_ch + 1
        ax.set_xticks(xtick_pos); ax.set_xticklabels(xtick_lbl, fontsize=8)
        ax.set_ylabel("Mean activation", fontsize=9)
        ax.set_title(f"Hierarchy {h_i}" if meta.is_multi else "Bottleneck hierarchy", fontsize=10)
        ax.legend(fontsize=7, ncol=4, loc="upper right"); ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Per-node Mean Activation — TinyImageNet", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def fig_hierarchy_gradcam(model, val_loader, device, save_path, n_images=4, max_depth_shown=4):
    _apply_style()
    metas = discover_hierarchies(model)
    if not metas:
        return
    meta = metas[0]
    n_levels = min(meta.n_levels, max_depth_shown)
    acts_dict, images_cpu = collect_node_activations(model, val_loader, [meta], device, max_samples=500)
    act_mat = acts_dict[meta.label]
    imgs_show = images_cpu[:n_images]
    fig, axes = plt.subplots(n_images, n_levels, figsize=(n_levels * 2, n_images * 2))
    if n_images == 1:
        axes = axes[np.newaxis, :]
    for img_i, img_t in enumerate(imgs_show):
        acts_row = act_mat[img_i]
        path = constrained_path_for_image(acts_row, meta)
        offs = _depth_offsets(meta.layer_channels)
        path_nodes = [offs[d] + path[d] for d in range(n_levels)]
        img_np = _tensor_to_uint8(img_t)
        model.eval()
        cams = _gradcam_for_image_node_subset(model, img_t.unsqueeze(0), meta, path_nodes, device)
        for d in range(n_levels):
            ax = axes[img_i, d]
            ax.imshow(_overlay(img_np, cams[d], alpha=0.45)); ax.axis("off")
            if img_i == 0:
                ax.set_title(f"Depth {d+1}\nNode {path[d]}", fontsize=7)
    fig.suptitle("GradCAM — Activation Path Nodes (L8 Taxon — ImageNet)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./data/imagenet")
    p.add_argument("--out-dir", default="paper_graphics_imagenet")
    p.add_argument("--max-images", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--n-probe-train", type=int, default=5000,
                   help="Max training images used for linear probe")
    p.add_argument("--skip-hierarchy", action="store_true")
    p.add_argument("--force-recompute", action="store_true")
    p.add_argument("--n-gallery", type=int, default=6)
    p.add_argument("--n-hier-images", type=int, default=4)
    p.add_argument("--skip-case-studies", action="store_true",
                   help="Skip t-SNE and neuron-steering case studies")
    p.add_argument("--n-tsne-samples", type=int, default=3000)
    p.add_argument("--steer-scale", type=float, default=10.0)
    p.add_argument("--n-steer-classes", type=int, default=6,
                   help="Number of classes to steer per model")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    main_dir = out_dir / "main"
    ablation_dir = out_dir / "ablation"
    taxonomy_dir = out_dir / "taxonomy"
    case_study_dir = out_dir / "case_studies"
    cache_dir = out_dir / "cache"
    for d in (main_dir, ablation_dir, taxonomy_dir, case_study_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading model configs ===")
    main_runs = [r for p in MAIN_CONFIG_RELPATHS
                 if (r := build_run_dict(ROOT / p)) is not None]
    abl_taxon_runs = [r for p in ABLATION_TAXON_RELPATHS
                      if (r := build_run_dict(ROOT / p)) is not None]
    abl_multi_runs = [r for p in ABLATION_MULTI_RELPATHS
                      if (r := build_run_dict(ROOT / p)) is not None]

    print(f"Main models: {[r['short'] for r in main_runs]}")
    print(f"Ablation (taxon):  {[r['short'] for r in abl_taxon_runs]}")
    print(f"Ablation (multi):  {[r['short'] for r in abl_multi_runs]}")

    if not main_runs:
        print("ERROR: no main model checkpoints found. Check configs and output dirs.")
        return

    print("\n=== Building data loaders ===")
    train_loader, val_loader = make_imagenet_loaders(
        args.data_root, args.image_size, args.batch_size, args.num_workers, args.max_images)

    # Subsample training loader for probe
    probe_train_loader = None
    if train_loader is not None:
        from torch.utils.data import Subset, DataLoader
        ds = train_loader.dataset
        n = min(args.n_probe_train, len(ds))
        probe_train_loader = DataLoader(
            Subset(ds, list(range(n))),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

    print("\n=== Computing / loading metrics ===")
    def _get_metrics(runs):
        return [compute_all_metrics(
            r, val_loader, device, probe_train_loader,
            args.max_images, cache_dir, args.force_recompute
        ) for r in runs]

    main_metrics = _get_metrics(main_runs)
    abl_taxon_metrics = _get_metrics(abl_taxon_runs)
    abl_multi_metrics = _get_metrics(abl_multi_runs)

    print("\n=== Generating figures ===")
    fig_training_curves(main_runs, main_dir / "fig01_training_curves.png")
    fig_reconstruction_gallery(main_runs, val_loader, device,
                                main_dir / "fig02_reconstruction_gallery.png",
                                n_images=args.n_gallery)
    _scatter_vs_l0(main_runs, main_metrics,
                   y_key="probe_top1_acc", y_label="Linear-probe top-1 accuracy (200 classes)",
                   title="Downstream probing vs. sparsity — TinyImageNet",
                   save_path=main_dir / "fig03_sparsity_probing.png")
    _scatter_vs_l0(main_runs, main_metrics,
                   y_key="knn_k5_mean_acc", y_label="5-NN top-1 accuracy",
                   title="KNN accuracy vs. sparsity — TinyImageNet",
                   save_path=main_dir / "fig04_sparsity_knn.png")
    _scatter_vs_l0(main_runs, main_metrics,
                   y_key="monosemanticity",
                   y_label="Monosemanticity\n(mean max-class lift, top-50 images)",
                   title="Monosemanticity vs. sparsity — TinyImageNet",
                   save_path=main_dir / "fig05_monosemanticity.png")
    fig_metrics_overview(main_runs, main_metrics, main_dir / "fig06_metrics_overview.png")

    if abl_taxon_runs:
        fig_ablation_bar(abl_taxon_runs, abl_taxon_metrics,
                         ablation_dir / "fig07_ablation_depth.png",
                         title="Depth ablation — Bottleneck TopK Taxon (L4/L6/L8/L10) — ImageNet")
    if abl_multi_runs:
        fig_ablation_bar(abl_multi_runs, abl_multi_metrics,
                         ablation_dir / "fig08_ablation_multitaxon.png",
                         title="Multi-taxon sweep (K2/K4/K8, various depths) — ImageNet")

    if not args.skip_hierarchy:
        taxon_runs = [r for r in main_runs
                      if r["type"] in ("bottleneck_topk_taxon", "bottleneck_topk_multi_taxon")]
        for run in taxon_runs:
            run_tax_dir = taxonomy_dir / run["name"]
            run_tax_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  Hierarchy viz for {run['short']} ...")
            try:
                model, _ = _load_model_from_run_dict(run, device)
                model.eval()
            except Exception as e:
                print(f"    [skip] {e}"); continue

            metas = discover_hierarchies(model)
            single_metas = [m for m in metas if not m.is_multi]
            if single_metas:
                fig_hierarchy_prefix_recon(
                    model, val_loader, device,
                    run_tax_dir / "fig09_prefix_recon.png",
                    n_images=args.n_hier_images,
                    run_short=run["short"],
                )
            fig_hierarchy_activations(model, val_loader, device,
                                       run_tax_dir / "fig10_activations.png")
            fig_hierarchy_gradcam(model, val_loader, device,
                                   run_tax_dir / "fig11_gradcam.png",
                                   n_images=args.n_hier_images)
            del model

    # ── Case studies: t-SNE + neuron steering ────────────────────────────────
    if not args.skip_case_studies:
        print("\n=== Case studies ===")
        for run in main_runs:
            run_cs_dir = case_study_dir / run["name"]
            run_cs_dir.mkdir(parents=True, exist_ok=True)

            fig_tsne_imagenet(
                run=run,
                cache_dir=cache_dir,
                val_loader=val_loader,
                device=device,
                save_path=run_cs_dir / "fig12_tsne.png",
                n_classes_shown=9,
                max_tsne_samples=args.n_tsne_samples,
                data_root=args.data_root,
                force=args.force_recompute,
            )

            fig_neuron_steering_imagenet(
                run=run,
                cache_dir=cache_dir,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                save_path=run_cs_dir / "fig13_neuron_steering.png",
                data_root=args.data_root,
                n_classes=args.n_steer_classes,
                steer_scale=args.steer_scale,
                force=args.force_recompute,
            )

    print(f"\n✓ All paper graphics saved to {out_dir}/")


if __name__ == "__main__":
    main()
