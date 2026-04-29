#!/usr/bin/env python3
"""Publication-quality graphics for the TaxonomicWeights paper — CelebA-HQ.

Computes all metrics from scratch (with caching under paper_graphics_celeba_hq/cache/)
and produces clean, informative figures covering:

  Main figures (all main models)
  ─────────────────────────────
  fig01_training_curves.png        Val-recon + dead-frac training curves
  fig02_reconstruction_gallery.png Side-by-side orig / recon per model
  fig03_sparsity_probing.png       Linear-probe mean AUC-ROC vs L0 scatter
  fig04_sparsity_knn.png           5-NN mean accuracy vs L0 scatter
  fig05_monosemanticity.png        Monosemanticity score vs L0 scatter
  fig06_metrics_overview.png       Bar-chart overview (MSE, PSNR, L0, dead, probe)

  Ablation figures (taxon-only depth / multi-taxon sweep)
  ────────────────────────────────────────────────────────
  fig07_ablation_depth.png         L4 / L6 / L8 / L10 metric sweep
  fig08_ablation_multitaxon.png    K2 / K4 / K6 / K8 / K12 / K16 sweep

  Taxonomy exploration figures (per taxon model)
  ──────────────────────────────────────────────
  taxonomy/<label>/fig09_prefix_recon.png     Prefix reconstruction per depth
  taxonomy/<label>/fig10_activations.png      Per-depth node activation bars
  taxonomy/<label>/fig11_gradcam.png          GradCAM overlaid on top images

Usage
─────
  python src/paper/paper_graphics_celeba_hq.py \\
      [--data-root ./data/celeba_hq] \\
      [--celeba-attrs-root ./data/celeba_fallback] \\
      [--out-dir paper_graphics_celeba_hq] \\
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
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── reuse existing helpers ────────────────────────────────────────────────────
from src.compare.compare_celeba_hq import (   # noqa: E402
    load_model as _load_model_from_run_dict,
    _model_type,
    _short_name,
    model_colour,
    _to_display,
    load_bottleneck_topk_taxon_model,
    load_bottleneck_topk_multi_taxon_model,
    load_conv_topk_sae_model,
    load_conv_bottleneck_topk_taxon_model,
    load_sae_model,
    load_baseline_model,
)
from src.analyze._taxon_hierarchy_viz_lib import (  # noqa: E402
    discover_hierarchies,
    collect_node_activations,
    constrained_path_for_image,
    path_prefix_reconstructions,
    _gradcam_for_image_node_subset,
    _tensor_to_uint8,
    _overlay,
    _wrap_in_cell,
    _node_iter,
    _depth_offsets,
)
from src.utils.dataloader import CelebAHQLoader  # noqa: E402
from src.paper.paper_analysis_celeba import (  # noqa: E402
    _extract_probe_features,
    run_linear_probe,
    run_knn,
)
from torchvision import datasets, transforms  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

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

# Config paths relative to ROOT — exactly the models specified for the paper.
MAIN_CONFIG_RELPATHS: List[str] = [
    "configs/celeba_hq/baseline_ae_celeba_hq.json",
    "configs/celeba_hq/sae_celeba_hq.json",
    "configs/celeba_hq/jumprelu_sae_celeba_hq.json",
    "configs/celeba_hq/topk_sae_celeba_hq.json",
    "configs/celeba_hq/main/topk_sae/topk_sae_celeba_hq_r18_k7_ch254_3layer.json",
    "configs/celeba_hq/matryoshka_batch_topk_sae_celeba_hq.json",
    "configs/celeba_hq/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_celeba_hq_r18_v1_main_L8.json",
    "configs/celeba_hq/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_celeba_hq_r18_v1_main_L8_dkl1e2.json",
    "configs/celeba_hq/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_celeba_hq_r18_v1_main_3layer_L7.json",
    "configs/celeba_hq/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_celeba_hq_r18_v1_main_3layer_L7_dkl.json",
    "configs/celeba_hq/main/bottleneck_jumprelu_taxon/bottleneck_jumprelu_taxon_ae_celeba_hq_r18_v1_main_L8.json",
    "configs/celeba_hq/main/bottleneck_taxon/bottleneck_taxon_ae_celeba_hq_r18_v1_main_dkl1e-02_temp0p001.json",
    "configs/celeba_hq/main/bottleneck_taxon/bottleneck_taxon_ae_celeba_hq_r18_v1_main_dkl1e-02_temp0p01.json",
    "configs/celeba_hq/main/bottleneck_taxon/bottleneck_taxon_ae_celeba_hq_r18_v1_main_dkl1e-02_temp0p1.json",
    "configs/celeba_hq/main/bottleneck_taxon/bottleneck_taxon_ae_celeba_hq_r18_v1_main_dkl1e-02_temp0p1_hard.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K4_L6_gate4.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K4_L8_gate1.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K6_L8_gate1.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K8_L8_gate1.json",
    "configs/celeba_hq/main/conv_topk_sae/conv_topk_sae_ae_celeba_hq_k9.json",
    "configs/celeba_hq/main/conv_bottleneck_topk_taxon/conv_bottleneck_topk_taxon_ae_celeba_hq_L9.json",
]

ABLATION_TAXON_RELPATHS: List[str] = [
    "configs/celeba_hq/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_celeba_hq_r18_v1_main_L4.json",
    "configs/celeba_hq/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_celeba_hq_r18_v1_main_L6.json",
    "configs/celeba_hq/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_celeba_hq_r18_v1_main_L8.json",
    "configs/celeba_hq/main/bottleneck_topk_taxon/bottleneck_topk_taxon_ae_celeba_hq_r18_v1_main_L10.json",
]

ABLATION_MULTI_RELPATHS: List[str] = [
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K2_L6_gate1.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K4_L6_gate1.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K4_L8_gate1.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K8_L8_gate1.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K12_L4_gate1.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K12_L6_gate1.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K16_L4_gate1.json",
    "configs/celeba_hq/main/bottleneck_topk_multi_taxon/bottleneck_topk_multi_taxon_ae_celeba_hq_r18_v6_main_K16_L6_gate1.json",
]

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (consistent with compare scripts)
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = {
    "baseline":                   "#2ca02c",
    "sae":                        "#ff7f0e",
    "topk_sae":                   "#d62728",
    "gated_sae":                  "#17becf",
    "jumprelu_sae":               "#8c564b",
    "matryoshka_batch_topk_sae":  "#1a9850",
    "bottleneck_topk_taxon":      "#005f73",
    "bottleneck_jumprelu_taxon":  "#9b59b6",
    "bottleneck_topk_multi_taxon":"#000080",
    "bottleneck_taxon":           "#003f5c",
    "bottleneck_multi_taxon":     "#5e2d79",
    "conv_topk_sae":              "#e6550d",
    "conv_bottleneck_topk_taxon": "#084081",
}

_MULTI_TAXON_COLOURS = [
    "#000080", "#1f3a93", "#3b5998", "#5b7bbf", "#082567", "#22577a", "#0d3b77", "#2c4a8c",
]

def _colour(mtype: str, idx_within_type: int = 0) -> str:
    if mtype in _PALETTE:
        return _PALETTE[mtype]
    return "#999999"

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint resolution
# ─────────────────────────────────────────────────────────────────────────────

def _training_suffix_for_config(cfg: dict) -> str:
    """Return the run-directory suffix appended by the training script."""
    mc = cfg.get("model", {})
    tc = cfg.get("training", {})
    exp = cfg.get("experiment_name", "")
    variant = mc.get("model_variant", None)

    # ── conv topk sae ─────────────────────────────────────────────────────────
    if exp.startswith("conv_topk_sae_"):
        k = int(mc.get("topk_k", 64))
        return f"_k{k}"
    # ── conv bottleneck topk taxon ────────────────────────────────────────────
    if exp.startswith("conv_bottleneck_topk_taxon_"):
        L = int(mc.get("bottleneck_n_taxonomy_layers", 9))
        return f"_L{L}"
    # ── bottleneck topk taxon ─────────────────────────────────────────────────
    if exp.startswith("bottleneck_topk_taxon_"):
        L = int(mc.get("bottleneck_n_taxonomy_layers", 6))
        return f"_L{L}"
    # ── bottleneck jumprelu taxon ────────────────────────────────────────────
    if exp.startswith("bottleneck_jumprelu_taxon_"):
        L = int(mc.get("bottleneck_n_taxonomy_layers", 8))
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
        L = int(mc.get("bottleneck_n_taxonomy_layers", 6))
        return f"_L{L}"

    # ── SAE family ────────────────────────────────────────────────────────────
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

    # ── baseline ──────────────────────────────────────────────────────────────
    if "baseline" in exp:
        return ""

    # ── plain taxon family (topk_taxon, topk_multi_taxon, etc.) ─────────────
    auxk = float(tc.get("auxk_weight", 0.0))
    auxk_suffix = f"_auxk_{auxk:.0e}" if auxk else ""
    if "topk_multi_taxon" in exp:
        n_hier = mc.get("n_hierarchies", 3)
        return f"_K{n_hier}{auxk_suffix}"
    if "topk_taxon" in exp:
        return auxk_suffix
    dw = float(tc.get("dkl_weight", 1e-2))
    temperature = float(mc.get("temperature", tc.get("temperature", 1.0)))
    temp_str = f"{temperature:g}".replace(".", "p")
    hard_suffix = "_hard" if mc.get("hard", False) else ""
    ew = float(tc.get("entropy_weight", 0.0))
    ew_suffix = f"_ew_{ew:.0e}" if ew else ""
    return f"_dkl_{dw:.0e}_temp_{temp_str}{hard_suffix}{ew_suffix}"


def resolve_ckpt(cfg_path: Path) -> Optional[Path]:
    """Return path to best.pt for the given config file, or None if not found."""
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
    """Build a run-info dict (like compare_celeba_hq.discover_runs) from a config."""
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
        "name":       exp,
        "short":      short,
        "type":       mtype,
        "path":       run_path,
        "best_ckpt":  ckpt,
        "history":    run_path / "training_history.json",
        "cfg":        cfg,
        "cfg_path":   cfg_path,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def make_celeba_hq_val_loader(data_root: str, image_size: int = 256,
                               batch_size: int = 32, num_workers: int = 4,
                               max_images: int = 5000) -> DataLoader:
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    loader = CelebAHQLoader(
        data_root=data_root, batch_size=batch_size, num_workers=num_workers,
        image_size=image_size, val_split=0.05, transform=tf,
        val_subset=max_images,
    )
    _, val_loader = loader.get_loaders()
    return val_loader if val_loader is not None else loader.train_loader


def make_celeba_attr_loaders(celeba_root: str, image_size: int = 256,
                              batch_size: int = 128, num_workers: int = 4,
                              n_train: int = 4000, n_val: int = 1000
                              ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """Load CelebA attribute-labeled images for linear probing."""
    if not celeba_root or not Path(celeba_root).exists():
        return None, None
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    try:
        ds = datasets.CelebA(root=celeba_root, split="all", target_type="attr",
                              transform=tf, download=False)
    except Exception as e:
        print(f"  [probe] CelebA attrs not available at {celeba_root}: {e}")
        return None, None
    n_train = min(n_train, len(ds) // 2)
    n_val = min(n_val, len(ds) - n_train)
    make = lambda idx, sh: DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=sh,
                                       num_workers=num_workers, pin_memory=True)
    return make(list(range(n_train)), True), make(list(range(n_train, n_train + n_val)), False)


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _encode(model, imgs: torch.Tensor) -> torch.Tensor:
    z, *_ = model.encode(imgs)
    return z.detach().cpu().flatten(start_dim=1)   # [B, D]


@torch.no_grad()
def collect_latents_and_recon(
    model, loader, device: torch.device, max_images: int = 0
) -> Dict:
    """Run val loader; return latents, raw images, MSE, MAE."""
    Zs, imgs_out, mses, maes = [], [], [], []
    n_seen = 0
    for batch in tqdm(loader, desc="    collect latents", leave=False):
        if max_images > 0 and n_seen >= max_images:
            break
        imgs = batch[0].to(device)
        take = min(imgs.shape[0], max_images - n_seen) if max_images > 0 else imgs.shape[0]
        imgs = imgs[:take]
        z = _encode(model, imgs)
        recon_out = model.decode(z.to(device).reshape(imgs.shape[0], -1, 1, 1) if z.ndim == 2 else z.to(device))
        if isinstance(recon_out, (tuple, list)):
            recon_out = recon_out[0]
        Zs.append(z.numpy())
        imgs_out.append(imgs.cpu().numpy())
        mses.extend(((imgs - recon_out) ** 2).mean(dim=(1, 2, 3)).cpu().numpy().tolist())
        maes.extend(torch.abs(imgs - recon_out).mean(dim=(1, 2, 3)).cpu().numpy().tolist())
        n_seen += take
    Z = np.concatenate(Zs)        # [N, D]
    imgs_np = np.concatenate(imgs_out)    # [N, C, H, W]
    mse_arr = np.array(mses)
    return dict(Z=Z, imgs=imgs_np, mse=mse_arr, mae=np.array(maes))


@torch.no_grad()
def collect_latents_and_recon_safe(
    model, run: Dict, loader, device: torch.device, max_images: int = 0
) -> Dict:
    """Model-type-aware latent collection that handles diverse forward APIs."""
    Zs, imgs_out, mses, maes = [], [], [], []
    n_seen = 0
    mtype = run["type"]
    for batch in tqdm(loader, desc=f"    {run['short'][:25]}", leave=False):
        if max_images > 0 and n_seen >= max_images:
            break
        imgs = batch[0].to(device)
        take = min(imgs.shape[0], max_images - n_seen) if max_images > 0 else imgs.shape[0]
        imgs = imgs[:take]

        # Forward: get z and recon
        try:
            if mtype == "conv_bottleneck_topk_taxon":
                recon_out, _, _ = model(imgs)
            elif mtype == "conv_topk_sae":
                recon_out, _, _ = model(imgs)
            elif mtype in ("bottleneck_topk_taxon", "bottleneck_topk_multi_taxon"):
                recon_out, _ = model(imgs)
            elif mtype in ("bottleneck_taxon",):
                recon_out, _, _ = model(imgs)
            elif mtype in ("bottleneck_multi_taxon",):
                recon_out, _, _, _, _ = model(imgs)
            elif mtype == "matryoshka_batch_topk_sae":
                recon_list, _ = model(imgs)
                recon_out = recon_list[0] if isinstance(recon_list, (list, tuple)) else recon_list
            elif mtype in ("topk_sae", "jumprelu_sae", "gated_sae"):
                recon_out, _ = model(imgs)
            elif mtype == "sae":
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
        imgs_out.append(imgs.cpu().numpy())
        mses.extend(((imgs - recon_out) ** 2).mean(dim=(1, 2, 3)).cpu().numpy().tolist())
        maes.extend(torch.abs(imgs - recon_out).mean(dim=(1, 2, 3)).cpu().numpy().tolist())
        n_seen += take

    if not Zs:
        return {}
    Z = np.concatenate(Zs)
    imgs_np = np.concatenate(imgs_out)
    return dict(Z=Z, imgs=imgs_np, mse=np.array(mses), mae=np.array(maes))


def _psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 0:
        return float("inf")
    return 10 * math.log10(max_val ** 2 / mse)


def compute_sparsity_metrics(Z: np.ndarray, threshold: float = 1e-6,
                             model: Optional[object] = None) -> Dict:
    """Compute L0 and dead-frac sparsity metrics.

    Dead-frac alignment: if the model has a ``_steps_since_active`` buffer
    (TopK taxon / TopK SAE family), use it directly — this matches the training
    script which counts consecutive non-selection steps. Falls back to the
    magnitude-threshold approach for models without such a buffer (plain SAE,
    baseline, etc.).
    """
    import torch as _torch
    l0_per = (np.abs(Z) > threshold).sum(axis=1).astype(float)
    # Prefer model-internal dead tracker when available
    dead_frac_val: Optional[float] = None
    if model is not None:
        # Collect all _steps_since_active buffers in the model
        for name, buf in model.named_buffers():
            if "steps_since_active" in name:
                dead_steps = None
                # Find matching dead_steps attribute on the owning module
                # Walk the module hierarchy by prefix
                parts = name.split(".")
                owner = model
                for part in parts[:-1]:
                    owner = getattr(owner, part, owner)
                dead_steps = getattr(owner, "dead_steps", None)
                if dead_steps is None:
                    # Try one level up (bottleneck_stage.dead_steps)
                    dead_steps = getattr(model, "dead_steps", None)
                if dead_steps is not None:
                    dead_frac_val = float((buf >= dead_steps).float().mean().item())
                break  # use first found buffer
    if dead_frac_val is None:
        # Magnitude fallback: a feature is dead if it never activates above threshold
        max_act = np.abs(Z).max(axis=0)
        dead_frac_val = float((max_act < threshold).mean())
    return dict(
        l0_mean=float(l0_per.mean()),
        l0_median=float(np.median(l0_per)),
        dead_frac=dead_frac_val,
        n_features=int(Z.shape[1]),
    )


def compute_monosemanticity_celeba(
    Z: np.ndarray, attrs: np.ndarray,
    top_k: int = 50, min_active_frac: float = 0.005,
) -> float:
    """Mean max-attribute lift over active features.

    For each active feature collect the top-K activating images, then compute
    lift_j = freq(attr_j=1 | top-K) / (base_freq(attr_j) + eps) for every
    CelebA attribute j, and record the max lift.
    The model score is the mean max lift over qualifying features.

    A random feature gives lift ≈ 1; a perfectly attribute-selective feature
    gives lift ≈ 1/base_freq (up to ~40 for a rare attribute).

    attrs: [N, 40] binary integer array
    Z:     [N, D]  float latent matrix
    """
    N, D = Z.shape
    k = min(top_k, N)
    base_freq = attrs.astype(np.float32).mean(axis=0) + 1e-6  # [40]
    thresh = float(np.quantile(Z[Z > 0], min_active_frac)) if (Z > 0).any() else 0.0
    active_mask = Z.max(axis=0) > thresh
    scores: List[float] = []
    for c in range(D):
        if not active_mask[c]:
            continue
        top_idx = np.argpartition(Z[:, c], -k)[-k:]
        freq_top = attrs[top_idx].astype(np.float32).mean(axis=0)  # [40]
        lift = freq_top / base_freq
        scores.append(float(lift.max()))
    return float(np.mean(scores)) if scores else float("nan")


# extract_probe_features, run_linear_probe, run_knn are imported from paper_analysis_celeba


def compute_all_metrics(
    run: Dict, val_loader: DataLoader, device: torch.device,
    probe_tr_loader: Optional[DataLoader], probe_va_loader: Optional[DataLoader],
    max_images: int, cache_dir: Path, force: bool,
) -> Dict:
    """Compute (or load from cache) all scalar metrics for one run."""
    cache_file = cache_dir / run["name"] / "metrics.json"
    latents_file = cache_dir / run["name"] / "latents.npz"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and latents_file.exists() and not force:
        with open(cache_file) as f:
            return json.load(f)

    print(f"\n  Computing metrics for {run['short']} ...")
    model, _ = _load_model_from_run_dict(run, device)
    model.eval()

    # Reconstruction + latents
    data = collect_latents_and_recon_safe(model, run, val_loader, device, max_images)
    if not data:
        return {}
    Z = data["Z"]
    mse_mean = float(data["mse"].mean())
    psnr_mean = _psnr(mse_mean, max_val=2.0)   # images in [-1,1], range=2

    sp = compute_sparsity_metrics(Z, model=model)

    metrics: Dict = dict(
        mse_mean=mse_mean,
        mae_mean=float(data["mae"].mean()),
        psnr_mean=psnr_mean,
        **sp,
    )

    # Cache latents
    np.savez_compressed(str(latents_file), Z=Z, imgs=data["imgs"])

    # Linear probe + KNN — delegate to canonical paper_analysis_celeba implementations
    # (300 epochs, batch=512, GPU standardisation; vectorised GPU KNN)
    if probe_tr_loader is not None and probe_va_loader is not None:
        try:
            Z_tr, Y_tr = _extract_probe_features(model, probe_tr_loader, device)
            Z_va, Y_va = _extract_probe_features(model, probe_va_loader, device)
            probe_metrics = run_linear_probe(Z_tr, Y_tr, Z_va, Y_va, device=device)
            knn_metrics   = run_knn(Z_tr, Y_tr, Z_va, Y_va, ks=(1, 5), device=device)
            metrics.update(probe_metrics)
            metrics.update(knn_metrics)

            mono = compute_monosemanticity_celeba(Z_va, Y_va)
            metrics["monosemanticity"] = float(mono)
        except Exception as e:
            print(f"    [warn] probe failed: {e}")

    with open(cache_file, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
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

_MT_IDX: Dict[str, int] = {}   # per-type counter for colour index


def _run_colour(run: Dict) -> str:
    mtype = run["type"]
    idx = _MT_IDX.get(mtype, 0)
    return model_colour(run["name"], mtype, idx)


def _assign_colours(runs: List[Dict]) -> List[str]:
    """Return a colour per run, using the compare-script palette."""
    counts: Dict[str, int] = {}
    colours = []
    for r in runs:
        idx = counts.get(r["type"], 0)
        colours.append(model_colour(r["name"], r["type"], idx))
        counts[r["type"]] = idx + 1
    return colours


def _nice_short(run: Dict) -> str:
    """Compact label for plots."""
    return run["short"]


# ─────────────────────────────────────────────────────────────────────────────
# Main figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_training_curves(runs: List[Dict], save_path: Path) -> None:
    """Validation reconstruction loss curves across all main models."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = _assign_colours(runs)

    ax_recon, ax_dead = axes
    n_with_history = 0
    for run, col in zip(runs, colours):
        hist_path = run.get("history")
        if hist_path is None or not Path(hist_path).exists():
            continue
        try:
            with open(hist_path) as f:
                h = json.load(f)
        except Exception:
            continue
        # Reconstruction
        recon_key = next((k for k in ("val_recon", "val_recon_k0", "val_loss") if k in h), None)
        if recon_key is None:
            continue
        epochs = h.get("epochs", list(range(1, len(h[recon_key]) + 1)))
        ys = np.array(h[recon_key], dtype=float)
        ax_recon.plot(epochs, ys, label=run["short"], color=col, linewidth=1.5)
        # Dead fraction
        if "val_dead" in h:
            yd = np.array(h["val_dead"], dtype=float)
            ax_dead.plot(epochs, yd, label=run["short"], color=col, linewidth=1.5)
        n_with_history += 1

    ax_recon.set_title("Val reconstruction loss", fontsize=12)
    ax_recon.set_xlabel("Epoch"); ax_recon.set_ylabel("Recon loss")
    ax_recon.grid(alpha=0.25)
    ax_recon.legend(ncol=2, loc="upper right", framealpha=0.8)

    ax_dead.set_title("Val dead-feature fraction", fontsize=12)
    ax_dead.set_xlabel("Epoch"); ax_dead.set_ylabel("Dead frac")
    ax_dead.grid(alpha=0.25)
    ax_dead.legend(ncol=2, loc="upper right", framealpha=0.8)

    fig.suptitle("Training dynamics — CelebA-HQ", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def _scatter_vs_l0(runs: List[Dict], metrics_list: List[Dict],
                   y_key: str, y_label: str, title: str, save_path: Path,
                   higher_is_better: bool = True) -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colours = _assign_colours(runs)

    for run, m, col in zip(runs, metrics_list, colours):
        l0 = m.get("l0_mean", float("nan"))
        y = m.get(y_key, float("nan"))
        if math.isnan(l0) or math.isnan(y):
            continue
        ax.scatter(l0, y, color=col, s=100, zorder=5, edgecolors="white", linewidths=0.8)
        ax.annotate(run["short"], (l0, y),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color=col, ha="left")

    ax.set_xlabel("Mean L0 (active features per image)", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.25)

    # Legend patches
    import matplotlib.patches as mpatches
    seen_types: Dict[str, str] = {}
    for run, col in zip(runs, colours):
        if run["type"] not in seen_types:
            seen_types[run["type"]] = col
    handles = [mpatches.Patch(color=c, label=t.replace("_", " ")) for t, c in seen_types.items()]
    ax.legend(handles=handles, loc="best", framealpha=0.8, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def fig_metrics_overview(runs: List[Dict], metrics_list: List[Dict], save_path: Path) -> None:
    """Bar chart overview of all key metrics."""
    _apply_style()
    metric_keys = [
        ("mse_mean",         "MSE ↓",         False),
        ("psnr_mean",        "PSNR ↑ (dB)",   True),
        ("l0_mean",          "Mean L0 ↓",     False),
        ("dead_frac",        "Dead frac ↑=worse", False),
        ("probe_mean_auc",   "Probe AUC ↑",   True),
        ("knn_k5_mean_acc",  "5-NN Acc ↑",    True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.flatten()
    colours = _assign_colours(runs)
    labels = [r["short"] for r in runs]
    x = np.arange(len(labels))

    for ax, (key, label, hib) in zip(axes, metric_keys):
        vals = [m.get(key, float("nan")) for m in metrics_list]
        bars = ax.bar(x, vals, color=colours, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        for bar, v in zip(bars, vals):
            if not math.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    fig.suptitle("Metrics Overview — CelebA-HQ (main models)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


@torch.no_grad()
def fig_reconstruction_gallery(runs: List[Dict], val_loader: DataLoader,
                                device: torch.device, save_path: Path,
                                n_images: int = 6) -> None:
    """Side-by-side original + reconstruction for each model."""
    _apply_style()
    # Grab a fixed set of images
    batch = next(iter(val_loader))
    imgs = batch[0][:n_images].to(device)

    n_models = len(runs)
    n_rows = n_models + 1   # +1 for originals
    fig, axes = plt.subplots(n_rows, n_images, figsize=(n_images * 1.8, n_rows * 1.8))

    def _show(ax, img_t):
        np_img = (img_t.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        ax.imshow(np_img); ax.axis("off")

    def _row_label(ax, text, colour="#333333"):
        """Place a left-aligned row title that is never clipped by tight_layout."""
        ax.text(-0.04, 0.5, text, transform=ax.transAxes,
                ha="right", va="center", fontsize=7.5,
                color=colour, clip_on=False)

    # Row 0: originals
    for j in range(n_images):
        _show(axes[0, j], imgs[j])
    _row_label(axes[0, 0], "Original")

    for i, run in enumerate(runs):
        try:
            model, _ = _load_model_from_run_dict(run, device)
            model.eval()
        except Exception as e:
            print(f"  [warn] could not load {run['short']}: {e}")
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
            print(f"  [warn] forward error {run['short']}: {e}")
            continue

        row_idx = i + 1
        for j in range(n_images):
            _show(axes[row_idx, j], recon[j])
        _row_label(axes[row_idx, 0], run["short"],
                   colour=model_colour(run["name"], run["type"], i))
        del model

    fig.suptitle("Reconstruction Gallery — CelebA-HQ", fontsize=13, y=1.0)
    plt.subplots_adjust(hspace=0.05, wspace=0.05, left=0.14)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation_bar(runs: List[Dict], metrics_list: List[Dict], save_path: Path,
                     title: str, groupby_key: str = "short") -> None:
    """Multi-metric bar comparison for ablation sweeps."""
    _apply_style()
    metric_keys = [
        ("mse_mean",       "MSE ↓"),
        ("psnr_mean",      "PSNR ↑"),
        ("l0_mean",        "Mean L0"),
        ("dead_frac",      "Dead frac"),
        ("probe_mean_auc", "Probe AUC ↑"),
        ("knn_k5_mean_acc","5-NN Acc ↑"),
        ("monosemanticity","Monosemanticity ↑"),
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

def _run_tsne(Z: np.ndarray, n_components: int = 2, perplexity: float = 40,
              max_samples: int = 3000, seed: int = 42) -> np.ndarray:
    """Return t-SNE embedding [N, 2] (or [N, 3]) of Z."""
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    N = min(len(Z), max_samples)
    idx = rng.choice(len(Z), N, replace=False)
    Z_sub = StandardScaler().fit_transform(Z[idx])
    emb = TSNE(n_components=n_components, perplexity=perplexity,
               init="pca", random_state=seed, n_jobs=-1).fit_transform(Z_sub)
    return emb, idx


def fig_tsne_celeba(
    run: Dict, cache_dir: Path, probe_va_loader: Optional[DataLoader],
    device: torch.device, save_path: Path,
    n_attrs_shown: int = 6, max_tsne_samples: int = 3000, force: bool = False,
) -> None:
    """t-SNE of latent space coloured by the most-discriminating CelebA attributes."""
    _apply_style()
    tsne_cache = cache_dir / run["name"] / "tsne.npz"
    tsne_cache.parent.mkdir(parents=True, exist_ok=True)

    # ── Load or compute latents ──────────────────────────────────────────────
    latents_file = cache_dir / run["name"] / "latents.npz"
    if probe_va_loader is None:
        print(f"  [tsne] skipping {run['short']} — no attr loader")
        return

    if tsne_cache.exists() and not force:
        d = np.load(tsne_cache, allow_pickle=True)
        emb, idx, attrs = d["emb"], d["idx"], d["attrs"]
    else:
        print(f"  [tsne] encoding probe-val set for {run['short']} ...")
        model, _ = _load_model_from_run_dict(run, device)
        model.eval()
        Z_va, Y_va = _extract_probe_features(model, probe_va_loader, device)
        del model
        emb, idx = _run_tsne(Z_va, max_samples=max_tsne_samples)
        attrs = Y_va[idx]        # [N, 40]
        np.savez_compressed(str(tsne_cache), emb=emb, idx=idx, attrs=attrs)

    N = len(emb)
    # Pick n_attrs_shown attributes with highest variance (most discriminative)
    attr_var = attrs.var(axis=0)
    top_attr_idx = np.argsort(-attr_var)[:n_attrs_shown]

    ncols = 3
    nrows = math.ceil(n_attrs_shown / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4.5, nrows * 4.0))
    axes_flat = np.array(axes).reshape(-1)

    for plot_i, a_idx in enumerate(top_attr_idx):
        ax = axes_flat[plot_i]
        y = attrs[:, a_idx]           # binary [N]
        colours_pts = np.where(y == 1, "#e05252", "#aaaaaa")
        ax.scatter(emb[:, 0], emb[:, 1], c=colours_pts, s=3, alpha=0.6,
                   linewidths=0, rasterized=True)
        attr_name = CELEBA_ATTR_NAMES[a_idx].replace("_", " ")
        frac = y.mean()
        ax.set_title(f"{attr_name}\n(pos: {frac:.1%})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    for ax in axes_flat[n_attrs_shown:]:
        ax.axis("off")

    import matplotlib.patches as mpatches
    legend = [
        mpatches.Patch(color="#e05252", label="positive"),
        mpatches.Patch(color="#aaaaaa", label="negative"),
    ]
    fig.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.9)
    fig.suptitle(f"t-SNE latent space — {run['short']} — CelebA-HQ",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Case study: neuron steering
# ─────────────────────────────────────────────────────────────────────────────

def _find_selective_neurons_celeba(
    Z: np.ndarray, attrs: np.ndarray,
    n_per_attr: int = 3, min_active: float = 1e-6,
) -> Dict[int, List[int]]:
    """For each attribute return the top-n_per_attr neurons most selective for it.

    Selectivity = mean activation for positives / (mean activation overall + ε).
    Returns {attr_idx: [neuron_idx, ...]}, only for attrs with enough positives.
    """
    result: Dict[int, List[int]] = {}
    global_mean = Z.mean(axis=0) + 1e-8          # [D]
    live = (np.abs(Z).max(axis=0) >= min_active)  # [D]
    for a in range(attrs.shape[1]):
        pos = attrs[:, a] == 1
        if pos.sum() < 20:
            continue
        mean_pos = Z[pos].mean(axis=0)            # [D]
        selectivity = mean_pos / global_mean       # [D]
        selectivity[~live] = 0.0
        top = np.argsort(-selectivity)[:n_per_attr]
        result[a] = top.tolist()
    return result


def fig_neuron_steering_celeba(
    run: Dict, cache_dir: Path,
    probe_tr_loader: Optional[DataLoader], probe_va_loader: Optional[DataLoader],
    val_loader: DataLoader, device: torch.device, save_path: Path,
    n_attrs: int = 4, n_images: int = 5, steer_scale: float = 10.0,
    force: bool = False,
) -> None:
    """Find attribute-selective neurons; dial them up on out-of-class images.

    Layout: one row per selected attribute.
    Columns: [original image] [recon baseline] [steered ×1] [steered ×3] [steered ×10]
    The attribute label and neuron index are annotated on each row.
    """
    _apply_style()
    if probe_tr_loader is None or probe_va_loader is None:
        print(f"  [steer] skipping {run['short']} — no attr loader")
        return

    steer_cache = cache_dir / run["name"] / "steer_neurons.npz"
    steer_cache.parent.mkdir(parents=True, exist_ok=True)

    # ── Get Z + attrs for selectivity computation ────────────────────────────
    if steer_cache.exists() and not force:
        sc = np.load(steer_cache, allow_pickle=True)
        sel = {int(k): list(v) for k, v in sc["sel"].item().items()}
    else:
        print(f"  [steer] computing selectivity for {run['short']} ...")
        model, _ = _load_model_from_run_dict(run, device)
        model.eval()
        # Use train split for selectivity (more samples, same distribution)
        Z_tr, Y_tr = _extract_probe_features(model, probe_tr_loader, device)
        sel = _find_selective_neurons_celeba(Z_tr, Y_tr, n_per_attr=1)
        del model
        np.savez_compressed(str(steer_cache), sel=np.array(sel, dtype=object))

    if not sel:
        print(f"  [steer] no selective neurons found for {run['short']}")
        return

    # Pick n_attrs attrs with most selectivity (deterministic: take first n in sorted order)
    chosen_attrs = sorted(sel.keys())[:n_attrs]
    scales = [0.0, 1.0, 3.0, steer_scale]   # 0 = unsteered recon
    n_scales = len(scales)

    # Collect a large pool so each row can use a different face
    all_imgs, all_attrs_batch = [], []
    for imgs, attrs_b in probe_va_loader:
        all_imgs.append(imgs)
        all_attrs_batch.append(attrs_b.numpy())
        if sum(b.shape[0] for b in all_imgs) >= 400:
            break
    all_imgs_t = torch.cat(all_imgs, 0)
    all_attrs_np = np.concatenate(all_attrs_batch, 0)

    model, _ = _load_model_from_run_dict(run, device)
    model.eval()

    fig, axes = plt.subplots(
        n_attrs, 2 + n_scales,
        figsize=((2 + n_scales) * 2.5, n_attrs * 2.8),
    )
    if n_attrs == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["Original", "Recon"] + [f"steer ×{s:.0f}" for s in scales]
    for col_j, lbl in enumerate(col_labels):
        axes[0, col_j].set_title(lbl, fontsize=8)

    def _show(ax, img_t):
        np_img = (img_t.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)
        ax.imshow(np_img); ax.axis("off")

    for row_i, a_idx in enumerate(chosen_attrs):
        flat_idx = sel[a_idx][0] if isinstance(sel[a_idx], list) else int(sel[a_idx])
        attr_name = CELEBA_ATTR_NAMES[a_idx].replace("_", " ")

        # Find images that are *negative* for this attribute.
        # Use row_i as an offset so each row displays a clearly different face.
        neg_mask = all_attrs_np[:, a_idx] == 0
        neg_indices = np.where(neg_mask)[0]
        if len(neg_indices) == 0:
            continue
        # Rotate through available negatives per row so faces differ across rows
        offset = row_i % max(len(neg_indices) - n_images + 1, 1)
        face_indices = neg_indices[offset : offset + n_images]
        # Pad if not enough negatives remain
        if len(face_indices) < n_images:
            face_indices = neg_indices[:n_images]
        imgs_neg = all_imgs_t[face_indices].to(device)   # [n_images, C, H, W]

        # Encode
        with torch.no_grad():
            z_raw, *_ = model.encode(imgs_neg)   # [B, C, H, W] or [B, D]

        z_shape = z_raw.shape

        # Selectivity is computed on the FLATTENED latent (see _extract_probe_features
        # in paper_analysis_celeba.py).  For conv-shaped latents [B, C, H, W] this
        # means flat_idx points to a single (channel, h, w) position.  Editing
        # one scalar in a ~130k-element latent has no visible effect on the
        # decoded image, so we map back to the *channel* and overwrite the
        # entire spatial map of that channel — this is what matches the intent
        # of "steering neuron k".
        if z_raw.dim() == 4:
            B, C, H, W = z_shape
            channel_idx = int(flat_idx) // (H * W)
            # ref_val = 95th-percentile absolute activation of this channel
            ch_acts = z_raw[:, channel_idx].abs().float().flatten()
            p95 = float(torch.quantile(ch_acts, 0.95).item())
            ref_val = max(p95, 1e-3)
            label_idx = channel_idx
        else:
            z_flat = z_raw.flatten(start_dim=1)
            channel_idx = None
            p95 = float(torch.quantile(z_flat[:, int(flat_idx)].abs().float(), 0.95).item())
            ref_val = max(p95, 1e-3)
            label_idx = int(flat_idx)

        # Show original image with attribute label annotated inside the cell
        ax_orig = axes[row_i, 0]
        _show(ax_orig, imgs_neg[0].cpu())
        ax_orig.text(
            0.03, 0.97, f"{attr_name}\n(n°{label_idx})",
            transform=ax_orig.transAxes,
            fontsize=7, color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, lw=0),
        )

        # Unsteered baseline reconstruction (sanity check column).
        with torch.no_grad():
            recon_base = model.decode(z_raw)
            if isinstance(recon_base, (tuple, list)):
                recon_base = recon_base[0]
        _show(axes[row_i, 1], recon_base[0].cpu())

        for col_j, scale in enumerate(scales):
            z_edit = z_raw.clone()
            if channel_idx is not None:
                # Overwrite the entire spatial map of the chosen channel.
                z_edit[:, channel_idx, :, :] = scale * ref_val
            else:
                z_edit_flat = z_edit.flatten(start_dim=1)
                z_edit_flat[:, int(flat_idx)] = scale * ref_val
                z_edit = z_edit_flat.reshape(z_shape)
            with torch.no_grad():
                recon = model.decode(z_edit.to(device))
                if isinstance(recon, (tuple, list)):
                    recon = recon[0]
            _show(axes[row_i, 2 + col_j], recon[0].cpu())

    del model
    fig.suptitle(f"Neuron Steering — {run['short']} — CelebA-HQ",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy exploration figures
# ─────────────────────────────────────────────────────────────────────────────

def fig_hierarchy_prefix_recon(
    model, val_loader: DataLoader, device: torch.device,
    save_path: Path, n_images: int = 4, run_short: str = ""
) -> None:
    """Prefix reconstruction using model.forward_matryoshka (same as probe analysis).

    Layout: rows = (Original + each depth),  cols = sample images.
    Falls back to path_prefix_reconstructions if forward_matryoshka unavailable.
    """
    _apply_style()

    # ── collect images ────────────────────────────────────────────────────────
    all_imgs: List[torch.Tensor] = []
    for batch in val_loader:
        all_imgs.append(batch[0])
        if sum(b.shape[0] for b in all_imgs) >= n_images:
            break
    if not all_imgs:
        return
    imgs = torch.cat(all_imgs, 0)[:n_images].to(device)   # [N, C, H, W]

    # ── compute prefix reconstructions ───────────────────────────────────────
    if hasattr(model, "forward_matryoshka"):
        # Native model method — correct for all gating / normalisation
        model.eval()
        with torch.no_grad():
            try:
                prefix_recons, _ = model.forward_matryoshka(imgs)
            except Exception as e:
                print(f"  [prefix_recon] forward_matryoshka failed: {e}")
                return
        # prefix_recons: list of [N,C,H,W] tensors, one per depth
        rows = [imgs.cpu()] + [r.detach().cpu() for r in prefix_recons]
        n_levels = len(prefix_recons)
    else:
        # Fallback: path_prefix_reconstructions per image
        metas = discover_hierarchies(model)
        if not metas:
            print("  [skip] no hierarchy metas found")
            return
        meta = metas[0]
        n_levels = meta.n_levels
        acts, _ = collect_node_activations(model, val_loader, [meta], device, max_samples=500)
        act_mat = acts[meta.label]
        recon_stacks: List[List[np.ndarray]] = []   # [n_images][n_levels]
        for img_i, img_t in enumerate(imgs):
            acts_row = act_mat[img_i] if img_i < act_mat.shape[0] else np.zeros(meta.n_nodes)
            path = constrained_path_for_image(acts_row, meta)
            recon_stacks.append(
                path_prefix_reconstructions(model, img_t.unsqueeze(0), meta, path, device)
            )
        # convert to [N,C,H,W] tensors per depth for uniform handling below
        prefix_recons_np = []
        for d in range(n_levels):
            depth_imgs = []
            for img_i in range(len(imgs)):
                if d < len(recon_stacks[img_i]):
                    r = recon_stacks[img_i][d]  # HWC uint8
                    t = torch.from_numpy(r).float().permute(2, 0, 1) / 127.5 - 1.0
                else:
                    t = torch.zeros_like(imgs[0].cpu())
                depth_imgs.append(t)
            prefix_recons_np.append(torch.stack(depth_imgs))
        rows = [imgs.cpu()] + prefix_recons_np

    # ── render figure ─────────────────────────────────────────────────────────
    n_rows   = 1 + n_levels
    n_cols   = len(imgs)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 1.7, n_rows * 1.7))
    if n_rows   == 1: axes = axes[np.newaxis, :]
    if n_cols   == 1: axes = axes[:, np.newaxis]

    row_labels = ["Original"] + [f"Prefix d={d+1}" for d in range(n_levels)]
    for row_idx, (row_batch, row_lbl) in enumerate(zip(rows, row_labels)):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            img_np = (row_batch[col_idx].clamp(-1, 1) + 1.0) * 0.5
            ax.imshow(img_np.permute(1, 2, 0).float().numpy(),
                      interpolation="nearest")
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


def fig_flat_activations(
    model, val_loader: DataLoader, device: torch.device, save_path: Path,
    run_short: str = "",
) -> None:
    """Per-dimension mean activation bar chart for any autoencoder.

    For hierarchical taxon models the bars are coloured by depth and depth
    boundaries are marked with dashed vertical lines.  For all other models a
    single colour is used.  The x-axis is always the flat channel / dimension
    index (no depth grouping).
    """
    _apply_style()
    model.eval()

    # ── Try hierarchy-aware channel collection ──────────────────────────────
    try:
        metas = discover_hierarchies(model)
    except (RuntimeError, AttributeError):
        metas = []

    if metas:
        meta = metas[0]
        acts_dict, _ = collect_node_activations(
            model, val_loader, [meta], device, max_samples=2000)
        mean_acts = acts_dict[meta.label].mean(axis=0)   # [n_channels]

        fig, ax = plt.subplots(1, 1, figsize=(14, 3))
        colors_per_level = plt.cm.Blues(np.linspace(0.3, 0.9, meta.n_levels))
        offs = _depth_offsets(meta.layer_channels)
        x_pos = np.arange(len(mean_acts))
        bar_colors = np.empty((len(mean_acts), 4))
        for d in range(meta.n_levels):
            start = offs[d]
            end = start + meta.layer_channels[d]
            bar_colors[start:end] = colors_per_level[d]
        ax.bar(x_pos, mean_acts, color=bar_colors, width=1.0)
        # depth boundary vertical lines
        for d in range(1, meta.n_levels):
            ax.axvline(x=offs[d] - 0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
        # legend patches
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=colors_per_level[d], label=f"Depth {d+1}")
                   for d in range(meta.n_levels)]
        ax.legend(handles=patches, fontsize=7, ncol=min(meta.n_levels, 8), loc="upper right")
        ax.set_xlabel("Channel index", fontsize=9)
    else:
        # Non-hierarchical model: collect mean absolute activation per dim
        all_z: list = []
        n_seen = 0
        max_samples = 2000
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="    flat activations", leave=False):
                if n_seen >= max_samples:
                    break
                imgs = batch[0].to(device)
                take = min(imgs.shape[0], max_samples - n_seen)
                imgs = imgs[:take]
                z, *_ = model.encode(imgs)
                all_z.append(z.detach().cpu().flatten(start_dim=1).float().numpy())
                n_seen += take
        Z = np.concatenate(all_z, axis=0)   # [N, D]
        mean_acts = np.abs(Z).mean(axis=0)   # [D]

        fig, ax = plt.subplots(1, 1, figsize=(14, 3))
        ax.bar(np.arange(len(mean_acts)), mean_acts, color="#2171b5", width=1.0)
        ax.set_xlabel("Latent dimension index", fontsize=9)

    ax.set_ylabel("Mean activation", fontsize=9)
    title = f"Per-dimension Mean Activation — {run_short}" if run_short else "Per-dimension Mean Activation"
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def fig_hierarchy_activations(
    model, val_loader: DataLoader, device: torch.device, save_path: Path
) -> None:
    """Bar charts: mean activation per node at each depth level."""
    _apply_style()
    metas = discover_hierarchies(model)
    if not metas:
        return

    acts_dict, _ = collect_node_activations(model, val_loader, metas, device, max_samples=2000)

    n_hier = len(metas)
    fig, axes = plt.subplots(n_hier, 1, figsize=(14, 3 * n_hier), squeeze=False)

    for h_i, meta in enumerate(metas):
        ax = axes[h_i, 0]
        acts = acts_dict[meta.label]    # [N, n_nodes]
        mean_acts = acts.mean(axis=0)   # [n_nodes]
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
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_lbl, fontsize=8)
        ax.set_ylabel("Mean activation", fontsize=9)
        hier_label = f"Hierarchy {h_i}" if meta.is_multi else "Bottleneck hierarchy"
        ax.set_title(hier_label, fontsize=10)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Per-node Mean Activation across Val Set", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def fig_hierarchy_gradcam(
    model, val_loader: DataLoader, device: torch.device, save_path: Path,
    n_images: int = 4, max_depth_shown: int = 4
) -> None:
    """GradCAM overlay on path-nodes for a few fixed images."""
    _apply_style()
    metas = discover_hierarchies(model)
    if not metas:
        return
    meta = metas[0]  # first hierarchy
    n_levels = min(meta.n_levels, max_depth_shown)

    acts_dict, images_cpu = collect_node_activations(
        model, val_loader, [meta], device, max_samples=500)
    act_mat = acts_dict[meta.label]

    imgs_show = images_cpu[:n_images]
    cell_px = 128
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
        cams = _gradcam_for_image_node_subset(
            model, img_t.unsqueeze(0), meta, path_nodes, device)

        for d in range(n_levels):
            ax = axes[img_i, d]
            overlay = _overlay(img_np, cams[d], alpha=0.45)
            ax.imshow(overlay)
            ax.axis("off")
            if img_i == 0:
                ax.set_title(f"Depth {d+1}\nNode {path[d]}", fontsize=7)

    fig.suptitle("GradCAM — Activation Path Nodes (L8 Taxon)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./data/celeba_hq")
    p.add_argument("--celeba-attrs-root", default="./data/celeba_fallback",
                   help="Root of torchvision CelebA (with attr labels) for probing")
    p.add_argument("--out-dir", default="paper_graphics_celeba_hq")
    p.add_argument("--max-images", type=int, default=3000,
                   help="Max val images for metric computation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--skip-hierarchy", action="store_true",
                   help="Skip taxonomy hierarchy visualisation")
    p.add_argument("--force-recompute", action="store_true",
                   help="Recompute metrics even if cache exists")
    p.add_argument("--n-gallery", type=int, default=6,
                   help="Number of images in reconstruction gallery")
    p.add_argument("--n-hier-images", type=int, default=4)
    p.add_argument("--skip-case-studies", action="store_true",
                   help="Skip t-SNE and neuron-steering case studies")
    p.add_argument("--n-tsne-samples", type=int, default=3000)
    p.add_argument("--steer-scale", type=float, default=10.0,
                   help="Maximum steering multiplier for neuron activation")
    p.add_argument("--n-steer-attrs", type=int, default=4,
                   help="Number of attributes to steer per model")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    main_dir = out_dir / "main"
    ablation_dir = out_dir / "ablation"
    taxonomy_dir = out_dir / "taxonomy"
    case_study_dir = out_dir / "case_studies"
    activations_dir = out_dir / "activations"
    cache_dir = out_dir / "cache"
    for d in (main_dir, ablation_dir, taxonomy_dir, case_study_dir, activations_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load run dicts ────────────────────────────────────────────────────────
    print("\n=== Loading model configs ===")
    main_runs = [r for p in MAIN_CONFIG_RELPATHS
                 if (r := build_run_dict(ROOT / p)) is not None]
    abl_taxon_runs = [r for p in ABLATION_TAXON_RELPATHS
                      if (r := build_run_dict(ROOT / p)) is not None]
    abl_multi_runs = [r for p in ABLATION_MULTI_RELPATHS
                      if (r := build_run_dict(ROOT / p)) is not None]

    print(f"Main models: {[r['short'] for r in main_runs]}")
    print(f"Ablation (single taxon): {[r['short'] for r in abl_taxon_runs]}")
    print(f"Ablation (multi-taxon):  {[r['short'] for r in abl_multi_runs]}")

    if not main_runs:
        print("ERROR: no main model checkpoints found. Check configs and output dirs.")
        return

    # ── Data loaders ──────────────────────────────────────────────────────────
    print("\n=== Building data loaders ===")
    val_loader = make_celeba_hq_val_loader(
        args.data_root, args.image_size, args.batch_size, args.num_workers, args.max_images)
    probe_tr_loader, probe_va_loader = make_celeba_attr_loaders(
        args.celeba_attrs_root, args.image_size, batch_size=128,
        num_workers=args.num_workers)
    if probe_tr_loader is None:
        print("  [info] CelebA attrs not found — probe/KNN/monosemanticity skipped")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\n=== Computing / loading metrics ===")
    def _get_metrics(runs):
        return [compute_all_metrics(
            r, val_loader, device, probe_tr_loader, probe_va_loader,
            args.max_images, cache_dir, args.force_recompute
        ) for r in runs]

    main_metrics = _get_metrics(main_runs)
    abl_taxon_metrics = _get_metrics(abl_taxon_runs)
    abl_multi_metrics = _get_metrics(abl_multi_runs)

    # ── Training curves ───────────────────────────────────────────────────────
    print("\n=== Generating figures ===")
    fig_training_curves(main_runs, main_dir / "fig01_training_curves.png")

    # ── Reconstruction gallery ────────────────────────────────────────────────
    fig_reconstruction_gallery(main_runs, val_loader, device,
                                main_dir / "fig02_reconstruction_gallery.png",
                                n_images=args.n_gallery)

    # ── Scatter: probe AUC vs L0 ──────────────────────────────────────────────
    _scatter_vs_l0(main_runs, main_metrics,
                   y_key="probe_mean_auc", y_label="Linear-probe mean AUC-ROC (40 attrs)",
                   title="Downstream probing vs. sparsity — CelebA-HQ",
                   save_path=main_dir / "fig03_sparsity_probing.png")

    # ── Scatter: 5-NN vs L0 ──────────────────────────────────────────────────
    _scatter_vs_l0(main_runs, main_metrics,
                   y_key="knn_k5_mean_acc", y_label="5-NN mean accuracy (40 attrs)",
                   title="KNN accuracy vs. sparsity — CelebA-HQ",
                   save_path=main_dir / "fig04_sparsity_knn.png")

    # ── Scatter: monosemanticity vs L0 ───────────────────────────────────────
    _scatter_vs_l0(main_runs, main_metrics,
                   y_key="monosemanticity",
                   y_label="Monosemanticity\n(mean max-attribute lift, top-50 images)",
                   title="Monosemanticity vs. sparsity — CelebA-HQ",
                   save_path=main_dir / "fig05_monosemanticity.png")

    # ── Metrics overview bar chart ─────────────────────────────────────────────
    fig_metrics_overview(main_runs, main_metrics, main_dir / "fig06_metrics_overview.png")

    # ── Ablation: depth sweep ─────────────────────────────────────────────────
    if abl_taxon_runs:
        fig_ablation_bar(abl_taxon_runs, abl_taxon_metrics,
                         ablation_dir / "fig07_ablation_depth.png",
                         title="Depth ablation — Bottleneck TopK Taxon (L4/L6/L8/L10)")

    # ── Ablation: multi-taxon sweep ───────────────────────────────────────────
    if abl_multi_runs:
        fig_ablation_bar(abl_multi_runs, abl_multi_metrics,
                         ablation_dir / "fig08_ablation_multitaxon.png",
                         title="Multi-taxon sweep (K2/K4/K8/K12/K16, various depths)")

    # ── Case studies: t-SNE + neuron steering ────────────────────────────────
    if not args.skip_case_studies:
        print("\n=== Case studies ===")
        for run in main_runs:
            run_cs_dir = case_study_dir / run["name"]
            run_cs_dir.mkdir(parents=True, exist_ok=True)

            fig_tsne_celeba(
                run=run,
                cache_dir=cache_dir,
                probe_va_loader=probe_va_loader,
                device=device,
                save_path=run_cs_dir / "fig12_tsne.png",
                n_attrs_shown=6,
                max_tsne_samples=args.n_tsne_samples,
                force=args.force_recompute,
            )

            fig_neuron_steering_celeba(
                run=run,
                cache_dir=cache_dir,
                probe_tr_loader=probe_tr_loader,
                probe_va_loader=probe_va_loader,
                val_loader=val_loader,
                device=device,
                save_path=run_cs_dir / "fig13_neuron_steering.png",
                n_attrs=args.n_steer_attrs,
                steer_scale=args.steer_scale,
                force=args.force_recompute,
            )

    # ── Taxonomy exploration ──────────────────────────────────────────────────
    if not args.skip_hierarchy:
        taxon_runs = [r for r in main_runs
                      if r["type"] in ("bottleneck_jumprelu_taxon", "bottleneck_topk_taxon",
                                       "bottleneck_topk_multi_taxon", "conv_bottleneck_topk_taxon",
                                       "bottleneck_taxon")]
        for run in taxon_runs:
            run_tax_dir = taxonomy_dir / run["name"]
            run_tax_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n  Hierarchy viz for {run['short']} ...")
            try:
                model, _ = _load_model_from_run_dict(run, device)
                model.eval()
            except Exception as e:
                print(f"    [skip] could not load: {e}")
                continue

            # Only prefix recon for single-taxon models (is_multi=False)
            metas = discover_hierarchies(model)
            single_metas = [m for m in metas if not m.is_multi]
            if single_metas:
                fig_hierarchy_prefix_recon(
                    model, val_loader, device,
                    run_tax_dir / "fig09_prefix_recon.png",
                    n_images=args.n_hier_images,
                    run_short=run["short"],
                )

            fig_hierarchy_activations(
                model, val_loader, device,
                run_tax_dir / "fig10_activations.png")

            fig_hierarchy_gradcam(
                model, val_loader, device,
                run_tax_dir / "fig11_gradcam.png",
                n_images=args.n_hier_images)

            del model

    # ── Flat per-dimension activation charts for ALL autoencoders ─────────────
    if not args.skip_hierarchy:
        print("\n=== Flat activation charts (all models) ===")
        for run in main_runs:
            run_act_dir = activations_dir / run["name"]
            run_act_dir.mkdir(parents=True, exist_ok=True)
            save_path = run_act_dir / "fig10_flat_activations.png"
            print(f"\n  Flat activations for {run['short']} ...")
            try:
                model, _ = _load_model_from_run_dict(run, device)
                model.eval()
            except Exception as e:
                print(f"    [skip] could not load: {e}")
                continue
            try:
                fig_flat_activations(
                    model, val_loader, device, save_path,
                    run_short=run["short"],
                )
            except Exception as e:
                print(f"    [warn] flat activations failed: {e}")
            del model

    print(f"\n✓ All paper graphics saved to {out_dir}/")


if __name__ == "__main__":
    main()
