#!/usr/bin/env python3
"""Per-model paper analysis for CelebA-HQ.

Computes all evaluation metrics for ONE model checkpoint and saves results to
``outputs/paper/celeba_hq/<run_name>/``.

Metrics
-------
1. Reconstruction  : MSE, MAE, PSNR  (on full val set)
2. Sparsity        : mean L0, dead-neuron fraction (threshold 1e-6)
3. Linear probing  : logistic regression on 40 CelebA binary attributes
                     (requires ``--celeba-root``; skipped if unavailable)
4. KNN             : 1-NN and 5-NN accuracy on the same attributes
5. Targeted editing: zero the top-K active channels and decode

Saved files (all under ``<save_dir>/<run_name>/``)
---------------------------------------------------
metrics.json                  — all scalar metrics
metrics_summary.csv           — same, CSV form
latents.npy                   — [N, D] flattened latent codes
images_orig.npy               — [N, C, H, W] original val images (float [0,1])
labels.npy                    — [N, 40] binary CelebA attributes (if available)
attr_names.json               — list of 40 attribute name strings
reconstruction_samples.png
editing_samples.png

Usage
-----
python src/paper/paper_analysis_celeba.py \\
    --config configs/celeba_hq/taxon_ae_celeba_hq.json \\
    --celeba-root ./data/celeba_fallback \\
    --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
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
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.taxon.taxon_ae import TaxonAutoencoder
from src.model.cnn.taxon.multi_taxon_ae import MultiTaxonAutoencoder
from src.model.cnn.taxon.topk_taxon_ae import TopKTaxonAutoencoder
from src.model.cnn.taxon.topk_multi_taxon_ae import TopKMultiTaxonAutoencoder
from src.model.cnn.taxon.bias_taxon_ae import BiasTaxonAutoencoder
from src.model.cnn.taxon.bias_multi_taxon_ae import BiasMultiTaxonAutoencoder
from src.model.cnn.baseline.sae import SparseConvAutoencoder
from src.model.cnn.baseline.topk_sae import TopKSparseConvAutoencoder
from src.model.cnn.baseline.gated_sae import GatedSparseConvAutoencoder
from src.model.cnn.baseline.jumprelu_sae import JumpReLUSparseConvAutoencoder
from src.model.cnn.baseline.matryoshka_batch_topk_sae import MatryoshkaBatchTopKSparseConvAutoencoder
from src.model.cnn.baseline.softmax_sae import SoftmaxSparseConvAutoencoder
from src.model.cnn.baseline.baseline_ae import BaselineConvAutoencoder
from src.utils.dataloader import CelebAHQLoader


# ---------------------------------------------------------------------------
# CelebA attribute names
# ---------------------------------------------------------------------------

CELEBA_ATTR_NAMES = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young",
]

# ---------------------------------------------------------------------------
# Suffix reconstruction (matches training scripts)
# ---------------------------------------------------------------------------

def _training_output_dir_suffix(cfg: dict) -> str:
    mc = cfg.get("model", {})
    tc = cfg.get("training", {})
    variant = mc.get("model_variant", None)
    exp_name = cfg.get("experiment_name", "")

    if variant is not None:
        # SAE family
        sw = tc.get("sparsity_weight", 1e-3)
        if variant == "matryoshka_batch_topk":
            k_vals = sorted(mc.get("k_values", [64, 32, 16]), reverse=True)
            k_str = "-".join(str(k) for k in k_vals)
            return f"_k{k_str}_sw{sw:.0e}"
        if variant == "softmax_sae":
            dw = float(mc.get("dkl_weight", tc.get("dkl_weight", 1e-2)))
            temperature = float(mc.get("temperature", 1.0))
            temp_str = f"{temperature:g}".replace(".", "p")
            ew = float(mc.get("entropy_weight", tc.get("entropy_weight", 0.0)))
            ew_suffix = f"_ew_{ew:.0e}" if ew else ""
            return f"_dkl_{dw:.0e}_temp_{temp_str}{ew_suffix}"
        if variant == "topk":
            k = mc.get("topk_k", 64)
            return f"_k{k}_sw{sw:.0e}"
        if variant == "gated":
            ste_tag = "_ste" if mc.get("use_gate_ste", False) else ""
            return f"_sw{sw:.0e}{ste_tag}"
        if variant == "jumprelu":
            l0 = int(mc.get("target_l0", 64))
            return f"_l0{l0}_sw{sw:.0e}"
        # plain SAE (l1/kl)
        spt = mc.get("sparsity_type", "l1")
        return f"_spw_{sw:.0e}_spt_{spt}"

    # Taxon family — infer from experiment name
    dw = float(tc.get("dkl_weight", 1e-2))
    temperature = float(mc.get("temperature", tc.get("temperature", 1.0)))
    temp_str = f"{temperature:g}".replace(".", "p")
    hard_suffix = "_hard" if mc.get("hard", False) else ""
    ew = float(tc.get("entropy_weight", 0.0))
    ew_suffix = f"_ew_{ew:.0e}" if ew else ""

    if "topk_multi_taxon" in exp_name:
        n_hier = mc.get("n_hierarchies", 3)
        auxk = float(tc.get("auxk_weight", 0.0))
        auxk_suffix = f"_auxk_{auxk:.0e}" if auxk else ""
        return f"_K{n_hier}{auxk_suffix}"

    if "topk_taxon" in exp_name:
        auxk = float(tc.get("auxk_weight", 0.0))
        auxk_suffix = f"_auxk_{auxk:.0e}" if auxk else ""
        return auxk_suffix

    if "bias_multi_taxon" in exp_name:
        n_hier = mc.get("n_hierarchies", 3)
        bur = float(mc.get("bias_update_rate", 1e-3))
        return f"_K{n_hier}_bur_{bur:.0e}"

    if "bias_taxon" in exp_name:
        bur = float(mc.get("bias_update_rate", 1e-3))
        return f"_bur_{bur:.0e}"

    if "multi_taxon" in exp_name:
        n_hier = mc.get("n_hierarchies", 3)
        gdkl = float(tc.get("gate_dkl_weight", 0.0))
        gdkl_suffix = f"_gdkl_{gdkl:.0e}" if gdkl else ""
        gew = float(tc.get("gate_entropy_weight", 0.0))
        gew_suffix = f"_gew_{gew:.0e}" if gew else ""
        return (f"_dkl_{dw:.0e}_temp_{temp_str}{hard_suffix}"
                f"_K{n_hier}{ew_suffix}{gdkl_suffix}{gew_suffix}")

    if "baseline" in exp_name:
        return ""

    # plain taxon_ae
    return f"_dkl_{dw:.0e}_temp_{temp_str}{hard_suffix}{ew_suffix}"


def _resolve_checkpoint(cfg: dict, ckpt_override: Optional[str]) -> Path:
    """Find best.pt, respecting explicit override and suffix logic."""
    if ckpt_override:
        p = Path(ckpt_override)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    # Try analysis.checkpoint_path from config
    ckpt_from_cfg = cfg.get("analysis", {}).get("checkpoint_path", "")
    if ckpt_from_cfg:
        p = ROOT / ckpt_from_cfg if not Path(ckpt_from_cfg).is_absolute() else Path(ckpt_from_cfg)
        if p.exists():
            return p

    base = Path(cfg["output"]["output_dir"])
    if not base.is_absolute():
        base = ROOT / base

    # Try with suffix
    suffix = _training_output_dir_suffix(cfg)
    suffixed = Path(str(base) + suffix)
    for candidate in (suffixed, base):
        ckpt = candidate / "checkpoints" / "best.pt"
        if ckpt.exists():
            return ckpt

    raise FileNotFoundError(
        f"Could not find best.pt under {base} or {suffixed}. "
        "Pass --checkpoint explicitly."
    )


def _run_name(cfg: dict) -> str:
    base = Path(cfg["output"]["output_dir"]).name
    suffix = _training_output_dir_suffix(cfg)
    return base + suffix


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _infer_model_type(cfg: dict) -> str:
    mc = cfg.get("model", {})
    variant = mc.get("model_variant", None)
    exp = cfg.get("experiment_name", "")
    if variant == "matryoshka_batch_topk":
        return "matryoshka_batch_topk_sae"
    if variant == "softmax_sae":
        return "softmax_sae"
    if variant == "topk":
        return "topk_sae"
    if variant == "gated":
        return "gated_sae"
    if variant == "jumprelu":
        return "jumprelu_sae"
    if variant in ("l1", "kl"):
        return "sae"
    if variant is not None:
        return "sae"
    # Taxon family
    if "topk_multi_taxon" in exp:
        return "topk_multi_taxon"
    if "topk_taxon" in exp:
        return "topk_taxon"
    if "bias_multi_taxon" in exp:
        return "bias_multi_taxon"
    if "bias_taxon" in exp:
        return "bias_taxon"
    if "multi_taxon" in exp:
        return "multi_taxon"
    if "baseline" in exp:
        return "baseline"
    if "stage_taxonomy_layers" in mc:
        return "taxon"
    return "baseline"


def load_model(ckpt_path: Path, model_type: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt.get("args", {})

    if model_type == "taxon":
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
    elif model_type == "multi_taxon":
        model = MultiTaxonAutoencoder(
            in_channels=a.get("in_channels", 3),
            resnet_variant=a.get("resnet_variant", "18"),
            stage_taxonomy_layers=tuple(a.get("stage_taxonomy_layers", [5, 6, 7, 8])),
            stage_strides=tuple(a.get("stage_strides", [1, 2, 2, 2])),
            stage_blocks=a.get("stage_blocks", None),
            n_hierarchies=a.get("n_hierarchies", 3),
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
    elif model_type == "topk_taxon":
        model = TopKTaxonAutoencoder(
            in_channels=a.get("in_channels", 3),
            resnet_variant=a.get("resnet_variant", "18"),
            stage_taxonomy_layers=tuple(a.get("stage_taxonomy_layers", [5, 6, 7, 8])),
            stage_strides=tuple(a.get("stage_strides", [1, 2, 2, 2])),
            stage_blocks=a.get("stage_blocks", None),
            k_aux=a.get("k_aux", None),
            dead_steps=a.get("dead_steps", 2000),
            kernel_size=a.get("kernel_size", 3),
            use_stem=a.get("use_stem", True),
            stem_channels=a.get("stem_channels", 64),
            stem_stride=a.get("stem_stride", 2),
            use_stem_maxpool=a.get("use_stem_maxpool", True),
            output_activation=a.get("output_activation", "none"),
            depth_decay=a.get("depth_decay", 0.5),
            temperature=a.get("temperature", 1.0),
            hard=a.get("hard", False),
        )
    elif model_type == "topk_multi_taxon":
        model = TopKMultiTaxonAutoencoder(
            in_channels=a.get("in_channels", 3),
            resnet_variant=a.get("resnet_variant", "18"),
            stage_taxonomy_layers=tuple(a.get("stage_taxonomy_layers", [3, 4, 5, 6])),
            stage_strides=tuple(a.get("stage_strides", [1, 2, 2, 2])),
            stage_blocks=a.get("stage_blocks", None),
            n_hierarchies=a.get("n_hierarchies", 3),
            k_aux=a.get("k_aux", None),
            dead_steps=a.get("dead_steps", 2000),
            gate_k=a.get("gate_k", 1),
            kernel_size=a.get("kernel_size", 3),
            use_stem=a.get("use_stem", True),
            stem_channels=a.get("stem_channels", 64),
            stem_stride=a.get("stem_stride", 2),
            use_stem_maxpool=a.get("use_stem_maxpool", True),
            output_activation=a.get("output_activation", "none"),
            depth_decay=a.get("depth_decay", 0.5),
            temperature=a.get("temperature", 1.0),
            hard=a.get("hard", False),
        )
    elif model_type == "bias_taxon":
        model = BiasTaxonAutoencoder(
            in_channels=a.get("in_channels", 3),
            resnet_variant=a.get("resnet_variant", "18"),
            stage_taxonomy_layers=tuple(a.get("stage_taxonomy_layers", [5, 6, 7, 8])),
            stage_strides=tuple(a.get("stage_strides", [1, 2, 2, 2])),
            stage_blocks=a.get("stage_blocks", None),
            bias_update_rate=a.get("bias_update_rate", 0.001),
            bias_ema_decay=a.get("bias_ema_decay", 0.99),
            kernel_size=a.get("kernel_size", 3),
            use_stem=a.get("use_stem", True),
            stem_channels=a.get("stem_channels", 64),
            stem_stride=a.get("stem_stride", 2),
            use_stem_maxpool=a.get("use_stem_maxpool", True),
            output_activation=a.get("output_activation", "none"),
            depth_decay=a.get("depth_decay", 0.5),
            temperature=a.get("temperature", 1.0),
            hard=a.get("hard", False),
        )
    elif model_type == "bias_multi_taxon":
        model = BiasMultiTaxonAutoencoder(
            in_channels=a.get("in_channels", 3),
            resnet_variant=a.get("resnet_variant", "18"),
            stage_taxonomy_layers=tuple(a.get("stage_taxonomy_layers", [3, 4, 5, 6])),
            stage_strides=tuple(a.get("stage_strides", [1, 2, 2, 2])),
            stage_blocks=a.get("stage_blocks", None),
            n_hierarchies=a.get("n_hierarchies", 3),
            bias_update_rate=a.get("bias_update_rate", 0.001),
            bias_ema_decay=a.get("bias_ema_decay", 0.99),
            gate_k=a.get("gate_k", 1),
            kernel_size=a.get("kernel_size", 3),
            use_stem=a.get("use_stem", True),
            stem_channels=a.get("stem_channels", 64),
            stem_stride=a.get("stem_stride", 2),
            use_stem_maxpool=a.get("use_stem_maxpool", True),
            output_activation=a.get("output_activation", "none"),
            depth_decay=a.get("depth_decay", 0.5),
            temperature=a.get("temperature", 1.0),
            hard=a.get("hard", False),
        )
    elif model_type == "baseline":
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
    else:
        # SAE family
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
        if variant == "matryoshka_batch_topk":
            k_vals = sorted(a.get("k_values", [8, 4, 2]), reverse=True)
            model = MatryoshkaBatchTopKSparseConvAutoencoder(
                **common, k_values=k_vals, k_aux=a.get("k_aux", None),
                use_aux_loss=False, dead_threshold=a.get("dead_threshold", 1e-3),
            )
        elif variant == "softmax_sae":
            model = SoftmaxSparseConvAutoencoder(**common, temperature=a.get("temperature", 1.0))
        elif variant == "topk":
            model = TopKSparseConvAutoencoder(
                **common, topk_k=a.get("topk_k", 64), k_aux=a.get("k_aux", 64),
                use_aux_loss=False, dead_threshold=a.get("dead_threshold", 1e-3),
            )
        elif variant == "gated":
            model = GatedSparseConvAutoencoder(**common, use_gate_ste=a.get("use_gate_ste", False))
        elif variant == "jumprelu":
            model = JumpReLUSparseConvAutoencoder(
                **common, target_l0=a.get("target_l0", 64.0),
                bandwidth=a.get("bandwidth", 0.001), theta_init=a.get("theta_init", 0.1),
            )
        else:
            model = SparseConvAutoencoder(
                **common, sparsity_type=a.get("sparsity_type", "l1"),
                sparsity_target=a.get("sparsity_target", 0.05),
                latent_activation=a.get("latent_activation", "relu"),
            )

    model.load_state_dict(ckpt["model_state"], strict=(model_type != "baseline"))
    model.to(device).eval()
    epoch = ckpt.get("epoch", "?")
    best = ckpt.get("best_val", float("nan"))
    print(f"  epoch={epoch}  best_val={best:.6f}")
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_display(t: torch.Tensor) -> np.ndarray:
    img = t.detach().cpu().squeeze(0).permute(1, 2, 0)
    return img.clamp(0, 1).numpy()


@torch.no_grad()
def _encode(model, imgs: torch.Tensor) -> torch.Tensor:
    """Return flat latent [B, D] for any model type."""
    z, *_ = model.encode(imgs)
    return z.detach().cpu().flatten(start_dim=1)  # [B, C*H*W]


@torch.no_grad()
def _encode_spatial(model, imgs: torch.Tensor) -> torch.Tensor:
    """Return spatial latent [B, C, H, W] for editing."""
    z, *_ = model.encode(imgs)
    return z.detach()


@torch.no_grad()
def _decode(model, z: torch.Tensor) -> torch.Tensor:
    """Decode a spatial latent tensor."""
    return model.decode(z).detach().cpu()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_latents_and_recon(
    model,
    loader,
    device: torch.device,
    dead_threshold: float = 1e-6,
    max_images: int = 0,
) -> dict:
    """Run the val loader; return latents, images, and reconstruction metrics."""
    all_latents: List[np.ndarray] = []
    all_imgs: List[np.ndarray] = []
    all_mse: List[float] = []
    all_mae: List[float] = []
    n_seen = 0

    for imgs, _ in tqdm(loader, desc="  Collecting latents"):
        if max_images > 0 and n_seen >= max_images:
            break
        imgs = imgs.to(device)
        z, *_ = model.encode(imgs)      # single encoder pass
        recon = model.decode(z)
        if isinstance(recon, (tuple, list)):
            recon = recon[0]

        z_flat = z.detach().cpu().flatten(start_dim=1)  # [B, D]
        all_latents.append(z_flat.numpy())

        all_imgs.append(imgs.cpu().numpy())
        mse_batch = ((imgs - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
        mae_batch = torch.abs(imgs - recon).mean(dim=(1, 2, 3)).cpu().numpy()
        all_mse.extend(mse_batch.tolist())
        all_mae.extend(mae_batch.tolist())
        n_seen += imgs.shape[0]

    Z = np.concatenate(all_latents, axis=0)      # [N, D]
    imgs_np = np.concatenate(all_imgs, axis=0)   # [N, C, H, W]
    mse = np.array(all_mse)
    mae = np.array(all_mae)

    # L0: number of dims > threshold per sample
    l0_per_sample = (np.abs(Z) > dead_threshold).sum(axis=1).astype(float)

    # Dead neurons: channels never active across all samples
    # Z is [N, D]; a feature is dead if max(|z|) < threshold
    max_act = np.abs(Z).max(axis=0)              # [D]
    dead_frac = float((max_act < dead_threshold).mean())

    return {
        "Z": Z,
        "imgs": imgs_np,
        "mse": mse,
        "mae": mae,
        "l0_per_sample": l0_per_sample,
        "dead_frac": dead_frac,
    }


# ---------------------------------------------------------------------------
# CelebA attribute loader
# ---------------------------------------------------------------------------

def try_load_celeba_attrs(celeba_root: Optional[str], image_size: int, batch_size: int,
                          num_workers: int, n_train: int = 4000, n_val: int = 1000,
                          probe_batch_size: int = 256,
                          ) -> Tuple[Optional[DataLoader], Optional[DataLoader], List[str]]:
    """Load CelebA attribute-labeled images for probing.

    Uses a fixed subset of CelebA: first ``n_train`` for fitting the probe,
    next ``n_val`` for evaluation. Returns (train_loader, val_loader, attr_names)
    or (None, None, []) if CelebA is not available.
    """
    if not celeba_root:
        return None, None, []

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    try:
        ds_all = datasets.CelebA(
            root=celeba_root, split="all", target_type="attr",
            transform=transform, download=False,
        )
    except Exception as e:
        print(f"  [skip attr] Could not load CelebA from {celeba_root}: {e}")
        return None, None, []

    total = len(ds_all)
    n_train = min(n_train, total // 2)
    n_val = min(n_val, total - n_train)
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_train + n_val))

    train_ds = Subset(ds_all, train_idx)
    val_ds = Subset(ds_all, val_idx)

    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=probe_batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), drop_last=False,
    )
    return make_loader(train_ds, True), make_loader(val_ds, False), CELEBA_ATTR_NAMES


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_probe_features(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Z [N, D], Y [N, 40]) from a CelebA attribute loader."""
    Zs, Ys = [], []
    t0 = time.time()
    n_batches = len(loader)
    for batch_idx, (imgs, attrs) in enumerate(tqdm(loader, desc="    Encoding probe set")):
        imgs = imgs.to(device)
        z = _encode(model, imgs)
        Zs.append(z.numpy())
        Ys.append(attrs.numpy())
        if (batch_idx + 1) % max(1, n_batches // 4) == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            n_done = sum(x.shape[0] for x in Zs)
            print(f"    [{batch_idx+1}/{n_batches}]  {n_done} samples  "
                  f"z_dim={Zs[-1].shape[1]}  t={elapsed:.1f}s", flush=True)
    Z = np.concatenate(Zs)
    Y = np.concatenate(Ys)
    print(f"    Done: Z={Z.shape}  Y={Y.shape}  t={time.time()-t0:.1f}s", flush=True)
    return Z, Y


def run_linear_probe(Z_train, Y_train, Z_val, Y_val, device=None,
                     n_epochs=300, lr=3e-3, batch_size=512) -> dict:
    """GPU-accelerated multi-label logistic regression over all 40 attributes.

    Trains a single Linear(D→40) layer with BCEWithLogitsLoss using Adam.
    All 40 classifiers are fit simultaneously on GPU in one pass.
    """
    t0 = time.time()
    if device is None:
        device = torch.device("cpu")
    D = Z_train.shape[1]
    n_attrs = Y_train.shape[1]
    print(f"    GPU probe: D={D}  n_train={len(Z_train)}  n_val={len(Z_val)}"
          f"  n_attrs={n_attrs}  epochs={n_epochs}  device={device}", flush=True)

    Z_tr = torch.from_numpy(Z_train).float().to(device)
    Y_tr = torch.from_numpy(Y_train).float().to(device)
    Z_va = torch.from_numpy(Z_val).float().to(device)
    Y_va = torch.from_numpy(Y_val).float().to(device)

    # Standardise features (fit on train)
    mean = Z_tr.mean(0, keepdim=True)
    std  = Z_tr.std(0, keepdim=True).clamp(min=1e-6)
    Z_tr = (Z_tr - mean) / std
    Z_va = (Z_va - mean) / std

    clf = torch.nn.Linear(D, n_attrs, bias=True).to(device)
    torch.nn.init.zeros_(clf.weight)
    torch.nn.init.zeros_(clf.bias)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)

    n = len(Z_tr)
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = clf(Z_tr[idx])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, Y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch+1}/{n_epochs}  loss={epoch_loss/n_batches:.4f}"
                  f"  t={time.time()-t0:.1f}s", flush=True)

    with torch.no_grad():
        proba_va = torch.sigmoid(clf(Z_va)).cpu().numpy()  # [N_val, 40]
    Y_va_np = Y_val.astype(int)
    pred_np  = (proba_va >= 0.5).astype(int)

    aucs, accs = [], []
    for i in range(n_attrs):
        y = Y_va_np[:, i]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        try:
            aucs.append(roc_auc_score(y, proba_va[:, i]))
        except Exception:
            aucs.append(float("nan"))
        accs.append(float((pred_np[:, i] == y).mean()))

    print(f"    probe done  mean_auc={np.nanmean(aucs):.3f}  "
          f"mean_acc={np.nanmean(accs):.3f}  t={time.time()-t0:.1f}s", flush=True)
    return {
        "probe_mean_auc": float(np.nanmean(aucs)),
        "probe_mean_acc": float(np.nanmean(accs)),
        "probe_per_attr_auc": aucs,
        "probe_per_attr_acc": accs,
    }


def run_knn(Z_train, Y_train, Z_val, Y_val, ks=(1, 5), device=None) -> dict:
    """GPU-accelerated cosine KNN over all 40 attributes simultaneously.

    Computes the full [N_val × N_train] cosine similarity matrix in one
    matmul on GPU, then reads off majority-vote labels for every k and every
    attribute in a single vectorised pass — no per-attribute loop needed.
    """
    t0 = time.time()
    if device is None:
        device = torch.device("cpu")
    max_k = max(ks)
    print(f"    GPU KNN: k={ks}  n_train={len(Z_train)}  n_val={len(Z_val)}"
          f"  D={Z_train.shape[1]}  device={device}", flush=True)

    Z_tr = torch.from_numpy(Z_train).float().to(device)
    Z_va = torch.from_numpy(Z_val).float().to(device)
    Y_tr = torch.from_numpy(Y_train).float().to(device)  # [N_tr, 40]
    Y_va_np = Y_val.astype(int)  # keep on CPU for accuracy computation

    # L2-normalise for cosine similarity
    Z_tr_n = torch.nn.functional.normalize(Z_tr, dim=1)
    Z_va_n = torch.nn.functional.normalize(Z_va, dim=1)

    print(f"    Computing similarity matrix ({Z_va.shape[0]}×{Z_tr.shape[0]})...",
          flush=True)
    sim = Z_va_n @ Z_tr_n.T          # [N_val, N_train]
    print(f"    Similarity done  t={time.time()-t0:.1f}s", flush=True)

    # Retrieve top-max_k neighbours once; slice for smaller k values
    topk_idx = sim.topk(max_k, dim=1).indices   # [N_val, max_k]

    n_attrs = Y_train.shape[1]
    results = {}
    for k in ks:
        neighbors = Y_tr[topk_idx[:, :k]]        # [N_val, k, 40]
        # Majority vote: mean >= 0.5
        pred = (neighbors.mean(dim=1) >= 0.5).cpu().numpy().astype(int)  # [N_val, 40]
        accs = []
        for i in range(n_attrs):
            y = Y_va_np[:, i]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            accs.append(float((pred[:, i] == y).mean()))
        acc = float(np.mean(accs)) if accs else float("nan")
        results[f"knn_k{k}_mean_acc"] = acc
        print(f"    k={k}: mean_acc={acc:.3f}  t={time.time()-t0:.1f}s", flush=True)

    return results


# ---------------------------------------------------------------------------
# Targeted editing
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_targeted_editing(model, val_loader, device, n_images=4, topk_channels=32,
                         image_size=256):
    """Zero the top-K active channels; decode and compare."""
    imgs_list, orig_recons, edited_recons = [], [], []
    for imgs, _ in val_loader:
        imgs = imgs.to(device)
        batch_size = imgs.shape[0]
        need = n_images - len(imgs_list)
        if need <= 0:
            break
        imgs = imgs[:need]

        # Original reconstruction
        out = model(imgs)
        recon_orig = out[0].clamp(0, 1)

        # Get spatial latent
        z_spatial = _encode_spatial(model, imgs)  # [B, C, H, W]

        # Channel activity: mean abs over spatial dims
        activity = z_spatial.abs().mean(dim=(2, 3))  # [B, C]
        topk_idx = activity.topk(topk_channels, dim=1).indices  # [B, topk]

        z_edit = z_spatial.clone()
        for b in range(z_edit.shape[0]):
            z_edit[b, topk_idx[b]] = 0.0

        recon_edit = model.decode(z_edit).clamp(0, 1).cpu()

        imgs_list.append(imgs.cpu())
        orig_recons.append(recon_orig.cpu())
        edited_recons.append(recon_edit)

        if len(imgs_list) * batch_size >= n_images:
            break

    return imgs_list, orig_recons, edited_recons


# ---------------------------------------------------------------------------
# Save graphics
# ---------------------------------------------------------------------------

def save_recon_samples(imgs, loader, model, device, save_path, n_images=8):
    img_batch, label_batch = next(iter(loader))
    img_batch = img_batch[:n_images].to(device)
    with torch.no_grad():
        recon_batch = model(img_batch)[0].clamp(0, 1).cpu()

    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 4))
    for i in range(n_images):
        axes[0, i].imshow(_to_display(img_batch[i:i+1].cpu()))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("orig", fontsize=8)
        axes[1, i].imshow(_to_display(recon_batch[i:i+1]))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("recon", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def save_editing_plot(imgs_list, orig_recons, edited_recons, save_path, topk):
    imgs_cat = torch.cat(imgs_list)
    origs_cat = torch.cat(orig_recons)
    edits_cat = torch.cat(edited_recons)
    n = min(len(imgs_cat), 4)

    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
    row_labels = ["Original", "Reconstruction", f"Edited (zero top-{topk} ch)"]
    for row, batch in enumerate([imgs_cat, origs_cat, edits_cat]):
        for col in range(n):
            axes[row, col].imshow(_to_display(batch[col:col+1]))
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(row_labels[row], fontsize=9)
    plt.suptitle(f"Targeted editing: zero top-{topk} active channels", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Paper analysis for one CelebA-HQ model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--celeba-root", default=None,
                        help="Root for torchvision datasets.CelebA (for attribute probing). "
                             "Defaults to <data-root>/../celeba_fallback if not set.")
    parser.add_argument("--n-edit", type=int, default=4)
    parser.add_argument("--topk-edit", type=int, default=32)
    parser.add_argument("--n-recon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--probe-batch-size", type=int, default=256,
                        help="Batch size for probe feature extraction (inference only; can be large)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--save-dir", default=None,
                        help="Root paper dir; default outputs/paper/celeba_hq")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-n-train", type=int, default=4000)
    parser.add_argument("--probe-n-val", type=int, default=1000)
    args = parser.parse_args()

    # ── config ──────────────────────────────────────────────────────────────
    cfg_path = ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
    with open(cfg_path) as f:
        cfg = json.load(f)

    dc = cfg.get("data", {})
    image_size = dc.get("image_size", 256)
    batch_size = args.batch_size or dc.get("batch_size", 16)
    num_workers = args.num_workers
    val_split = args.val_split if args.val_split is not None else dc.get("val_split", 0.05)
    data_root = args.data_root or dc.get("data_root", "./data/celeba_hq")
    if not Path(data_root).is_absolute():
        data_root = str(ROOT / data_root)

    celeba_root = args.celeba_root
    if celeba_root is None:
        candidate = Path(data_root).parent / "celeba_fallback"
        if candidate.exists():
            celeba_root = str(candidate)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_name = _run_name(cfg)
    save_dir = Path(args.save_dir or (ROOT / "outputs" / "paper" / "celeba_hq")) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run : {run_name}")
    print(f"Save: {save_dir}")

    # ── checkpoint ──────────────────────────────────────────────────────────
    ckpt_path = _resolve_checkpoint(cfg, args.checkpoint)
    print(f"Ckpt: {ckpt_path}")
    model_type = _infer_model_type(cfg)
    print(f"Type: {model_type}")
    model = load_model(ckpt_path, model_type, device)

    # ── val data ────────────────────────────────────────────────────────────
    loader_obj = CelebAHQLoader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        val_split=val_split,
    )
    _, val_loader = loader_obj.get_loaders()
    if val_loader is None:
        val_loader = loader_obj.train_loader
        print("  [warn] No val split; using training data for analysis.")

    # ── 1. Latents + reconstruction metrics ─────────────────────────────────
    print("\n[1] Collecting latents and reconstruction metrics...")
    result = collect_latents_and_recon(model, val_loader, device)
    Z = result["Z"]
    imgs_np = result["imgs"]
    mse = result["mse"]
    mae = result["mae"]
    l0 = result["l0_per_sample"]
    dead_frac = result["dead_frac"]

    # PSNR from MSE (images in [0,1])
    psnr = -10 * np.log10(np.maximum(mse, 1e-10))

    metrics: dict = {
        "mse_mean": float(mse.mean()),
        "mse_std": float(mse.std()),
        "mae_mean": float(mae.mean()),
        "mae_std": float(mae.std()),
        "psnr_mean": float(psnr.mean()),
        "psnr_std": float(psnr.std()),
        "l0_mean": float(l0.mean()),
        "l0_std": float(l0.std()),
        "dead_neuron_frac": dead_frac,
        "latent_dim": int(Z.shape[1]),
        "n_val_images": int(Z.shape[0]),
    }
    print(f"  MSE={metrics['mse_mean']:.5f}  MAE={metrics['mae_mean']:.5f}  "
          f"PSNR={metrics['psnr_mean']:.2f}dB")
    print(f"  L0={metrics['l0_mean']:.1f}  dead={dead_frac*100:.1f}%  "
          f"latent_dim={metrics['latent_dim']}")

    np.save(save_dir / "latents.npy", Z)
    np.save(save_dir / "images_orig.npy", imgs_np[:min(200, len(imgs_np))])   # keep small

    # ── 2. Reconstruction samples plot ──────────────────────────────────────
    print("\n[2] Saving reconstruction samples...")
    save_recon_samples(imgs_np, val_loader, model, device,
                       save_dir / "reconstruction_samples.png", n_images=args.n_recon)

    # ── 3. CelebA attribute probing ─────────────────────────────────────────
    print("\n[3] Attribute probing (linear + KNN)...")
    probe_train_loader, probe_val_loader, attr_names = try_load_celeba_attrs(
        celeba_root, image_size, batch_size, num_workers,
        n_train=args.probe_n_train, n_val=args.probe_n_val,
        probe_batch_size=args.probe_batch_size,
    )
    labels_sub = None
    if probe_train_loader is not None:
        print("  Extracting probe features (train)...")
        Z_ptr, Y_ptr = _extract_probe_features(model, probe_train_loader, device)
        print("  Extracting probe features (val)...")
        Z_pva, Y_pva = _extract_probe_features(model, probe_val_loader, device)
        np.save(save_dir / "labels.npy", np.concatenate([Y_ptr, Y_pva]))
        print("  Fitting linear probes (GPU)...")
        probe_metrics = run_linear_probe(Z_ptr, Y_ptr, Z_pva, Y_pva, device=device)
        metrics.update(probe_metrics)
        print("  Fitting KNN (GPU)...")
        knn_metrics = run_knn(Z_ptr, Y_ptr, Z_pva, Y_pva, ks=(1, 5), device=device)
        metrics.update(knn_metrics)
        print(f"  probe_mean_auc={metrics['probe_mean_auc']:.3f}  "
              f"knn_k1={metrics.get('knn_k1_mean_acc', float('nan')):.3f}  "
              f"knn_k5={metrics.get('knn_k5_mean_acc', float('nan')):.3f}", flush=True)
        labels_sub = Y_pva
        # Save attr names
        with open(save_dir / "attr_names.json", "w") as f:
            json.dump(attr_names, f)
    else:
        print("  [skip] CelebA attributes not available.")
        metrics.update({
            "probe_mean_auc": float("nan"),
            "probe_mean_acc": float("nan"),
            "knn_k1_mean_acc": float("nan"),
            "knn_k5_mean_acc": float("nan"),
        })

    # ── 4. Targeted editing ──────────────────────────────────────────────────
    print(f"\n[4] Targeted editing (zero top-{args.topk_edit} channels)...")
    try:
        imgs_list, orig_recons, edited_recons = run_targeted_editing(
            model, val_loader, device,
            n_images=args.n_edit, topk_channels=args.topk_edit,
        )
        save_editing_plot(imgs_list, orig_recons, edited_recons,
                          save_dir / "editing_samples.png", topk=args.topk_edit)
    except Exception as e:
        print(f"  [warn] Targeted editing failed: {e}")

    # ── Save metrics ─────────────────────────────────────────────────────────
    print("\n[5] Saving metrics...")
    # Remove large list fields before saving json
    json_metrics = {k: v for k, v in metrics.items()
                    if not isinstance(v, list)}
    json_metrics["run_name"] = run_name
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(json_metrics, f, indent=2)

    # CSV
    with open(save_dir / "metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in json_metrics.items():
            writer.writerow([k, v])

    print(f"\nDone. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
