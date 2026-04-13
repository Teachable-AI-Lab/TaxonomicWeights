#!/usr/bin/env python3
"""Hierarchy visualization using GradCAM and Gradient Maximization.

For every taxon model run directory found under ``outputs/``, this script
produces per-stage binary-tree visualizations saved to
``<run_dir>/hierarchy_visualizations/`` containing:

  *_gradcam_full.png   — GradCAM: for each tree node, a 2×2 grid of the top-4
                         activating validation images overlaid with the GradCAM
                         heatmap, arranged as a top-down binary tree.

  *_gradmax_full.png   — Gradient Maximization: for each tree node, the
                         synthetic image that maximally activates that node,
                         arranged as a top-down binary tree.

For hierarchies deeper than four levels both the full (compressed) tree and a
set of four-level subtree panels are also saved.

Usage::

    python src/analyze/visualize_hierarchy.py
    python src/analyze/visualize_hierarchy.py --skip-existing
    python src/analyze/visualize_hierarchy.py --outputs-dir outputs/imagenet
    python src/analyze/visualize_hierarchy.py --dry-run
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.taxon.taxon_ae import TaxonAutoencoder
from src.model.cnn.taxon.multi_taxon_ae import MultiTaxonAutoencoder
from src.model.cnn.taxon.topk_taxon_ae import TopKTaxonAutoencoder
from src.model.cnn.taxon.topk_multi_taxon_ae import TopKMultiTaxonAutoencoder
from src.model.cnn.taxon.bias_taxon_ae import BiasTaxonAutoencoder
from src.model.cnn.taxon.bias_multi_taxon_ae import BiasMultiTaxonAutoencoder
from src.utils.dataloader import CelebAHQLoader, CIFAR10Loader, ImageNet1kHFLoader
from torchvision import transforms

# ── constants ────────────────────────────────────────────────────────────────

TAXON_MODEL_TYPES = (
    "taxon", "multi_taxon",
    "topk_taxon", "topk_multi_taxon",
    "bias_taxon", "bias_multi_taxon",
)

# Pixel size of each tree cell for the full-tree rendering.
# Adaptive based on the number of leaf nodes.
SUBTREE_CELL_PX = 128  # used for all subtree (≤4-level) panels

# Gradient-max optimization settings (overridable via CLI).
GRADMAX_STEPS_DEFAULT = 512
GRADMAX_LR_DEFAULT = 0.02
GRADMAX_TV_DEFAULT = 1e-4
GRADMAX_L2_DEFAULT = 1e-5


# ── model-type detection ─────────────────────────────────────────────────────

def detect_model_type(run_name: str) -> Optional[str]:
    if run_name.startswith("topk_multi_taxon_ae_"):
        return "topk_multi_taxon"
    if run_name.startswith("topk_taxon_ae_"):
        return "topk_taxon"
    if run_name.startswith("bias_multi_taxon_ae_"):
        return "bias_multi_taxon"
    if run_name.startswith("bias_taxon_ae_"):
        return "bias_taxon"
    if run_name.startswith("multi_taxon_ae_"):
        return "multi_taxon"
    if run_name.startswith("taxon_ae_"):
        return "taxon"
    return None


def _is_cifar(cfg: dict) -> bool:
    return cfg.get("data", {}).get("image_size", 256) == 32


def _is_imagenet(cfg: dict) -> bool:
    dc = cfg.get("data", {})
    return dc.get("image_size", 256) == 224 or "imagenet" in dc.get("data_root", "")


def _detect_taxon_variant(state_dict: dict) -> str:
    for k in state_dict:
        if "_steps_since_active" in k:
            return "topk"
        if "_bias" in k and "taxon_stages" in k:
            return "bias"
    for k in state_dict:
        if "multi_taxon_stages" in k:
            return "multi"
    return "vanilla"


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device, config: dict):
    """Load a taxon AE model from checkpoint, auto-detecting variant."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    mc = config.get("model", {})
    state = ckpt.get("model_state", ckpt)

    variant = mc.get("model_variant", _detect_taxon_variant(state))

    is_multi = "multi" in str(variant)
    is_topk = "topk" in str(variant)
    is_bias = "bias" in str(variant)

    common = dict(
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
        temperature=mc.get("temperature", 1.0),
        hard=mc.get("hard", False),
    )

    multi_extra = dict(n_hierarchies=mc.get("n_hierarchies", 3))

    if is_topk and is_multi:
        model = TopKMultiTaxonAutoencoder(**common, **multi_extra)
    elif is_topk:
        model = TopKTaxonAutoencoder(**common)
    elif is_bias and is_multi:
        model = BiasMultiTaxonAutoencoder(**common, **multi_extra)
    elif is_bias:
        model = BiasTaxonAutoencoder(**common)
    elif is_multi:
        model = MultiTaxonAutoencoder(**common, **multi_extra)
    else:
        model = TaxonAutoencoder(**common)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    args = ckpt.get("args", {})
    return model, args


# ── data loading ──────────────────────────────────────────────────────────────

def load_eval_loader(config: dict, batch_size: int = 16, num_workers: int = 4):
    """Return an evaluation DataLoader matching the dataset in config."""
    dc = config.get("data", {})
    data_root = dc.get("data_root", "./data/celeba_hq")
    image_size = dc.get("image_size", 256)
    val_split = dc.get("val_split", 0.05)
    max_val_samples = dc.get("max_val_samples", None)

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if _is_cifar(config):
        loader = CIFAR10Loader(batch_size=batch_size, root=data_root)
        # get_loaders() returns (train_loader, test_loader) for CIFAR
        _, eval_loader = loader.get_loaders()
    elif _is_imagenet(config):
        loader = ImageNet1kHFLoader(
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            transform=tf,
            max_train_samples=1,          # only need val; avoid loading huge train split
            max_val_samples=max_val_samples,
        )
        _, eval_loader = loader.get_loaders()
    else:
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

    return eval_loader, image_size


# ── encoder stage helpers ─────────────────────────────────────────────────────

def _get_stage_list(model) -> Tuple[List, bool]:
    """Return (stage_list, is_multi_taxon)."""
    enc = model.encoder
    if hasattr(enc, "multi_taxon_stages"):
        return list(enc.multi_taxon_stages), True
    if hasattr(enc, "taxon_stages"):
        return list(enc.taxon_stages), False
    raise ValueError("Cannot find taxon stages in model encoder")


def _partial_forward(model, x: torch.Tensor, target_idx: int) -> torch.Tensor:
    """Forward encoder stem + stages 0..target_idx, return target stage output."""
    enc = model.encoder
    stages, _ = _get_stage_list(model)
    feat = enc.stem(x)
    for i, stage in enumerate(stages):
        out, _, _ = stage(feat)
        if i == target_idx:
            return out
        feat = out
    return out  # fallback if target_idx >= len(stages)


def _get_stage_input(model, x: torch.Tensor, target_idx: int) -> torch.Tensor:
    """Return the INPUT to stage target_idx (output of stage target_idx-1 or stem)."""
    enc = model.encoder
    stages, _ = _get_stage_list(model)
    feat = enc.stem(x)
    for i, stage in enumerate(stages):
        if i == target_idx:
            return feat
        out, _, _ = stage(feat)
        feat = out
    return feat


def _node_activation_from_out(
    stage_out: torch.Tensor,
    stage,
    is_multi: bool,
    hier_idx: int,
    depth: int,
    node_i: int,
) -> torch.Tensor:
    """Extract scalar mean activation for a specific tree node from a stage output."""
    if is_multi:
        C_h = stage.hierarchy_out_channels
        k_out = stage_out[:, hier_idx * C_h : (hier_idx + 1) * C_h, :, :]
        lc = stage.hierarchies[hier_idx].layer_channels
    else:
        k_out = stage_out
        lc = stage.layer_channels

    splits = list(torch.split(k_out, list(lc), dim=1))
    return splits[depth][:, node_i, :, :].mean()


def _get_target_layer(stage, is_multi: bool, hier_idx: int) -> nn.Module:
    """Return the last Conv2d in the last residual block of the target hierarchy."""
    if is_multi:
        return stage.hierarchies[hier_idx].blocks[-1].main[4]
    return stage.blocks[-1].main[4]


def _build_chan_to_node(stage, is_multi: bool) -> Dict[Tuple, Tuple]:
    """Map absolute channel index → (hier_idx, depth, node_i) for vectorised scanning."""
    mapping: Dict[Tuple, Tuple] = {}
    if is_multi:
        hier_list = list(stage.hierarchies)
        C_h = stage.hierarchy_out_channels
        for h_idx, hier in enumerate(hier_list):
            offset = h_idx * C_h
            ch_offset = 0
            for depth, n_ch in enumerate(hier.layer_channels):
                for node_i in range(n_ch):
                    mapping[(h_idx, depth, node_i)] = offset + ch_offset + node_i
                ch_offset += n_ch
    else:
        ch_offset = 0
        for depth, n_ch in enumerate(stage.layer_channels):
            for node_i in range(n_ch):
                mapping[(0, depth, node_i)] = ch_offset + node_i
            ch_offset += n_ch
    return mapping  # (hier_idx, depth, node_i) → abs_channel


# ── fixed image collection ────────────────────────────────────────────────────

def collect_fixed_images(
    eval_loader,
    n: int = 4,
) -> List[torch.Tensor]:
    """Return a fixed list of n images (each a (1, C, H, W) CPU tensor) from the
    beginning of the eval loader.  The same images are shown at every tree node so
    that GradCAM highlights are directly comparable across the hierarchy.
    """
    images: List[torch.Tensor] = []
    for batch in eval_loader:
        for img in batch[0].cpu().unbind(0):
            images.append(img.unsqueeze(0))
            if len(images) >= n:
                return images
    return images


# ── top-K image scanning (kept for reference / future use) ───────────────────

def collect_top_k_images(
    model,
    eval_loader,
    device: torch.device,
    stage_idx: int,
    stage,
    is_multi: bool,
    k: int = 4,
    max_batches: int = 30,
) -> Dict[Tuple, List[torch.Tensor]]:
    """Scan eval_loader and return the top-K activating images per node.

    Returns: dict (hier_idx, depth, node_i) → list of ≤K (1,C,H,W) CPU tensors.
    """
    node_to_chan = _build_chan_to_node(stage, is_multi)
    # Invert: abs_channel → node key
    chan_to_node = {v: k for k, v in node_to_chan.items()}

    # min-heaps: node_key → [(act_val, img_global_idx)]
    heaps: Dict[Tuple, list] = {key: [] for key in node_to_chan}
    all_images: List[torch.Tensor] = []  # flat list of (3, H, W) CPU tensors

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= max_batches:
                break
            images = batch[0].to(device)
            B = images.shape[0]

            img_start = len(all_images)
            all_images.extend(images.cpu().unbind(0))  # (3, H, W) each

            stage_out = _partial_forward(model, images, stage_idx)
            # Compute per-channel mean activations: (B, total_ch)
            node_means = stage_out.mean(dim=(2, 3)).cpu().numpy()

            for abs_ch, node_key in chan_to_node.items():
                if abs_ch >= node_means.shape[1]:
                    continue
                col = node_means[:, abs_ch]
                heap = heaps[node_key]
                for i, act_val in enumerate(col):
                    g_idx = img_start + i
                    entry = (float(act_val), g_idx)
                    if len(heap) < k:
                        heapq.heappush(heap, entry)
                    elif float(act_val) > heap[0][0]:
                        heapq.heapreplace(heap, entry)

    # Build result: node_key → list of (1, C, H, W) tensors
    top_k: Dict[Tuple, List[torch.Tensor]] = {}
    for node_key, heap in heaps.items():
        sorted_entries = sorted(heap, key=lambda e: e[0], reverse=True)
        top_k[node_key] = [
            all_images[idx].unsqueeze(0) for _, idx in sorted_entries
        ]

    return top_k


# ── GradCAM ───────────────────────────────────────────────────────────────────

def compute_gradcam(
    model,
    x: torch.Tensor,
    stage_idx: int,
    stage,
    is_multi: bool,
    hier_idx: int,
    depth: int,
    node_i: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GradCAM overlay for one (stage, hierarchy, depth, node) on image x.

    Returns:
        orig_np  — (H, W, 3) uint8 original image in [0, 255].
        overlay  — (H, W, 3) uint8 GradCAM heatmap blended over original.
    """
    model.eval()
    stages, _ = _get_stage_list(model)
    target_layer = _get_target_layer(stage, is_multi, hier_idx)

    feat_maps: List[Optional[torch.Tensor]] = [None]
    gradients: List[Optional[torch.Tensor]] = [None]

    def fwd_hook(module, inp, out):
        feat_maps[0] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients[0] = grad_out[0]

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        x_dev = x.to(device)
        enc = model.encoder
        feat = enc.stem(x_dev)
        for i, s in enumerate(stages):
            out, _, _ = s(feat)
            if i == stage_idx:
                break
            feat = out

        activation = _node_activation_from_out(out, stage, is_multi, hier_idx, depth, node_i)
        model.zero_grad()
        activation.backward(retain_graph=False)
    finally:
        fh.remove()
        bh.remove()

    f = feat_maps[0]
    g = gradients[0]

    if f is None or g is None:
        H, W = x.shape[2], x.shape[3]
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        return blank, blank

    # Global average pool gradients → channel weights
    weights = g.detach().mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
    cam = F.relu((weights * f.detach()).sum(dim=1, keepdim=True))  # (1, 1, h, w)
    cam_np = cam.squeeze().cpu().numpy()

    cam_max = cam_np.max()
    if cam_max > 1e-8:
        cam_np = cam_np / cam_max

    H, W = x.shape[2], x.shape[3]
    cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((W, H), Image.BILINEAR)
    cam_np_full = np.array(cam_pil) / 255.0

    # Original image: denormalize [-1, 1] → [0, 255]
    orig_np = x.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    orig_np = ((orig_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    heatmap = cm.jet(cam_np_full)[:, :, :3]  # (H, W, 3) float [0,1]
    heat_u8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
    overlay = (0.5 * orig_np + 0.5 * heat_u8).clip(0, 255).astype(np.uint8)

    return orig_np, overlay


# ── Gradient maximization ─────────────────────────────────────────────────────

def compute_gradmax(
    model,
    stage_idx: int,
    stage,
    is_multi: bool,
    hier_idx: int,
    depth: int,
    node_i: int,
    device: torch.device,
    image_size: int = 224,
    num_steps: int = GRADMAX_STEPS_DEFAULT,
    lr: float = GRADMAX_LR_DEFAULT,
    tv_weight: float = GRADMAX_TV_DEFAULT,
    l2_weight: float = GRADMAX_L2_DEFAULT,
) -> np.ndarray:
    """Find the image that maximally activates a given tree node via gradient ascent.

    Uses a parametric image (through tanh for smooth clamping) and Adam optimizer
    with total-variation + L2 regularization.

    Returns: (H, W, 3) uint8 numpy array in [0, 255].
    """
    model.eval()
    stages, _ = _get_stage_list(model)

    img = nn.Parameter(torch.randn(1, 3, image_size, image_size, device=device) * 0.1)
    optimizer = torch.optim.Adam([img], lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()
        x = torch.tanh(img)

        enc = model.encoder
        feat = enc.stem(x)
        for i, s in enumerate(stages):
            out, _, _ = s(feat)
            if i == stage_idx:
                break
            feat = out

        activation = _node_activation_from_out(out, stage, is_multi, hier_idx, depth, node_i)

        tv_loss = (
            (x[:, :, 1:, :] - x[:, :, :-1, :]).pow(2).mean()
            + (x[:, :, :, 1:] - x[:, :, :, :-1]).pow(2).mean()
        )
        l2_loss = x.pow(2).mean()

        loss = -activation + tv_weight * tv_loss + l2_weight * l2_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final = torch.tanh(img).squeeze().permute(1, 2, 0).cpu().numpy()
        return ((final * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


# ── cell image builders ───────────────────────────────────────────────────────

def _resize_np(img_np: np.ndarray, h: int, w: int) -> np.ndarray:
    pil = Image.fromarray(img_np.astype(np.uint8))
    return np.array(pil.resize((max(1, w), max(1, h)), Image.LANCZOS))


def build_gradcam_cell(overlays: List[np.ndarray], cell_px: int) -> np.ndarray:
    """Build a 2×2 GradCAM grid cell of size (cell_px, cell_px, 3) uint8."""
    bg = np.full((cell_px, cell_px, 3), 200, dtype=np.uint8)
    b = max(1, cell_px // 32)  # border thickness
    th = (cell_px - 3 * b) // 2
    tw = (cell_px - 3 * b) // 2
    if th < 1 or tw < 1:
        return bg

    positions = [(b, b), (b, b + tw + b), (b + th + b, b), (b + th + b, b + tw + b)]
    for idx, ov in enumerate(overlays[:4]):
        if ov is not None:
            thumb = _resize_np(ov, th, tw)
            r, c = positions[idx]
            bg[r : r + th, c : c + tw] = thumb
    return bg


def build_gradmax_cell(img_np: np.ndarray, cell_px: int) -> np.ndarray:
    """Resize gradient-max image to a square cell of (cell_px, cell_px, 3) uint8."""
    b = max(1, cell_px // 32)
    inner = max(1, cell_px - 2 * b)
    thumb = _resize_np(img_np, inner, inner)
    bg = np.full((cell_px, cell_px, 3), 200, dtype=np.uint8)
    bg[b : b + inner, b : b + inner] = thumb
    return bg


# ── tree rendering ────────────────────────────────────────────────────────────

def _render_tree(
    node_cells: Dict[Tuple[int, int], np.ndarray],
    n_levels: int,
    cell_px: int,
    title: str = "",
) -> np.ndarray:
    """Render a binary tree into a numpy (H, W, 3) uint8 image.

    Args:
        node_cells: dict (depth, node_i) → (cell_px, ?, 3) uint8 array.
        n_levels:   number of depth levels.
        cell_px:    pixel height for each cell row.
        title:      optional title drawn at the top.
    """
    n_leaves = 1 << n_levels          # 2^n_levels leaf slots
    conn_h = max(8, cell_px // 4)     # vertical gap between levels for lines
    row_h = cell_px + conn_h
    title_h = 20 if title else 0

    img_w = n_leaves * cell_px
    img_h = n_levels * row_h + title_h

    canvas = Image.new("RGB", (img_w, img_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    if title:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((4, 2), title, fill=(30, 30, 30), font=font)

    # Precompute center x for each (depth, node) for connecting lines
    node_cx: Dict[Tuple[int, int], int] = {}
    for depth in range(n_levels):
        n_nodes = 1 << (depth + 1)
        node_w_px = img_w // n_nodes
        for node_i in range(n_nodes):
            cx = node_i * node_w_px + node_w_px // 2
            node_cx[(depth, node_i)] = cx

    # Draw connecting lines first (behind cells)
    for depth in range(n_levels - 1):
        n_nodes = 1 << (depth + 1)
        y_parent_bottom = title_h + depth * row_h + cell_px
        y_child_top = title_h + (depth + 1) * row_h
        for node_i in range(n_nodes):
            px = node_cx[(depth, node_i)]
            for child_i in [2 * node_i, 2 * node_i + 1]:
                cx = node_cx.get((depth + 1, child_i), 0)
                draw.line(
                    [(px, y_parent_bottom), (cx, y_child_top)],
                    fill=(160, 160, 160),
                    width=max(1, cell_px // 32),
                )

    # Draw node cells
    try:
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                        max(6, cell_px // 10))
    except Exception:
        small_font = ImageFont.load_default()

    for depth in range(n_levels):
        n_nodes = 1 << (depth + 1)
        node_w_px = img_w // n_nodes
        y_top = title_h + depth * row_h

        for node_i in range(n_nodes):
            x_left = node_i * node_w_px
            cell = node_cells.get((depth, node_i))

            if cell is not None:
                # Keep the cell square (do NOT stretch to node_w_px) and
                # centre it horizontally in the allocated node slot.
                cell_size = max(4, cell_px - 2)
                cell_pil = Image.fromarray(cell)
                cell_pil = cell_pil.resize((cell_size, cell_size), Image.LANCZOS)
                x_paste = x_left + max(0, (node_w_px - cell_size) // 2)
                canvas.paste(cell_pil, (x_paste, y_top + 1))

            # Border
            draw.rectangle(
                [x_left, y_top, x_left + node_w_px - 1, y_top + cell_px - 1],
                outline=(220, 220, 220),
                width=1,
            )
            # Node label (small, bottom-left of cell)
            label = f"d{depth}n{node_i}"
            if cell_px >= 24:
                draw.text((x_left + 2, y_top + cell_px - 12), label,
                          fill=(255, 255, 100), font=small_font)

    return np.array(canvas)


def _render_subtree(
    node_cells: Dict[Tuple[int, int], np.ndarray],
    all_n_levels: int,
    start_depth: int,
    start_node: int,
    n_sub_levels: int,
    cell_px: int,
    title: str = "",
) -> np.ndarray:
    """Render a sub-tree rooted at (start_depth, start_node) for n_sub_levels levels.

    Re-indexes so (start_depth, start_node) becomes (0, 0) in the output tree.
    """
    sub_cells: Dict[Tuple[int, int], np.ndarray] = {}
    for sub_d in range(n_sub_levels):
        real_depth = start_depth + sub_d
        if real_depth >= all_n_levels:
            break
        n_sub_nodes = 1 << sub_d             # nodes at this sub-level
        real_node_base = start_node * n_sub_nodes
        for sub_i in range(n_sub_nodes << 1):
            real_node = real_node_base + (sub_i if sub_d > 0 else 0)
            if sub_d == 0:
                sub_cells[(0, 0)] = node_cells.get((real_depth, start_node))
                break
            cell = node_cells.get((real_depth, real_node))
            sub_cells[(sub_d, sub_i)] = cell

    # Simpler re-indexing: for sub_d, real nodes span [start_node * 2^sub_d,
    # (start_node+1) * 2^sub_d).
    sub_cells2: Dict[Tuple[int, int], np.ndarray] = {}
    for sub_d in range(n_sub_levels):
        real_depth = start_depth + sub_d
        if real_depth >= all_n_levels:
            break
        n_sub_nodes = 1 << sub_d
        node_base = start_node * n_sub_nodes
        for sub_i in range(n_sub_nodes):
            real_node = node_base + sub_i
            sub_cells2[(sub_d, sub_i)] = node_cells.get((real_depth, real_node))

    return _render_tree(sub_cells2, min(n_sub_levels, all_n_levels - start_depth),
                        cell_px, title=title)


# ── per-hierarchy visualization driver ───────────────────────────────────────

def _adaptive_cell_px(n_levels: int) -> int:
    """Return a cell pixel size sized from the level count.

    Each cell holds a 2×2 grid of images.  We target ~96 px per sub-image,
    giving cell_px ≈ 200 px.  For very deep trees (>6 levels) we shrink
    proportionally so the canvas stays under ~25600 px wide.
    """
    n_leaves = 1 << n_levels
    # Desired: each sub-image ≥ 96 px  →  cell_px = 2*96 + borders ≈ 200
    target = 200
    # Clamp so total canvas width stays ≤ 25600 px
    return max(64, min(target, 25600 // max(1, n_leaves)))


def visualize_one_hierarchy(
    model,
    fixed_images: List[torch.Tensor],
    stage_idx: int,
    stage,
    is_multi: bool,
    hier_idx: int,
    n_hier_levels: int,
    image_size: int,
    device: torch.device,
    save_dir: Path,
    prefix: str,
    args: argparse.Namespace,
) -> None:
    """Produce GradCAM + GradMax tree visualizations for one hierarchy."""
    n_levels = n_hier_levels
    cell_px_full = _adaptive_cell_px(n_levels)
    cell_px_sub = SUBTREE_CELL_PX

    print(f"    Hierarchy {hier_idx}: {n_levels} levels, "
          f"{(1 << (n_levels + 1)) - 2} nodes, cell_px={cell_px_full}")

    gradcam_cells: Dict[Tuple[int, int], np.ndarray] = {}
    gradmax_cells: Dict[Tuple[int, int], np.ndarray] = {}

    total_nodes = sum(1 << (d + 1) for d in range(n_levels))
    pbar = tqdm(total=total_nodes, desc=f"      nodes", leave=False)

    for depth in range(n_levels):
        n_nodes = 1 << (depth + 1)
        for node_i in range(n_nodes):
            key = (hier_idx, depth, node_i)

            # --- GradCAM: same fixed images at every node ---
            overlays = []
            for img_t in fixed_images[:4]:
                try:
                    _, ov = compute_gradcam(
                        model, img_t, stage_idx, stage, is_multi,
                        hier_idx, depth, node_i, device,
                    )
                    overlays.append(ov)
                except Exception as e:
                    warnings.warn(f"GradCAM failed for {key}: {e}")
                    overlays.append(None)

            while len(overlays) < 4:
                overlays.append(None)

            gradcam_cells[(depth, node_i)] = build_gradcam_cell(overlays, cell_px_sub)

            # --- Gradient Maximization ---
            try:
                gm_np = compute_gradmax(
                    model, stage_idx, stage, is_multi,
                    hier_idx, depth, node_i, device,
                    image_size=image_size,
                    num_steps=args.gradmax_steps,
                    lr=args.gradmax_lr,
                    tv_weight=args.gradmax_tv,
                    l2_weight=args.gradmax_l2,
                )
                gradmax_cells[(depth, node_i)] = build_gradmax_cell(gm_np, cell_px_sub)
            except Exception as e:
                warnings.warn(f"GradMax failed for {key}: {e}")
                gradmax_cells[(depth, node_i)] = np.full(
                    (cell_px_sub, cell_px_sub, 3), 200, dtype=np.uint8
                )

            pbar.update(1)
    pbar.close()

    # --- Render and save full trees ---
    for mode, cells in (("gradcam", gradcam_cells), ("gradmax", gradmax_cells)):
        # Full tree (possibly compressed)
        full_img = _render_tree(
            cells, n_levels, cell_px=cell_px_full,
            title=f"{prefix} — {mode} (full tree, {n_levels} levels)",
        )
        Image.fromarray(full_img).save(
            save_dir / f"{prefix}_{mode}_full.png"
        )
        print(f"      Saved {prefix}_{mode}_full.png  "
              f"({full_img.shape[1]}×{full_img.shape[0]})")

        # Subtree panels for deep hierarchies
        if n_levels > 4:
            # Top 4 levels
            sub_top = _render_subtree(
                cells, n_levels,
                start_depth=0, start_node=0,
                n_sub_levels=4, cell_px=cell_px_sub,
                title=f"{prefix} — {mode} (top 4 levels)",
            )
            Image.fromarray(sub_top).save(
                save_dir / f"{prefix}_{mode}_top4.png"
            )

            # Leaf subtrees: one per root at depth n_levels-4
            leaf_root_depth = n_levels - 4
            n_leaf_roots = 1 << (leaf_root_depth + 1)
            for lr_i in range(n_leaf_roots):
                sub_leaf = _render_subtree(
                    cells, n_levels,
                    start_depth=leaf_root_depth, start_node=lr_i,
                    n_sub_levels=4, cell_px=cell_px_sub,
                    title=f"{prefix} — {mode} (leaf subtree d{leaf_root_depth}n{lr_i})",
                )
                Image.fromarray(sub_leaf).save(
                    save_dir / f"{prefix}_{mode}_leaf{leaf_root_depth}_{lr_i}.png"
                )

            print(f"      Saved top4 + {n_leaf_roots} leaf subtree panels")


# ── per-run driver ────────────────────────────────────────────────────────────

def process_run(run_dir: Path, args: argparse.Namespace) -> bool:
    """Run hierarchy visualization for one model run directory."""
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"  SKIP: checkpoint not found: {ckpt_path}")
        return False

    save_dir = run_dir / "hierarchy_visualizations"
    if args.skip_existing and save_dir.exists() and any(save_dir.iterdir()):
        print(f"  SKIP: hierarchy_visualizations/ already exists and is non-empty")
        return True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load checkpoint args to reconstruct config ──
    raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    a: dict = raw_ckpt.get("args", {})

    if "cifar" in run_dir.name.lower() or a.get("image_size", 256) == 32:
        data_root = a.get("data_root", "./data")
        image_size = 32
        is_cifar = True
    elif "imagenet" in run_dir.name.lower() or a.get("image_size", 256) == 224:
        data_root = a.get("data_root", "./data")
        image_size = 224
        is_cifar = False
    else:
        data_root = a.get("data_root", "./data/celeba_hq")
        image_size = a.get("image_size", 256)
        is_cifar = False

    # Derive model_variant from directory name so load_model picks the right class.
    _n = run_dir.name
    if _n.startswith("topk_multi_taxon_ae_"):
        _model_variant = "topk_multi"
    elif _n.startswith("bias_multi_taxon_ae_"):
        _model_variant = "bias_multi"
    elif _n.startswith("multi_taxon_ae_"):
        _model_variant = "multi"
    elif _n.startswith("topk_taxon_ae_"):
        _model_variant = "topk"
    elif _n.startswith("bias_taxon_ae_"):
        _model_variant = "bias"
    else:
        _model_variant = "vanilla"

    config = {
        "model": {
            "model_variant":         _model_variant,
            "in_channels":           a.get("in_channels", 3),
            "resnet_variant":        a.get("resnet_variant", "18"),
            "stage_taxonomy_layers": a.get("stage_taxonomy_layers", [5, 6, 7, 8]),
            "stage_strides":         a.get("stage_strides", [1, 2, 2, 2]),
            "stage_blocks":          a.get("stage_blocks", None),
            "temperature":           a.get("temperature", 1.0),
            "hard":                  a.get("hard", False),
            "kernel_size":           a.get("kernel_size", 3),
            "use_stem":              a.get("use_stem", True),
            "stem_channels":         a.get("stem_channels", 64),
            "stem_stride":           a.get("stem_stride", 2),
            "use_stem_maxpool":      a.get("use_stem_maxpool", True),
            "output_activation":     a.get("output_activation", "none"),
            "depth_decay":           a.get("depth_decay", 0.5),
            "n_hierarchies":         a.get("n_hierarchies", 3),
        },
        "data": {
            "data_root":       data_root,
            "image_size":      image_size,
            "batch_size":      args.batch_size,
            "num_workers":     args.num_workers,
            "val_split":       a.get("val_split", 0.05),
            "max_val_samples": a.get("max_val_samples", None),
        },
    }

    print(f"  Loading model from {ckpt_path}")
    model, _ = load_model(ckpt_path, device, config)

    print(f"  Loading eval data  (image_size={image_size})")
    eval_loader, _ = load_eval_loader(
        config, batch_size=args.batch_size, num_workers=args.num_workers
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    stages, is_multi = _get_stage_list(model)

    for stage_idx, stage in enumerate(stages):
        stage_name = f"stage{stage_idx + 1}"
        print(f"\n  ── {stage_name}  (is_multi={is_multi}) ──")

        if is_multi:
            hier_list = list(stage.hierarchies)
        else:
            hier_list = [stage]

        # Collect 4 fixed images once per stage — the same images are shown
        # at every node so GradCAM highlights are directly comparable.
        print(f"    Collecting {args.top_k} fixed images for GradCAM comparison…")
        fixed_images = collect_fixed_images(eval_loader, n=args.top_k)

        for hier_idx, hier in enumerate(hier_list):
            prefix = f"{stage_name}_h{hier_idx}" if is_multi else stage_name

            if args.dry_run:
                print(f"    DRY RUN — would visualize {prefix}")
                continue

            visualize_one_hierarchy(
                model=model,
                fixed_images=fixed_images,
                stage_idx=stage_idx,
                stage=stage,
                is_multi=is_multi,
                hier_idx=hier_idx,
                n_hier_levels=hier.n_taxonomy_layers,
                image_size=image_size,
                device=device,
                save_dir=save_dir,
                prefix=f"{run_dir.name}_{prefix}",
                args=args,
            )

    return True


# ── dispatch loop ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GradCAM + GradMax hierarchy visualizations for all taxon runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--outputs-dir", default="outputs",
                   help="Root outputs directory to scan (default: outputs/)")
    p.add_argument("--run-dir", default=None,
                   help="Path to a single run directory to process (skips --outputs-dir scan).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs that already have hierarchy_visualizations/")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without executing.")

    # Data
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--top-k",        type=int, default=4,
                   help="Number of fixed images used for GradCAM comparison across the tree.")

    # Gradient maximization
    p.add_argument("--gradmax-steps", type=int,   default=GRADMAX_STEPS_DEFAULT)
    p.add_argument("--gradmax-lr",    type=float, default=GRADMAX_LR_DEFAULT)
    p.add_argument("--gradmax-tv",    type=float, default=GRADMAX_TV_DEFAULT)
    p.add_argument("--gradmax-l2",    type=float, default=GRADMAX_L2_DEFAULT)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = ROOT / args.outputs_dir

    if not outputs_dir.exists():
        print(f"ERROR: outputs directory not found: {outputs_dir}")
        sys.exit(1)

    if args.run_dir:
        run_dirs = [ROOT / args.run_dir]
    else:
        run_dirs = sorted(
            d for d in outputs_dir.iterdir()
            if d.is_dir() and not d.name.startswith("comparison")
        )

    # Expand dataset-level grouping folders (e.g. outputs/imagenet/)
    expanded: List[Path] = []
    for d in run_dirs:
        if detect_model_type(d.name) is None:
            expanded.extend(
                sorted(c for c in d.iterdir()
                       if c.is_dir() and not c.name.startswith("comparison"))
            )
        else:
            expanded.append(d)
    run_dirs = expanded

    successes, failures, skipped = [], [], []

    for run_dir in run_dirs:
        model_type = detect_model_type(run_dir.name)
        if model_type not in TAXON_MODEL_TYPES:
            skipped.append(run_dir.name)
            continue

        print(f"\n{'=' * 72}")
        print(f"  {run_dir.name}  [{model_type}]")
        print(f"{'=' * 72}")

        try:
            ok = process_run(run_dir, args)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            ok = False

        (successes if ok else failures).append(run_dir.name)

    print(f"\n{'=' * 72}")
    print(f"Summary: {len(successes)} succeeded, "
          f"{len(failures)} failed, {len(skipped)} skipped")
    if failures:
        print("Failed:")
        for n in failures:
            print(f"  {n}")
        sys.exit(1)


if __name__ == "__main__":
    main()
