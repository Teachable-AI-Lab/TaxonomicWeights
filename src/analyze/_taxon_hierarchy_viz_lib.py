"""Shared helpers for hierarchy-tree visualisations of taxon AE encoders.

Used by:
    src/analyze/visualize_hierarchy_celeba_hq.py
    src/analyze/visualize_hierarchy_imagenet.py

Three figure types per (run, hierarchy):
  * ``single/`` — one fixed image at every node, overlaid with that node's
    GradCAM heatmap.  Inactive nodes (top activation <= eps) are blank.
  * ``topk/``   — a 2x2 grid of the top-K activating validation images per
    node, each overlaid with its own GradCAM.  Inactive nodes are blank.
  * ``prefix_recon/`` — for each of a few input images, follow the image's
    activation path through the tree and at each path node render the
    decoder's reconstruction using ONLY the channels along the path so far
    (path-restricted prefix reconstruction).  Off-path nodes are blank.

Disk caching of raw .npy intermediates (under ``cache/``) lets re-runs
re-render figures without re-extracting activations or running GradCAM.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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

from src.analyze.visualize_hierarchy import (  # noqa: E402
    _adaptive_cell_px,
    build_gradcam_cell,
)

SUBTREE_CELL_PX = 128
ACTIVE_EPS = 1e-6  # node is "active" if its top-1 activation magnitude > this


# ── stage iteration ──────────────────────────────────────────────────────────

@dataclass
class HierarchyMeta:
    """Describes one (encoder-stage, hierarchy) tree to visualise."""
    stage_name: str            # e.g. "stage1" or "bottleneck"
    stage_obj: nn.Module       # the stage (or bottleneck) module
    is_multi: bool             # multi-taxon (per-stage hierarchies grouped)
    hier_idx: int              # 0 if not multi, else hierarchy index
    n_levels: int              # depth of the binary tree
    layer_channels: Tuple[int, ...]  # per-depth node count
    n_nodes: int               # total = sum(layer_channels)
    is_bottleneck: bool        # uses encoder.plain_stages + bottleneck_stage
    target_stage_idx: int      # used by multi-stage partial forward
    label: str                 # short label for filenames, e.g. "stage1_h0"


def _has(obj, name) -> bool:
    return hasattr(obj, name)


def _hier_layer_channels(stage, is_multi: bool, hier_idx: int) -> Sequence[int]:
    if is_multi:
        return stage.hierarchies[hier_idx].layer_channels
    return stage.layer_channels


def _hier_n_levels(stage, is_multi: bool, hier_idx: int) -> int:
    if is_multi:
        return stage.hierarchies[hier_idx].n_taxonomy_layers
    return stage.n_taxonomy_layers


def discover_hierarchies(model: nn.Module) -> List[HierarchyMeta]:
    """Return a HierarchyMeta for every (stage, hierarchy) tree in the model."""
    enc = model.encoder
    metas: List[HierarchyMeta] = []

    if _has(enc, "bottleneck_stage"):
        stage = enc.bottleneck_stage
        is_multi = _has(stage, "hierarchies")
        n_hier = len(stage.hierarchies) if is_multi else 1
        for h in range(n_hier):
            n_levels = _hier_n_levels(stage, is_multi, h)
            lc = tuple(_hier_layer_channels(stage, is_multi, h))
            metas.append(HierarchyMeta(
                stage_name="bottleneck",
                stage_obj=stage,
                is_multi=is_multi,
                hier_idx=h,
                n_levels=n_levels,
                layer_channels=lc,
                n_nodes=sum(lc),
                is_bottleneck=True,
                target_stage_idx=0,
                label=f"bottleneck_h{h}" if is_multi else "bottleneck",
            ))
        return metas

    if _has(enc, "multi_taxon_stages"):
        stages = list(enc.multi_taxon_stages)
        is_multi = True
    elif _has(enc, "taxon_stages"):
        stages = list(enc.taxon_stages)
        is_multi = False
    else:
        raise RuntimeError(
            "discover_hierarchies: encoder has no taxon_stages / "
            "multi_taxon_stages / bottleneck_stage attribute"
        )

    for s_idx, stage in enumerate(stages):
        n_hier = len(stage.hierarchies) if is_multi else 1
        for h in range(n_hier):
            n_levels = _hier_n_levels(stage, is_multi, h)
            lc = tuple(_hier_layer_channels(stage, is_multi, h))
            label = f"stage{s_idx + 1}" + (f"_h{h}" if is_multi else "")
            metas.append(HierarchyMeta(
                stage_name=f"stage{s_idx + 1}",
                stage_obj=stage,
                is_multi=is_multi,
                hier_idx=h,
                n_levels=n_levels,
                layer_channels=lc,
                n_nodes=sum(lc),
                is_bottleneck=False,
                target_stage_idx=s_idx,
                label=label,
            ))
    return metas


# ── stage forward + activation extraction ────────────────────────────────────

def _forward_to_target_stage(
    model: nn.Module, x: torch.Tensor, meta: HierarchyMeta
) -> torch.Tensor:
    """Forward through stem + (plain_stages|preceding_taxon_stages) + target stage.

    Returns the raw (concatenated) output of the target taxon stage.
    """
    enc = model.encoder
    feat = enc.stem(x)

    if meta.is_bottleneck:
        for ps in enc.plain_stages:
            feat = ps(feat)
        out = enc.bottleneck_stage(feat)
        if isinstance(out, tuple):
            out = out[0]
        return out

    stages = (
        list(enc.multi_taxon_stages)
        if _has(enc, "multi_taxon_stages")
        else list(enc.taxon_stages)
    )
    out = feat
    for i, stage in enumerate(stages):
        out, _, _ = stage(feat)
        if i == meta.target_stage_idx:
            return out
        feat = out
    return out


def _slice_hierarchy_out(stage_out: torch.Tensor, meta: HierarchyMeta) -> torch.Tensor:
    """Slice the stage-output channels belonging to this hierarchy."""
    if meta.is_multi:
        C_h = meta.stage_obj.hierarchy_out_channels
        return stage_out[:, meta.hier_idx * C_h : (meta.hier_idx + 1) * C_h, :, :]
    return stage_out


def _node_columns(stage_out: torch.Tensor, meta: HierarchyMeta) -> torch.Tensor:
    """``(B, n_nodes)`` mean-spatial activations, depth-major / node-major order."""
    return _slice_hierarchy_out(stage_out, meta).mean(dim=(2, 3))


def _node_targets(stage_out: torch.Tensor, meta: HierarchyMeta) -> torch.Tensor:
    """``(B, n_nodes, H, W)`` per-node spatial maps (no pooling)."""
    return _slice_hierarchy_out(stage_out, meta)


def _get_target_layer(meta: HierarchyMeta) -> nn.Module:
    """Last Conv2d in the last residual block of the target hierarchy."""
    if meta.is_multi:
        return meta.stage_obj.hierarchies[meta.hier_idx].blocks[-1].main[4]
    return meta.stage_obj.blocks[-1].main[4]


# ── per-image, per-node activation pass ──────────────────────────────────────

@torch.no_grad()
def collect_node_activations(
    model: nn.Module,
    val_loader,
    metas: List[HierarchyMeta],
    device: torch.device,
    max_samples: int,
) -> Tuple[Dict[str, np.ndarray], List[torch.Tensor]]:
    """Stream val_loader and compute per-image per-node mean activations."""
    model.eval()
    per_meta: Dict[str, List[np.ndarray]] = {m.label: [] for m in metas}
    images_cpu: List[torch.Tensor] = []
    n_seen = 0

    pbar = tqdm(total=max_samples, desc="    activations", leave=False)
    for batch in val_loader:
        if n_seen >= max_samples:
            break
        x = batch[0]
        take = min(x.shape[0], max_samples - n_seen)
        x = x[:take].to(device, non_blocking=True)

        key_to_out: Dict[Tuple[bool, int], torch.Tensor] = {}
        for m in metas:
            key = (m.is_bottleneck, m.target_stage_idx)
            if key not in key_to_out:
                key_to_out[key] = _forward_to_target_stage(model, x, m)
            stage_out = key_to_out[key]
            cols = _node_columns(stage_out, m).detach().cpu().float().numpy()
            per_meta[m.label].append(cols)

        images_cpu.extend(x.detach().cpu().unbind(0))
        n_seen += take
        pbar.update(take)
    pbar.close()

    out: Dict[str, np.ndarray] = {
        label: np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 0), dtype=np.float32)
        for label, arrs in per_meta.items()
    }
    return out, images_cpu


def select_topk_per_node(
    acts: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Given ``(N, n_nodes)``, return (topk_indices, topk_acts) of shape (n_nodes, K)."""
    N, _ = acts.shape
    k_eff = min(k, N)
    part = np.argpartition(-acts, kth=k_eff - 1, axis=0)[:k_eff, :]
    rows = part.T
    vals = np.take_along_axis(acts.T, rows, axis=1)
    order = np.argsort(-vals, axis=1)
    return (
        np.take_along_axis(rows, order, axis=1).astype(np.int64),
        np.take_along_axis(vals, order, axis=1).astype(np.float32),
    )


# ── path-on-tree extraction (greedy constrained argmax) ──────────────────────

def _depth_offsets(layer_channels: Sequence[int]) -> List[int]:
    offsets = [0]
    for c in layer_channels[:-1]:
        offsets.append(offsets[-1] + int(c))
    return offsets


def constrained_path_for_image(
    acts_row: np.ndarray, meta: HierarchyMeta
) -> List[int]:
    """Return the path ``[node_i_at_d0, node_i_at_d1, ...]`` of length n_levels.

    At depth 0 picks argmax over the 2 root children.  At depth d>0 picks
    argmax over the 2 children of the chosen parent at depth d-1, so the
    returned path is a real root-to-leaf walk in the binary tree.
    """
    offs = _depth_offsets(meta.layer_channels)
    path: List[int] = []
    for d in range(meta.n_levels):
        if d == 0:
            a, b = float(acts_row[offs[0]]), float(acts_row[offs[0] + 1])
            path.append(0 if a >= b else 1)
        else:
            parent = path[-1]
            a = float(acts_row[offs[d] + 2 * parent])
            b = float(acts_row[offs[d] + 2 * parent + 1])
            path.append(2 * parent + (0 if a >= b else 1))
    return path


# ── GradCAM (per node) ───────────────────────────────────────────────────────

def _gradcam_for_image_one_node(
    model: nn.Module,
    image: torch.Tensor,
    meta: HierarchyMeta,
    node_idx: int,
    device: torch.device,
) -> np.ndarray:
    """Compute a GradCAM heatmap for one node only on one image."""
    model.eval()
    target_layer = _get_target_layer(meta)
    H, W = image.shape[-2], image.shape[-1]

    feat_cache: List[Optional[torch.Tensor]] = [None]

    def fwd_hook(module, inp, out):
        feat_cache[0] = out

    fh = target_layer.register_forward_hook(fwd_hook)
    cam_out = np.zeros((H, W), dtype=np.float32)
    try:
        x = image.to(device)
        stage_out = _forward_to_target_stage(model, x, meta)
        targets = _node_targets(stage_out, meta)
        feat = feat_cache[0]
        if feat is None:
            return cam_out
        scalar = targets[:, node_idx].mean()
        grads = torch.autograd.grad(
            outputs=scalar, inputs=feat, retain_graph=False, create_graph=False,
        )[0]
        w = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * feat).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().detach().cpu().numpy()
        mx = float(cam_np.max())
        if mx > 1e-8:
            cam_np = cam_np / mx
        cam_out = cam_np.astype(np.float32)
    finally:
        fh.remove()
    return cam_out


def _gradcam_for_image_node_subset(
    model: nn.Module,
    image: torch.Tensor,
    meta: HierarchyMeta,
    node_indices: Sequence[int],
    device: torch.device,
) -> np.ndarray:
    """Compute GradCAM heatmaps for a SUBSET of nodes on one image.

    Returns ``(len(node_indices), H, W)`` float32 in [0, 1].  Reuses one
    forward pass + one feature-map hook for all nodes in the subset.
    """
    model.eval()
    target_layer = _get_target_layer(meta)
    H, W = image.shape[-2], image.shape[-1]
    n_sub = len(node_indices)
    cams = np.zeros((n_sub, H, W), dtype=np.float32)
    if n_sub == 0:
        return cams

    feat_cache: List[Optional[torch.Tensor]] = [None]

    def fwd_hook(module, inp, out):
        feat_cache[0] = out

    fh = target_layer.register_forward_hook(fwd_hook)
    try:
        x = image.to(device)
        stage_out = _forward_to_target_stage(model, x, meta)
        targets = _node_targets(stage_out, meta)
        feat = feat_cache[0]
        if feat is None:
            return cams
        scalars = targets.mean(dim=(2, 3)).squeeze(0)
        for j, n in enumerate(node_indices):
            grads = torch.autograd.grad(
                outputs=scalars[n],
                inputs=feat,
                retain_graph=(j < n_sub - 1),
                create_graph=False,
                only_inputs=True,
            )[0]
            w = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * feat).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
            cam_np = cam.squeeze().detach().cpu().numpy()
            mx = float(cam_np.max())
            if mx > 1e-8:
                cam_np = cam_np / mx
            cams[j] = cam_np.astype(np.float32)
    finally:
        fh.remove()
    return cams


# ── path-restricted prefix reconstruction ────────────────────────────────────

@torch.no_grad()
def path_prefix_reconstructions(
    model: nn.Module,
    image: torch.Tensor,
    meta: HierarchyMeta,
    path: List[int],
    device: torch.device,
) -> List[np.ndarray]:
    """For each prefix depth d, decode the latent with all channels zeroed
    except those along the chosen path at depths 0..d.

    Only supported for single-hierarchy bottleneck models that expose
    ``encode`` / ``decode`` / ``_apply_output_activation``.

    Returns a list of length ``meta.n_levels`` of ``(H, W, 3)`` uint8 arrays.
    """
    if meta.is_multi:
        return []
    if not (hasattr(model, "encode") and hasattr(model, "decode")):
        return []

    x = image.to(device)
    H, W = x.shape[-2], x.shape[-1]

    enc_out = model.encode(x)
    z = enc_out[0] if isinstance(enc_out, tuple) else enc_out

    offs = _depth_offsets(meta.layer_channels)
    out: List[np.ndarray] = []
    for d in range(meta.n_levels):
        mask = torch.zeros(1, z.size(1), 1, 1, device=z.device, dtype=z.dtype)
        for dprime in range(d + 1):
            ch = offs[dprime] + path[dprime]
            mask[:, ch] = 1.0
        z_masked = z * mask
        try:
            recon, _ = model.decode(z_masked, output_size=(H, W))
        except TypeError:
            recon = model.decode(z_masked)
            if isinstance(recon, tuple):
                recon = recon[0]
        if hasattr(model, "_apply_output_activation"):
            recon = model._apply_output_activation(recon)
        out.append(_tensor_to_uint8(recon[0]))
    return out


# ── helpers for converting tensors -> displayable images ─────────────────────

def _tensor_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """``(C,H,W)`` or ``(1,C,H,W)`` in [-1,1] -> ``(H,W,3)`` uint8."""
    if img_t.dim() == 4:
        img_t = img_t.squeeze(0)
    arr = img_t.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * 0.5 + 0.5) * 255.0
    return arr.clip(0, 255).astype(np.uint8)


def _overlay(orig_uint8: np.ndarray, cam01: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a [0,1] cam with an HWCu8 image using a jet colormap."""
    if cam01.shape[:2] != orig_uint8.shape[:2]:
        cam01 = np.array(
            Image.fromarray((cam01 * 255).clip(0, 255).astype(np.uint8))
            .resize((orig_uint8.shape[1], orig_uint8.shape[0]), Image.BILINEAR)
        ) / 255.0
    heat = cm.jet(cam01)[:, :, :3]
    heat_u8 = (heat * 255).clip(0, 255).astype(np.uint8)
    return ((1 - alpha) * orig_uint8 + alpha * heat_u8).clip(0, 255).astype(np.uint8)


def _wrap_in_cell(img_uint8: np.ndarray, cell_px: int) -> np.ndarray:
    """Resize an HWC u8 image into a ``cell_px`` square with a thin border."""
    pil = Image.fromarray(img_uint8).resize((cell_px - 4, cell_px - 4), Image.LANCZOS)
    bg = np.full((cell_px, cell_px, 3), 200, dtype=np.uint8)
    bg[2:cell_px - 2, 2:cell_px - 2] = np.array(pil)
    return bg


# ── tree-cell builders for each figure type ──────────────────────────────────

def _node_iter(n_levels: int) -> Iterator[Tuple[int, int, int]]:
    """Yield ``(node_idx, depth, node_i)`` in depth-then-node order."""
    n = 0
    for depth in range(n_levels):
        for node_i in range(1 << (depth + 1)):
            yield n, depth, node_i
            n += 1


def build_single_image_tree_cells(
    image_uint8: np.ndarray,
    cams: np.ndarray,
    active_mask: np.ndarray,
    meta: HierarchyMeta,
    cell_px: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    """One image at every active node, overlaid with that node's GradCAM.

    Inactive nodes are omitted (rendered as a blank slot by the renderer).
    """
    cells: Dict[Tuple[int, int], np.ndarray] = {}
    for n, d, ni in _node_iter(meta.n_levels):
        if not active_mask[n]:
            continue
        cells[(d, ni)] = _wrap_in_cell(_overlay(image_uint8, cams[n]), cell_px)
    return cells


def build_topk_tree_cells(
    images_per_node: np.ndarray,
    cams_per_node: np.ndarray,
    active_mask: np.ndarray,
    meta: HierarchyMeta,
    cell_px: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    """K=4 grid per active node; tiles are top-K images with their CAMs."""
    cells: Dict[Tuple[int, int], np.ndarray] = {}
    K = images_per_node.shape[1]
    for n, d, ni in _node_iter(meta.n_levels):
        if not active_mask[n]:
            continue
        overlays: List[Optional[np.ndarray]] = []
        for k in range(K):
            img_t = torch.from_numpy(images_per_node[n, k])
            overlays.append(_overlay(_tensor_to_uint8(img_t), cams_per_node[n, k]))
        while len(overlays) < 4:
            overlays.append(None)
        cells[(d, ni)] = build_gradcam_cell(overlays, cell_px)
    return cells


def build_path_recon_tree_cells(
    path: List[int],
    recons_uint8: List[np.ndarray],
    meta: HierarchyMeta,
    cell_px: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    """For one image's path, render reconstructions only at each path node."""
    cells: Dict[Tuple[int, int], np.ndarray] = {}
    for d, node_i in enumerate(path):
        if d >= len(recons_uint8):
            break
        cells[(d, node_i)] = _wrap_in_cell(recons_uint8[d], cell_px)
    return cells


# ── tree rendering ───────────────────────────────────────────────────────────

def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _render_full_tree(
    cells: Dict[Tuple[int, int], np.ndarray],
    n_levels: int,
    cell_px: int,
    title: str = "",
    layer_channels: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Render a binary tree where depth d has ``layer_channels[d]`` nodes.

    Default ``layer_channels[d] = 2^(d+1)`` (the standard taxon binary tree).
    """
    if layer_channels is None:
        layer_channels = [1 << (d + 1) for d in range(n_levels)]

    n_leaves = layer_channels[-1]
    conn_h = max(8, cell_px // 4)
    row_h = cell_px + conn_h
    title_h = 20 if title else 0

    img_w = n_leaves * cell_px
    img_h = n_levels * row_h + title_h

    canvas = Image.new("RGB", (img_w, img_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((4, 2), title, fill=(30, 30, 30), font=_font(14))

    node_cx: Dict[Tuple[int, int], int] = {}
    for depth in range(n_levels):
        n_nodes = layer_channels[depth]
        node_w_px = img_w // n_nodes
        for node_i in range(n_nodes):
            node_cx[(depth, node_i)] = node_i * node_w_px + node_w_px // 2

    for depth in range(n_levels - 1):
        n_parent = layer_channels[depth]
        y_parent_bottom = title_h + depth * row_h + cell_px
        y_child_top = title_h + (depth + 1) * row_h
        for node_i in range(n_parent):
            px = node_cx[(depth, node_i)]
            for child_i in (2 * node_i, 2 * node_i + 1):
                if (depth + 1, child_i) not in node_cx:
                    continue
                cx = node_cx[(depth + 1, child_i)]
                draw.line(
                    [(px, y_parent_bottom), (cx, y_child_top)],
                    fill=(160, 160, 160),
                    width=max(1, cell_px // 32),
                )

    small_font = _font(max(6, cell_px // 10))
    for depth in range(n_levels):
        n_nodes = layer_channels[depth]
        node_w_px = img_w // n_nodes
        y_top = title_h + depth * row_h
        for node_i in range(n_nodes):
            x_left = node_i * node_w_px
            cell = cells.get((depth, node_i))
            if cell is not None:
                cell_size = max(4, cell_px - 2)
                cell_pil = Image.fromarray(cell).resize(
                    (cell_size, cell_size), Image.LANCZOS,
                )
                x_paste = x_left + max(0, (node_w_px - cell_size) // 2)
                canvas.paste(cell_pil, (x_paste, y_top + 1))
            draw.rectangle(
                [x_left, y_top, x_left + node_w_px - 1, y_top + cell_px - 1],
                outline=(220, 220, 220), width=1,
            )
            if cell_px >= 24:
                draw.text(
                    (x_left + 2, y_top + cell_px - 12),
                    f"d{depth}n{node_i}",
                    fill=(255, 255, 100), font=small_font,
                )

    return np.array(canvas)


def _render_subtree_correct(
    cells: Dict[Tuple[int, int], np.ndarray],
    start_depth: int,
    start_node: int,
    n_sub_levels: int,
    cell_px: int,
    title: str = "",
) -> np.ndarray:
    """Render the subtree rooted at ``(start_depth, start_node)``.

    Sub-tree level ``sub_d`` contains ``2^sub_d`` nodes (root has 1 node at
    sub_d=0, doubling per level).  Cells are pulled by their ORIGINAL
    ``(depth, node_i)`` keys.
    """
    sub_layer_channels = [1 << sub_d for sub_d in range(n_sub_levels)]
    sub_cells: Dict[Tuple[int, int], np.ndarray] = {}
    for sub_d in range(n_sub_levels):
        n_sub_nodes = 1 << sub_d
        node_base = start_node * n_sub_nodes
        for sub_i in range(n_sub_nodes):
            real_depth = start_depth + sub_d
            real_node = node_base + sub_i
            cell = cells.get((real_depth, real_node))
            if cell is not None:
                sub_cells[(sub_d, sub_i)] = cell
    return _render_full_tree(
        sub_cells, n_sub_levels, cell_px, title=title,
        layer_channels=sub_layer_channels,
    )


def _subtree_has_cells(
    cells: Dict[Tuple[int, int], np.ndarray],
    start_depth: int,
    start_node: int,
    n_sub_levels: int,
) -> bool:
    """True iff at least one (depth, node) inside the subtree appears in cells."""
    for sub_d in range(n_sub_levels):
        n_sub_nodes = 1 << sub_d
        node_base = start_node * n_sub_nodes
        for sub_i in range(n_sub_nodes):
            if (start_depth + sub_d, node_base + sub_i) in cells:
                return True
    return False


def render_and_save_tree(
    cells: Dict[Tuple[int, int], np.ndarray],
    meta: HierarchyMeta,
    out_dir: Path,
    prefix: str,
    title: str,
) -> None:
    """Save full + (if depth > 4) top-4 + leaf-subtree panels under ``out_dir``.

    Skips any panel whose covered subtree has zero populated cells (avoids
    the all-blank "white tree" outputs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not cells:
        return
    n_levels = meta.n_levels
    cell_px_full = _adaptive_cell_px(n_levels)

    full = _render_full_tree(
        cells, n_levels, cell_px=cell_px_full,
        title=f"{title} (full, {n_levels} levels)",
    )
    Image.fromarray(full).save(out_dir / f"{prefix}_full.png")

    if n_levels > 4:
        if _subtree_has_cells(cells, 0, 0, 4):
            sub_top = _render_subtree_correct(
                cells, start_depth=0, start_node=0,
                n_sub_levels=4, cell_px=SUBTREE_CELL_PX,
                title=f"{title} (top 4 levels)",
            )
            Image.fromarray(sub_top).save(out_dir / f"{prefix}_top4.png")

        leaf_root_depth = n_levels - 4
        n_leaf_roots = 1 << leaf_root_depth
        for lr in range(n_leaf_roots):
            if not _subtree_has_cells(cells, leaf_root_depth, lr, 4):
                continue
            sub = _render_subtree_correct(
                cells, start_depth=leaf_root_depth, start_node=lr,
                n_sub_levels=4, cell_px=SUBTREE_CELL_PX,
                title=f"{title} (leaf d{leaf_root_depth}n{lr})",
            )
            Image.fromarray(sub).save(
                out_dir / f"{prefix}_leaf{leaf_root_depth}_{lr}.png"
            )


# ── caching ──────────────────────────────────────────────────────────────────

def cache_paths(cache_dir: Path, label: str) -> Dict[str, Path]:
    return {
        "node_acts":     cache_dir / f"{label}_node_acts.npy",
        "topk_idx":      cache_dir / f"{label}_topk_indices.npy",
        "topk_acts":     cache_dir / f"{label}_topk_acts.npy",
        "topk_images":   cache_dir / f"{label}_topk_images.npy",
        "topk_gradcam":  cache_dir / f"{label}_topk_gradcam.npy",
        "active_mask":   cache_dir / f"{label}_active_mask.npy",
        "fixed_gradcam": cache_dir / f"{label}_fixed_gradcam.npy",
        "fixed_paths":   cache_dir / f"{label}_fixed_paths.npy",
        "prefix_recons": cache_dir / f"{label}_prefix_recons.npy",
    }


def cache_complete(paths: Dict[str, Path]) -> bool:
    keys = ("topk_idx", "topk_acts", "topk_images", "topk_gradcam",
            "active_mask", "fixed_gradcam")
    return all(paths[k].exists() for k in keys)


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def quantize_cam_uint8(cam: np.ndarray) -> np.ndarray:
    return (cam.clip(0, 1) * 255).astype(np.uint8)


def dequantize_cam_uint8(q: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) / 255.0


# ── high-level driver: process one (run, hierarchy) ──────────────────────────

def process_one_run(
    *,
    model: nn.Module,
    val_loader,
    device: torch.device,
    out_dir: Path,
    n_fixed: int,
    top_k: int,
    max_samples: int,
    force_recompute: bool,
    short_label: str,
    skip_prefix_recon: bool = False,
    active_eps: float = ACTIVE_EPS,
) -> None:
    """Top-level driver: handle ALL hierarchies of one model run.

    Output layout under ``out_dir``::
        single/        single_img{i}_<label>_{full,top4,leaf*}.png
        topk/          topk_<label>_{full,top4,leaf*}.png
        prefix_recon/  prefix_recon_img{i}_<label>_{full,top4,leaf*}.png
        cache/         *.npy intermediates
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    single_dir = out_dir / "single"
    topk_dir = out_dir / "topk"
    prefix_dir = out_dir / "prefix_recon"
    for d in (cache_dir, single_dir, topk_dir, prefix_dir):
        d.mkdir(parents=True, exist_ok=True)

    metas = discover_hierarchies(model)
    if not metas:
        print(f"  [{short_label}] no taxon hierarchies discovered; skipping")
        return

    fixed_idx_path = cache_dir / "fixed_indices.npy"
    fixed_img_path = cache_dir / "fixed_images.npy"

    need_pass = force_recompute or not all(
        cache_complete(cache_paths(cache_dir, m.label)) for m in metas
    ) or not (fixed_idx_path.exists() and fixed_img_path.exists())

    if need_pass:
        print(f"  [{short_label}] activation pass ({max_samples} samples) "
              f"over {len(metas)} hierarchy/-ies …")
        acts, imgs = collect_node_activations(
            model, val_loader, metas, device, max_samples=max_samples,
        )
        N = len(imgs)
        if N == 0:
            print(f"  [{short_label}] WARNING: no samples returned by val_loader")
            return
        n_fixed_eff = min(n_fixed, N)
        fixed_local_idx = np.linspace(0, N - 1, num=n_fixed_eff, dtype=np.int64)
        fixed_imgs = torch.stack([imgs[i] for i in fixed_local_idx], dim=0)
        save_npy(fixed_idx_path, fixed_local_idx)
        save_npy(fixed_img_path, fixed_imgs.numpy().astype(np.float32))

        for m in metas:
            paths = cache_paths(cache_dir, m.label)
            save_npy(paths["node_acts"], acts[m.label])
            top_idx, top_v = select_topk_per_node(acts[m.label], k=top_k)
            save_npy(paths["topk_idx"], top_idx)
            save_npy(paths["topk_acts"], top_v)

            # Active mask: a node is "active" iff its top-1 activation magnitude
            # exceeds eps.  Inactive nodes are skipped for GradCAM and rendered
            # as blank cells (huge compute saving on sparse topk models).
            active_mask = np.abs(top_v[:, 0]) > active_eps
            save_npy(paths["active_mask"], active_mask.astype(np.bool_))
            n_active = int(active_mask.sum())
            active_node_indices = np.flatnonzero(active_mask).tolist()

            topk_imgs = np.stack([
                np.stack(
                    [imgs[idx].numpy().astype(np.float32) for idx in row],
                    axis=0,
                )
                for row in top_idx
            ], axis=0)
            save_npy(paths["topk_images"], topk_imgs)

            print(f"    [{m.label}] {n_active}/{m.n_nodes} active nodes; "
                  f"GradCAMs: {n_fixed_eff} fixed × {n_active} nodes "
                  f"+ {n_active * top_idx.shape[1]} topk passes")
            H, W = fixed_imgs.shape[-2], fixed_imgs.shape[-1]

            # Fixed-image GradCAMs: one forward+backward per fixed image; only
            # compute heatmaps for active nodes.
            fixed_cams = np.zeros((n_fixed_eff, m.n_nodes, H, W), dtype=np.uint8)
            for fi in tqdm(range(n_fixed_eff), desc="    fixed", leave=False):
                cam_subset = _gradcam_for_image_node_subset(
                    model, fixed_imgs[fi:fi + 1], m, active_node_indices, device,
                )
                for j, n in enumerate(active_node_indices):
                    fixed_cams[fi, n] = quantize_cam_uint8(cam_subset[j])
            save_npy(paths["fixed_gradcam"], fixed_cams)

            K = top_idx.shape[1]
            topk_cams = np.zeros((m.n_nodes, K, H, W), dtype=np.uint8)
            for n in tqdm(active_node_indices, desc="    topk", leave=False):
                for k in range(K):
                    img_t = torch.from_numpy(topk_imgs[n, k]).unsqueeze(0)
                    cam = _gradcam_for_image_one_node(model, img_t, m, n, device)
                    topk_cams[n, k] = quantize_cam_uint8(cam)
            save_npy(paths["topk_gradcam"], topk_cams)

            # Prefix-recon: only for single-hierarchy models with encode/decode.
            if (
                not skip_prefix_recon
                and not m.is_multi
                and hasattr(model, "encode")
                and hasattr(model, "decode")
            ):
                paths_per_img = np.zeros((n_fixed_eff, m.n_levels), dtype=np.int64)
                recons = np.zeros(
                    (n_fixed_eff, m.n_levels, 3, H, W), dtype=np.float32,
                )
                for fi in tqdm(range(n_fixed_eff),
                               desc="    prefix_recon", leave=False):
                    acts_row = acts[m.label][fixed_local_idx[fi]]
                    path = constrained_path_for_image(acts_row, m)
                    paths_per_img[fi] = np.asarray(path, dtype=np.int64)
                    img_t = fixed_imgs[fi:fi + 1]
                    recon_list = path_prefix_reconstructions(
                        model, img_t, m, path, device,
                    )
                    for d, r in enumerate(recon_list):
                        recons[fi, d] = (
                            (r.astype(np.float32) / 255.0 - 0.5) * 2.0
                        ).transpose(2, 0, 1)
                save_npy(paths["fixed_paths"], paths_per_img)
                save_npy(paths["prefix_recons"], recons)

    # ─── Re-render figures from cache ────────────────────────────────────────
    fixed_imgs_np = np.load(fixed_img_path)
    n_fixed_eff = fixed_imgs_np.shape[0]

    for m in metas:
        paths = cache_paths(cache_dir, m.label)
        if not cache_complete(paths):
            print(f"  [{short_label}] {m.label}: cache incomplete; skipping render")
            continue

        active_mask = np.load(paths["active_mask"])
        if active_mask.sum() == 0:
            print(f"  [{short_label}] {m.label}: 0 active nodes — skipping figures")
            continue

        fixed_cams = dequantize_cam_uint8(np.load(paths["fixed_gradcam"]))
        topk_imgs = np.load(paths["topk_images"])
        topk_cams = dequantize_cam_uint8(np.load(paths["topk_gradcam"]))
        cell_px_full = _adaptive_cell_px(m.n_levels)

        # Type A — single image per node, one figure per fixed image
        for fi in range(n_fixed_eff):
            img_u8 = _tensor_to_uint8(torch.from_numpy(fixed_imgs_np[fi]))
            cells = build_single_image_tree_cells(
                img_u8, fixed_cams[fi], active_mask, m, cell_px=cell_px_full,
            )
            render_and_save_tree(
                cells, m, single_dir,
                prefix=f"single_img{fi}_{m.label}",
                title=f"{short_label} {m.label} single-img-{fi}",
            )

        # Type B — top-K per node
        cells = build_topk_tree_cells(
            topk_imgs, topk_cams, active_mask, m, cell_px=cell_px_full,
        )
        render_and_save_tree(
            cells, m, topk_dir,
            prefix=f"topk_{m.label}",
            title=f"{short_label} {m.label} top-K",
        )

        # Type C — prefix recon (path-restricted) per fixed image
        if (
            not skip_prefix_recon
            and paths["prefix_recons"].exists()
            and paths["fixed_paths"].exists()
        ):
            recons = np.load(paths["prefix_recons"])
            paths_pi = np.load(paths["fixed_paths"])
            for fi in range(n_fixed_eff):
                recon_list = [
                    _tensor_to_uint8(torch.from_numpy(recons[fi, d]))
                    for d in range(m.n_levels)
                ]
                cells = build_path_recon_tree_cells(
                    list(paths_pi[fi].tolist()), recon_list, m,
                    cell_px=cell_px_full,
                )
                render_and_save_tree(
                    cells, m, prefix_dir,
                    prefix=f"prefix_recon_img{fi}_{m.label}",
                    title=f"{short_label} {m.label} prefix-recon-img-{fi}",
                )

    print(f"  [{short_label}] done -> {out_dir}")
