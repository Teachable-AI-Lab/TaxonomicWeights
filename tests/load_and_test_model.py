#!/usr/bin/env python3
"""Utility for loading a checkpoint and running basic sanity checks.

The script replicates the model-definition used during training so that a
checkpoint saved by :mod:`tests/train_*` can be reloaded exactly.  It
then runs a handful of simple tests:

* print the model architecture and parameter count
* validate encoder ``evaluate_path_encoding`` on a tiny random batch
* if a dataset is available, run a couple of real batches through the
  autoencoder and report reconstruction error

Usage::

    python tests/load_and_test_model.py \
        --config configs/celeba_hq_attention.json \
        --checkpoint outputs/.../checkpoints/best.pt \
        [--batch-size 32] [--data-root PATH]

The arguments mirror those exposed by the analysis scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

# make the repo root importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.taxon_ae import TaxonAutoencoder
from src.model.decoder import TaxonResNetDecoder
from src.utils.dataloader import CelebAHQLoader


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def instantiate_model(config: dict, device: torch.device) -> torch.nn.Module:
    """Instantiate the model described by ``config['model']``.

    If ``attn_heads`` is present and greater than zero we need to build the
    attention variant manually; the plain :class:`TaxonAutoencoder` does not
    expose that option.  This matches the behaviour of
    ``train_celeba_hq_ae_attention.py``.
    """
    mc = config.get("model", {})
    attn_heads = mc.get("attn_heads", 0)

    if attn_heads and attn_heads > 0:
        # attention-enabled encoder/decoder
        from src.model.encoder import TaxonResNetEncoderWithAttention

        encoder = TaxonResNetEncoderWithAttention(
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
            depth_decay=mc.get("depth_decay", 0.5),
            attn_heads=attn_heads,
        )
        decoder = TaxonResNetDecoder(
            latent_channels=encoder.final_channels,
            stage_input_channels=encoder.stage_input_channels,
            stage_blocks=encoder.stage_blocks,
            stage_strides=encoder.stage_strides,
            out_channels=mc.get("in_channels", 3),
            use_stem=mc.get("use_stem", True),
            stem_total_stride=encoder.stem_total_stride,
            kernel_size=mc.get("kernel_size", 3),
        )

        # wrapper taken from train_celeba_hq_ae_attention.py
        class _Wrapper(nn.Module):
            def __init__(self, enc, dec, out_act="none"):
                super().__init__()
                self.encoder = enc
                self.decoder = dec
                self.output_activation = out_act
                self.default_hard = mc.get("hard", False)

            def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
                if self.output_activation == "tanh":
                    return torch.tanh(x)
                if self.output_activation == "sigmoid":
                    return torch.sigmoid(x)
                return x

            def forward(self, x: torch.Tensor, hard: Optional[bool] = None):
                if hard is None:
                    hard = self.default_hard
                z, enc_details = self.encoder(x, hard=hard, return_details=True)
                recon, dec_details = self.decoder(
                    z, output_size=x.shape[-2:], return_details=True
                )
                recon = self._apply_output_activation(recon)
                dkl = enc_details["dkl"]
                entropy = enc_details["entropy"]
                return recon, dkl, entropy

        model = _Wrapper(encoder, decoder, out_act=mc.get("output_activation", "none"))
        model.to(device)
        return model

    # default (non-attention) case
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
    model.to(device)
    return model


def load_checkpoint(model: nn.Module, path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint '{path}' (epoch {checkpoint.get('epoch', '?')})")


def setup_loader(data_root: str, image_size: int, batch_size: int, num_workers: int):
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    celeba = CelebAHQLoader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        val_split=0.0,
        seed=0,
        pin_memory=False,
        transform=tf,
    )
    train_loader, val_loader = celeba.get_loaders()
    return train_loader, val_loader


def test_encoder_paths(model: nn.Module, device: torch.device) -> None:
    print("\n> encoder path encoding checks")
    enc = getattr(model, "encoder", None)
    if enc is None:
        print("  model has no .encoder attribute, skipping")
        return
    model.eval()
    with torch.no_grad():
        x = torch.randn(2, 3, 256, 256, device=device)
        # run through stem so each stage receives correctly-shaped input
        x = enc.stem(x)
        for idx, stage in enumerate(enc.taxon_stages, start=1):
            rep = stage.evaluate_path_encoding(x)
            print(f" stage {idx}: ok={rep['ok']} depth={rep['n_depths']} "
                  f"dkl={rep['dkl']:.4f} entr={rep['entropy']:.4f}")
            # propagate through stage for the next iteration
            x, _, _ = stage(x)


def extract_parse_tree(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> list:
    """Extract the taxonomy parse tree for each image in ``images``.

    For every encoder stage and every tree depth the function records the
    path-probability distribution (pooled over spatial dimensions) and
    identifies the most-probable leaf node.

    Returns a list with one entry per sample.  Each entry is a list of
    stage dicts::

        {
          "stage": int,               # 1-indexed
          "n_depths": int,
          "total_channels": int,
          "depths": [
            {
              "depth": int,           # 1-indexed within stage
              "n_nodes": int,         # 2^depth
              "probs": np.ndarray,    # (n_nodes,) spatially-pooled path prob
              "selected": int,        # argmax node index
              "path_bits": list[int], # binary path from root (0=L, 1=R)
              "path_str": str,        # e.g. "L-R-L"
              "cond_prob": float,     # P(direction | parent) at this depth
            },
            ...
          ]
        }
    """
    import numpy as np

    enc = getattr(model, "encoder", None)
    if enc is None:
        raise ValueError("model has no .encoder attribute")

    model.eval()
    images = images.to(device)
    B = images.shape[0]

    stage_data = []  # (stage_num, n_depths, [(depth_num, n_nodes, prob_pooled), ...])
    with torch.no_grad():
        x = enc.stem(images)
        for stage_idx, stage in enumerate(enc.taxon_stages):
            out, logp_all, _ = stage(x)
            depths = []
            for d_idx, (n_ch, logp_chunk) in enumerate(
                zip(stage.layer_channels,
                    torch.split(logp_all, stage.layer_channels, dim=1))
            ):
                # logp_chunk: (B, n_ch, H, W)  — log joint path probability
                prob = logp_chunk.exp()
                # pool over spatial dims -> (B, n_ch)
                prob_pooled = prob.mean(dim=(2, 3)).cpu().numpy()
                depths.append((d_idx + 1, n_ch, prob_pooled))
            stage_data.append((stage_idx + 1, stage.n_taxonomy_layers,
                                stage.total_out_channels, depths))
            x = out

    per_sample: list = []
    for s_idx in range(B):
        sample_stages = []
        for stage_num, n_depths, total_ch, depths in stage_data:
            depths_out = []
            for depth_num, n_nodes, prob_pooled in depths:
                p = prob_pooled[s_idx]           # (n_nodes,)
                selected = int(p.argmax())
                # decode binary path: MSB first
                bits = [(selected >> (depth_num - 1 - b)) & 1
                        for b in range(depth_num)]
                path_str = "-".join("R" if b else "L" for b in bits)
                # conditional P(choice | parent) via sibling pair
                sibling = selected ^ 1           # flip LSB = sibling sharing same parent
                pair_total = float(p[selected]) + float(p[sibling])
                cond_prob = float(p[selected]) / max(pair_total, 1e-9)
                depths_out.append({
                    "depth": depth_num,
                    "n_nodes": n_nodes,
                    "probs": p,
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


def print_parse_tree(per_sample: list, max_samples: int = 4) -> None:
    """Print a human-readable parse tree for up to ``max_samples`` images."""
    for s_idx, stages in enumerate(per_sample[:max_samples]):
        print(f"\n{'='*60}")
        print(f"  Parse tree — Sample {s_idx}")
        print(f"{'='*60}")
        for stage_info in stages:
            n_depths = stage_info["n_depths"]
            total_ch = stage_info["total_channels"]
            print(f"\n  Stage {stage_info['stage']}  "
                  f"({n_depths} depths, {total_ch} latent channels)")

            for d_info in stage_info["depths"]:
                d = d_info["depth"]
                bits = d_info["path_bits"]
                path_str = d_info["path_str"]
                cond = d_info["cond_prob"]
                leaf_p = float(d_info["probs"][d_info["selected"]])
                direction = "R" if bits[-1] else "L"
                indent = "  " * (d + 1)
                # show sibling probability too
                direction_opp = "L" if bits[-1] else "R"
                print(f"  {indent}depth {d:2d}: "
                      f"[{direction} p={cond:.3f} | {direction_opp} p={1-cond:.3f}] "
                      f"→  path={path_str}  joint_p={leaf_p:.5f}")

            # summary line for the leaf
            leaf = stage_info["depths"][-1]
            leaf_node = leaf["selected"]
            n_nodes = leaf["n_nodes"]
            print(f"\n  Stage {stage_info['stage']} leaf: node {leaf_node}/{n_nodes-1} "
                  f"({leaf['path_str']})  joint_p={float(leaf['probs'][leaf_node]):.5f}")


def save_parse_tree_json(per_sample: list, path: str) -> None:
    """Save parse tree (without raw prob arrays) to a JSON file."""
    import json as _json

    serialisable = []
    for stages in per_sample:
        s_list = []
        for stage_info in stages:
            d_list = []
            for d in stage_info["depths"]:
                d_list.append({
                    "depth":      d["depth"],
                    "n_nodes":    d["n_nodes"],
                    "selected":   d["selected"],
                    "path_bits":  d["path_bits"],
                    "path_str":   d["path_str"],
                    "cond_prob":  round(float(d["cond_prob"]), 6),
                    "joint_prob": round(float(d["probs"][d["selected"]]), 6),
                })
            s_list.append({
                "stage": stage_info["stage"],
                "n_depths": stage_info["n_depths"],
                "total_channels": stage_info["total_channels"],
                "depths": d_list,
            })
        serialisable.append(s_list)

    with open(path, "w") as f:
        _json.dump(serialisable, f, indent=2)
    print(f"Parse tree saved to {path}")


def visualize_parse_tree(
    per_sample: list,
    images: torch.Tensor,
    save_dir: str,
    max_samples: int = 4,
) -> None:
    """Render one PNG per sample showing the binary parse tree for every stage.

    Each stage is drawn as a full binary tree.  Node colour encodes the joint
    (root-to-node) path probability.  The greedy selected path is highlighted
    with a thick red border and connecting edges.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)

    for s_idx, stages in enumerate(per_sample[:max_samples]):
        n_stages = len(stages)
        # figure: [image column] + [one column per stage]
        fig_w = 3.5 + 4.5 * n_stages
        fig_h = 6
        fig, axes = plt.subplots(1, n_stages + 1, figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#1a1a2e")
        for ax in axes:
            ax.set_facecolor("#1a1a2e")

        # ── left panel: input image ──────────────────────────────────────
        ax_img = axes[0]
        img_np = images[s_idx].cpu().float().numpy()
        # denorm from [-1,1]
        img_np = np.clip(img_np * 0.5 + 0.5, 0, 1)
        if img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)
        elif img_np.shape[0] == 1:
            img_np = img_np[0]
        ax_img.imshow(img_np, cmap="gray" if img_np.ndim == 2 else None)
        ax_img.set_title(f"Sample {s_idx}", color="white", fontsize=10, pad=6)
        ax_img.axis("off")

        # ── per-stage tree panels ──────────────────────────────────────────
        cmap = cm.get_cmap("YlOrRd")
        for col, stage_info in enumerate(stages):
            ax = axes[col + 1]
            ax.set_xlim(-1, 1)
            depths = stage_info["depths"]
            n_depths = len(depths)
            ax.set_ylim(-0.5, n_depths + 0.5)
            ax.set_title(f"Stage {stage_info['stage']}\n({n_depths} depths)",
                         color="white", fontsize=9, pad=6)
            ax.axis("off")

            # collect selected node indices for each depth (for path highlighting)
            selected_per_depth = [d["selected"] for d in depths]

            # draw depth by depth
            for d_idx, d_info in enumerate(depths):
                y = n_depths - d_idx          # depth 1 near top
                n_nodes = d_info["n_nodes"]   # 2^depth
                probs = d_info["probs"]
                selected = d_info["selected"]

                # x positions: evenly spaced in [-0.9, 0.9]
                xs = np.linspace(-0.9, 0.9, n_nodes)

                # draw edges to children (next depth) if not last
                if d_idx + 1 < n_depths:
                    y_child = n_depths - (d_idx + 1)
                    n_children = d_info["n_nodes"] * 2
                    xs_child = np.linspace(-0.9, 0.9, n_children)
                    for node_i, x_parent in enumerate(xs):
                        lc = node_i * 2
                        rc = node_i * 2 + 1
                        for child_i in (lc, rc):
                            on_path = (selected_per_depth[d_idx] == node_i and
                                       selected_per_depth[d_idx + 1] == child_i)
                            ax.plot([x_parent, xs_child[child_i]], [y, y_child],
                                    color="#e63946" if on_path else "#444466",
                                    lw=2.5 if on_path else 0.7,
                                    zorder=1)

                # draw nodes
                for node_i, (x_node, p) in enumerate(zip(xs, probs)):
                    colour = cmap(float(p) / max(probs.max(), 1e-9))
                    is_selected = (node_i == selected)
                    circle = plt.Circle(
                        (x_node, y),
                        radius=0.07 if n_nodes <= 32 else 0.04,
                        color=colour,
                        ec="#e63946" if is_selected else "#888899",
                        lw=2.0 if is_selected else 0.5,
                        zorder=2,
                    )
                    ax.add_patch(circle)

            # colour bar legend (per stage panel)
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=plt.Normalize(vmin=0, vmax=float(depths[-1]["probs"].max())))
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02,
                                orientation="vertical")
            cbar.ax.yaxis.set_tick_params(color="white")
            cbar.outline.set_edgecolor("white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white", fontsize=6)
            cbar.set_label("joint prob", color="white", fontsize=7)

        # legend patch
        sel_patch = mpatches.Patch(color="#e63946", label="selected path")
        fig.legend(handles=[sel_patch], loc="lower center", ncol=1,
                   facecolor="#1a1a2e", edgecolor="white",
                   labelcolor="white", fontsize=8)

        plt.suptitle(f"Taxonomy Parse Tree — Sample {s_idx}",
                     color="white", fontsize=12, fontweight="bold", y=1.01)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"parse_tree_sample_{s_idx:03d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  saved {out_path}")


def extract_hierarchical_activations(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> list:
    """Run the encoder and return per-depth gated activations for every stage.

    For each encoder stage and each taxonomy depth we capture::

        gated_acts  (B, 2^d, H, W)  – logits * joint path probability
        raw_logits  (B, 2^d, H, W)  – pre-gating logits from the residual blocks
        probs       (B, 2^d, H, W)  – joint path probability (root-to-node)

    The list returned has one dict per stage::

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
                    "n_nodes": logits.shape[1],      # 2^(d_idx+1)
                    "gated_acts": out.cpu().float(),   # (B, n_nodes, H, W)
                    "raw_logits": logits.cpu().float(),
                    "probs": prob.cpu().float(),       # joint prob per node
                })
                prev_logp = logp

            x = torch.cat(stage_outputs, dim=1)        # advance to next stage
            all_stage_data.append({
                "stage": stage_idx + 1,
                "n_depths": stage.n_taxonomy_layers,
                "depths": depths,
            })

    return all_stage_data


def visualize_hierarchical_activations(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    save_dir: str,
    max_samples: int = 4,
    max_depth: int = 4,
    use_nms: bool = False,
    nms_threshold: float = 0.0,
) -> None:
    """Save a PNG per (sample, stage) showing gated activations as a binary tree.

    Layout
    ------
    Rows  -> depth levels (depth 1 at top, depth ``max_depth`` at bottom).
    Cols  -> tree nodes at that depth, left-to-right binary order (up to 16).

    Each cell is a spatial heatmap of the node's gated activation map.
    The selected node (argmax spatially-pooled probability) is highlighted with a
    red border.

    When ``use_nms=True`` spatial non-maximum suppression is applied: at each
    spatial location only the node with the highest joint path probability retains
    its activation; all others are zeroed.  An additional winner-map PNG is saved
    showing which node dominates each pixel at every depth level.
    Optional ``nms_threshold`` (0–1) further masks out pixels where even the
    winning node's probability is below the threshold.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)

    images_cpu = images[:max_samples]
    B = images_cpu.shape[0]

    print("  extracting hierarchical activations …")
    stage_data = extract_hierarchical_activations(model, images_cpu, device)

    for s_idx in range(B):
        for stage_info in stage_data:
            stage_num = stage_info["stage"]
            depths = stage_info["depths"][:max_depth]
            n_rows = len(depths)

            # max columns = node count at the deepest depth shown (≤16)
            max_cols = min(depths[-1]["n_nodes"], 16)

            cell = 1.6   # inches per cell
            fig_w = max(6.0, cell * max_cols)
            fig_h = cell * n_rows + 0.6   # +0.6 for suptitle

            fig, axes = plt.subplots(
                n_rows, max_cols,
                figsize=(fig_w, fig_h),
                squeeze=False,
                facecolor="#0f0f1c",
            )
            fig.subplots_adjust(hspace=0.55, wspace=0.12,
                                left=0.06, right=0.98, top=0.90, bottom=0.04)

            # hide all axes initially
            for ax in axes.flat:
                ax.set_visible(False)

            # build per-depth NMS winner maps if requested
            if use_nms:
                import numpy as np
                _winner_maps = []   # (n_depths,) of (H, W) int arrays
                _winner_probs = []  # (n_depths,) of (H, W) float arrays
                _conf_masks = []    # (n_depths,) of (H, W) bool arrays
                for d_info in depths:
                    pm = d_info["probs"][s_idx]        # (n_nodes, H, W)
                    w = pm.argmax(dim=0).numpy()       # (H, W) int
                    wp = pm.max(dim=0).values.numpy()  # (H, W) float
                    cm = wp > nms_threshold            # (H, W) bool
                    _winner_maps.append(w)
                    _winner_probs.append(wp)
                    _conf_masks.append(cm)

            for row, d_info in enumerate(depths):
                d = d_info["depth"]
                n_nodes = d_info["n_nodes"]
                gated = d_info["gated_acts"][s_idx]    # (n_nodes, H, W)
                prob_maps = d_info["probs"][s_idx]     # (n_nodes, H, W)
                # spatially-pooled probability → find selected node
                prob_vals = prob_maps.mean(dim=(-1, -2)).numpy()
                selected = int(prob_vals.argmax())

                # NMS masks for this depth
                if use_nms:
                    _wmap = _winner_maps[row]         # (H, W)
                    _cmask = _conf_masks[row]         # (H, W) bool

                # choose which node indices to show in this row
                if n_nodes <= max_cols:
                    show_nodes = list(range(n_nodes))
                    col_offset = (max_cols - n_nodes) // 2   # centre smaller rows
                else:
                    # window of max_cols around selected node
                    half = max_cols // 2
                    start = max(0, min(selected - half, n_nodes - max_cols))
                    show_nodes = list(range(start, start + max_cols))
                    col_offset = 0

                for col_idx, node_i in enumerate(show_nodes):
                    col = col_offset + col_idx
                    if col >= max_cols:
                        break
                    ax = axes[row, col]
                    ax.set_visible(True)
                    ax.set_facecolor("#0f0f1c")

                    act = gated[node_i].numpy()            # (H, W)
                    if use_nms:
                        # zero out every pixel where this node is NOT the winner
                        nms_mask = (_wmap == node_i) & _cmask
                        act = act * nms_mask.astype(act.dtype)

                    vmax = float(abs(act).max()) or 1e-6
                    ax.imshow(act, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                              aspect="auto", interpolation="nearest")
                    ax.set_xticks([])
                    ax.set_yticks([])

                    is_sel = (node_i == selected)
                    ec = "#e63946" if is_sel else "#2a2a4a"
                    lw = 2.2 if is_sel else 0.5
                    for spine in ax.spines.values():
                        spine.set_edgecolor(ec)
                        spine.set_linewidth(lw)

                    p = float(prob_vals[node_i])
                    label = (f"★{node_i}" if is_sel else f"{node_i}") + f"\n{p:.3f}"
                    ax.set_title(
                        label, fontsize=5.5,
                        color="#e63946" if is_sel else "#7070a0",
                        pad=1.5,
                    )

                # depth label on the left-most visible axis
                left_col = col_offset if n_nodes <= max_cols else 0
                axes[row, left_col].set_ylabel(
                    f"depth {d}", color="#aaaacc", fontsize=6.5,
                    rotation=90, labelpad=3,
                )

            nms_tag = " [NMS]" if use_nms else ""
            fig.suptitle(
                f"Hierarchical Activations{nms_tag}  ·  Sample {s_idx}  ·  Stage {stage_num}",
                color="white", fontsize=10, fontweight="bold",
            )

            suffix = "_nms" if use_nms else ""
            out_path = os.path.join(
                save_dir, f"hier_acts_s{s_idx:03d}_stage{stage_num}{suffix}.png"
            )
            plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
            plt.close()
            print(f"  saved {out_path}")

            # ── winner-map overview (NMS only) ─────────────────────────────────
            if use_nms:
                import numpy as np
                from matplotlib.colors import BoundaryNorm
                import matplotlib.cm as cm

                wfig, waxes = plt.subplots(
                    1, len(depths),
                    figsize=(3.5 * len(depths), 3.5),
                    facecolor="#0f0f1c",
                    squeeze=False,
                )
                wfig.subplots_adjust(wspace=0.15, left=0.04, right=0.96,
                                     top=0.82, bottom=0.06)

                for col, (d_info, wmap, wprob, cmask) in enumerate(
                    zip(depths, _winner_maps, _winner_probs, _conf_masks)
                ):
                    ax = waxes[0, col]
                    ax.set_facecolor("#0f0f1c")
                    n_nodes = d_info["n_nodes"]

                    # winner index coloured map; masked pixels shown as -1 (black)
                    disp = np.where(cmask, wmap, -1).astype(float)

                    cmap_nodes = cm.get_cmap("tab20", n_nodes)
                    bounds = [-1.5] + [i - 0.5 for i in range(n_nodes)] + [n_nodes - 0.5]
                    norm = BoundaryNorm(bounds, cmap_nodes.N + 1)

                    # use a ListedColormap that maps -1 → black
                    from matplotlib.colors import ListedColormap
                    colors = ["#0f0f1c"] + [
                        cmap_nodes(i / max(n_nodes - 1, 1)) for i in range(n_nodes)
                    ]
                    lmap = ListedColormap(colors)
                    lnorm = BoundaryNorm(
                        [-1.5] + [i - 0.5 for i in range(n_nodes + 1)],
                        len(colors),
                    )

                    ax.imshow(disp, cmap=lmap, norm=lnorm,
                              aspect="auto", interpolation="nearest")

                    # overlay confidence (winning prob) as alpha-blended contour
                    alpha_map = np.clip(wprob * cmask.astype(float), 0, 1)
                    ax.imshow(alpha_map, cmap="gray", alpha=0.25,
                              aspect="auto", interpolation="nearest")

                    ax.set_xticks([])
                    ax.set_yticks([])
                    d = d_info["depth"]
                    ax.set_title(
                        f"depth {d}  ({n_nodes} nodes)",
                        color="#aaaacc", fontsize=8, pad=3,
                    )
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#333355")

                thresh_str = f"  (thr={nms_threshold:.2f})" if nms_threshold > 0 else ""
                wfig.suptitle(
                    f"Winner Map{thresh_str}  ·  Sample {s_idx}  ·  Stage {stage_num}",
                    color="white", fontsize=10, fontweight="bold",
                )
                wout = os.path.join(
                    save_dir,
                    f"winner_map_s{s_idx:03d}_stage{stage_num}.png",
                )
                wfig.savefig(wout, dpi=140, facecolor=wfig.get_facecolor())
                plt.close(wfig)
                print(f"  saved {wout}")


def visualize_binary_split_maps(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    save_dir: str,
    max_samples: int = 4,
    max_depth: int = 4,
    max_pairs: int = 8,
) -> None:
    """For each encoder stage and depth, visualise how each sibling pair splits space.

    At depth ``d`` there are ``2^(d-1)`` sibling pairs.  For pair ``k``
    (covering nodes ``2k`` and ``2k+1``) we compute at every spatial pixel::

        cond_left(h,w) = P_joint(2k, h,w) / (P_joint(2k, h,w) + P_joint(2k+1, h,w))

    This is the conditional probability of routing *left* given the parent is
    active at that pixel.  Pixels where the parent itself has negligible
    probability (off-path regions) are faded to black via an alpha channel so
    only committed, unambiguous assignments are visible.

    Layout
    ------
    One PNG per (sample, stage).
    Rows  = depth levels.
    Cols  = sibling pairs at that depth (up to ``max_pairs``, windowed around
            the greedy-selected pair).
    ColourMap: RdBu_r — blue=right wins (cond_left→0), red=left wins
               (cond_left→1), white=uncertain (cond_left≈0.5).
    Alpha     = normalised parent joint probability (off-path → transparent).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)

    images_cpu = images[:max_samples]
    B = images_cpu.shape[0]

    print("  extracting activations for binary split maps …")
    stage_data = extract_hierarchical_activations(model, images_cpu, device)

    rdbu = cm.get_cmap("RdBu_r")          # blue=right, red=left
    bg_colour = np.array([0.06, 0.06, 0.11, 1.0])  # dark navy background

    for s_idx in range(B):
        for stage_info in stage_data:
            stage_num = stage_info["stage"]
            depths = stage_info["depths"][:max_depth]
            n_rows = len(depths)

            # The widest row is the deepest depth; compute max pairs to show
            max_pairs_here = min(depths[-1]["n_nodes"] // 2, max_pairs)
            max_pairs_here = max(max_pairs_here, 1)

            cell_w, cell_h = 2.0, 2.0
            fig_w = max(5.0, cell_w * max_pairs_here + 1.0)
            fig_h = cell_h * n_rows + 0.9

            fig, axes = plt.subplots(
                n_rows, max_pairs_here,
                figsize=(fig_w, fig_h),
                squeeze=False,
                facecolor="#0f0f1c",
            )
            fig.subplots_adjust(hspace=0.65, wspace=0.08,
                                left=0.07, right=0.97, top=0.88, bottom=0.05)

            for ax in axes.flat:
                ax.set_visible(False)

            for row, d_info in enumerate(depths):
                d = d_info["depth"]
                n_nodes = d_info["n_nodes"]
                n_pairs = n_nodes // 2

                # joint probs: (n_nodes, H, W) tensor
                probs = d_info["probs"][s_idx]          # tensor
                prob_vals = probs.mean(dim=(-1, -2)).numpy()
                selected_node = int(prob_vals.argmax())
                selected_pair = selected_node // 2

                # which pairs to show this row
                if n_pairs <= max_pairs_here:
                    show_pairs = list(range(n_pairs))
                    col_offset = (max_pairs_here - n_pairs) // 2
                else:
                    half = max_pairs_here // 2
                    start = max(0, min(selected_pair - half,
                                      n_pairs - max_pairs_here))
                    show_pairs = list(range(start, start + max_pairs_here))
                    col_offset = 0

                # max parent prob for alpha normalisation (across all pairs)
                p_parent_all = (probs[0::2] + probs[1::2]).max().item()
                p_parent_all = max(p_parent_all, 1e-9)

                for col_idx, pair_k in enumerate(show_pairs):
                    col = col_offset + col_idx
                    if col >= max_pairs_here:
                        break
                    ax = axes[row, col]
                    ax.set_visible(True)
                    ax.set_facecolor("#0f0f1c")

                    p_l = probs[2 * pair_k].numpy()      # (H, W) joint prob left
                    p_r = probs[2 * pair_k + 1].numpy()  # (H, W) joint prob right
                    p_parent = p_l + p_r                 # (H, W) parent joint prob

                    cond_left = p_l / np.maximum(p_parent, 1e-9)  # (H, W) in [0,1]

                    # alpha: how much of the image's probability mass flows through
                    # this pair (boosts contrast; cap at 1)
                    alpha = np.clip(p_parent / p_parent_all * 4.0, 0.0, 1.0)

                    # build RGBA image: colour from cond_left, alpha from parent mass
                    rgba = rdbu(cond_left).astype(np.float64)   # (H, W, 4)
                    rgba[..., 3] = alpha

                    # draw dark background then semi-transparent split map on top
                    H, W = cond_left.shape
                    bg = np.ones((H, W, 4)) * bg_colour
                    ax.imshow(bg, aspect="auto", interpolation="nearest")
                    ax.imshow(rgba, aspect="auto", interpolation="nearest")

                    ax.set_xticks([])
                    ax.set_yticks([])

                    # border: thick red if this pair is on the greedy path
                    is_sel = (pair_k == selected_pair)
                    ec = "#e63946" if is_sel else "#1e1e3a"
                    lw = 2.5 if is_sel else 0.6
                    for spine in ax.spines.values():
                        spine.set_edgecolor(ec)
                        spine.set_linewidth(lw)

                    # overall winner direction for this pair (spatially pooled)
                    mean_cond = float(cond_left.mean())
                    arrow = "← L" if mean_cond > 0.5 else "R →"
                    conf = abs(mean_cond - 0.5) * 2.0   # 0=uncertain, 1=certain
                    title_col = "#e63946" if is_sel else "#8888aa"
                    ax.set_title(
                        f"{'★' if is_sel else ''}L{2*pair_k}|R{2*pair_k+1}\n"
                        f"{arrow}  {conf:.2f}",
                        fontsize=5.5, color=title_col, pad=1.5,
                    )

                # row label
                left_col = col_offset if n_pairs <= max_pairs_here else 0
                axes[row, left_col].set_ylabel(
                    f"depth {d}", color="#aaaacc", fontsize=7,
                    rotation=90, labelpad=3,
                )

            # shared colour bar: blue=right, white=uncertain, red=left
            sm = plt.cm.ScalarMappable(
                cmap="RdBu_r", norm=plt.Normalize(vmin=0, vmax=1)
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes[:, -1], fraction=0.05, pad=0.02,
                                orientation="vertical")
            cbar.set_ticks([0.0, 0.5, 1.0])
            cbar.set_ticklabels(["Right\n(P=1)", "Uncertain", "Left\n(P=1)"],
                                fontsize=5.5)
            cbar.ax.yaxis.set_tick_params(color="white")
            cbar.outline.set_edgecolor("white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
            cbar.set_label("P(left | parent)", color="white", fontsize=7)

            fig.suptitle(
                f"Binary Split Maps  ·  Sample {s_idx}  ·  Stage {stage_num}\n"
                f"(alpha ∝ parent probability — bright = unambiguous split)",
                color="white", fontsize=9, fontweight="bold",
            )

            out_path = os.path.join(
                save_dir, f"split_map_s{s_idx:03d}_stage{stage_num}.png"
            )
            plt.savefig(out_path, dpi=140, facecolor=fig.get_facecolor())
            plt.close()
            print(f"  saved {out_path}")


def test_reconstruction(model: TaxonAutoencoder, loader: DataLoader, device: torch.device, n_batches: int = 2) -> None:
    if loader is None:
        print("no dataset loader available, skipping reconstruction test")
        return
    print("\n> reconstruction on real data")
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches:
                break
            imgs = imgs.to(device)
            recon, dkl, entropy = model(imgs)
            loss = F.mse_loss(recon, imgs).item()
            print(f" batch {i}: mse={loss:.6f} dkl={dkl.mean():.4f} ent={entropy.mean():.4f}")
            losses.append(loss)
    if losses:
        print(f" avg mse {sum(losses)/len(losses):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Load a TaxonAutoencoder checkpoint and run basic tests"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="JSON config used for training")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="path to checkpoint file")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--parse-tree", action="store_true",
                        help="extract and print the taxonomy parse tree")
    parser.add_argument("--parse-images", type=int, default=4,
                        help="number of images to run parse-tree extraction on (default 4)")
    parser.add_argument("--save-tree", type=str, default=None,
                        help="optional path to save parse tree as JSON")
    parser.add_argument("--viz-tree", type=str, default=None,
                        metavar="DIR",
                        help="directory to save parse-tree PNGs (enables parse tree)")
    parser.add_argument("--viz-acts", type=str, default=None,
                        metavar="DIR",
                        help="directory to save hierarchical activation PNGs")
    parser.add_argument("--max-act-depth", type=int, default=4,
                        help="max taxonomy depth shown per stage in activation viz (default 4)")
    parser.add_argument("--nms", action="store_true",
                        help="apply spatial NMS: each pixel keeps only its winning node's activation")
    parser.add_argument("--nms-threshold", type=float, default=0.0,
                        help="pixels where the winning probability < this value are masked to 0 (default 0.0)")
    parser.add_argument("--viz-splits", type=str, default=None,
                        metavar="DIR",
                        help="directory to save binary split-map PNGs")
    parser.add_argument("--max-splits", type=int, default=8,
                        help="max sibling pairs shown per depth row in split maps (default 8)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    batch_size = args.batch_size or data_cfg.get("batch_size", 16)
    num_workers = args.num_workers or data_cfg.get("num_workers", 4)
    data_root = args.data_root or data_cfg.get("data_root", "./data/celeba_hq")
    image_size = data_cfg.get("image_size", 256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = instantiate_model(cfg, device)
    load_checkpoint(model, args.checkpoint)

    print("\nmodel summary:")
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {n_params:,}")

    # run a couple of simple tests
    test_encoder_paths(model, device)

    # try loading data if available
    train_loader, val_loader = None, None
    try:
        train_loader, val_loader = setup_loader(data_root, image_size, batch_size, num_workers)
    except Exception as e:
        print(f"failed to create data loader: {e}")
    test_reconstruction(model, val_loader or train_loader, device)

    # parse tree
    if args.parse_tree or args.viz_tree:
        print("\n> extracting parse trees...")
        loader_for_tree = val_loader or train_loader
        if loader_for_tree is not None:
            imgs, _ = next(iter(loader_for_tree))
            imgs = imgs[: args.parse_images]
        else:
            print("  no data loader available, using random inputs")
            mc = cfg.get("model", {})
            imgs = torch.randn(args.parse_images, mc.get("in_channels", 3),
                               image_size, image_size)
        trees = extract_parse_tree(model, imgs, device)
        print_parse_tree(trees)
        if args.save_tree:
            save_parse_tree_json(trees, args.save_tree)
        if args.viz_tree:
            visualize_parse_tree(trees, imgs, save_dir=args.viz_tree,
                                 max_samples=args.parse_images)

    # hierarchical activation maps
    if args.viz_acts:
        print("\n> visualising hierarchical activations...")
        loader_for_acts = val_loader or train_loader
        if loader_for_acts is not None:
            imgs_acts, _ = next(iter(loader_for_acts))
            imgs_acts = imgs_acts[: args.parse_images]
        else:
            print("  no data loader, using random inputs")
            mc = cfg.get("model", {})
            imgs_acts = torch.randn(args.parse_images, mc.get("in_channels", 3),
                                    image_size, image_size)
        visualize_hierarchical_activations(
            model, imgs_acts, device,
            save_dir=args.viz_acts,
            max_samples=args.parse_images,
            max_depth=args.max_act_depth,
            use_nms=args.nms,
            nms_threshold=args.nms_threshold,
        )

    # binary split maps
    if args.viz_splits:
        print("\n> visualising binary split maps...")
        loader_for_splits = val_loader or train_loader
        if loader_for_splits is not None:
            imgs_splits, _ = next(iter(loader_for_splits))
            imgs_splits = imgs_splits[: args.parse_images]
        else:
            print("  no data loader, using random inputs")
            mc = cfg.get("model", {})
            imgs_splits = torch.randn(args.parse_images, mc.get("in_channels", 3),
                                      image_size, image_size)
        visualize_binary_split_maps(
            model, imgs_splits, device,
            save_dir=args.viz_splits,
            max_samples=args.parse_images,
            max_depth=args.max_act_depth,
            max_pairs=args.max_splits,
        )


if __name__ == "__main__":
    main()
