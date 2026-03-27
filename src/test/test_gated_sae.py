#!/usr/bin/env python3
"""Architecture test for GatedSparseConvAutoencoder.

Mirrors src/test/test_sae.py — validates shapes, forward pass, parameter
counts, and sparsity behaviour for both CelebA-HQ (256×256) and CIFAR-10
(32×32) configurations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.cnn.baseline.gated_sae import GatedSparseConvAutoencoder
from src.model.cnn.taxon.encoder import resolve_resnet_stage_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test GatedSparseConvAutoencoder architecture and sparsity behaviour"
    )
    parser.add_argument("--checkpoint",       type=str, default="", help="Optional checkpoint path")
    parser.add_argument("--image-size",       type=int, default=256)
    parser.add_argument("--batch-size",       type=int, default=2)
    parser.add_argument("--resnet-variant",   type=str, default="18")
    parser.add_argument("--stage-channels",   type=int, nargs=4, default=[64, 128, 256, 512])
    parser.add_argument("--stage-strides",    type=int, nargs=4, default=[1, 2, 2, 2])
    parser.add_argument("--use-gate-ste",     action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--use-stem-maxpool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include 3×3 max-pool in the stem (disable for CIFAR-10)",
    )
    parser.add_argument("--stem-stride",  type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_encoder_structure(model: GatedSparseConvAutoencoder) -> None:
    print("\n== Gated SAE Encoder Stage Structure ==")
    for idx, stage in enumerate(model.encoder.stages, start=1):
        print(
            f"stage{idx}: blocks={len(stage.blocks)}  "
            f"in_ch={model.encoder.stage_input_channels[idx - 1]}  "
            f"out_ch={model.encoder.stage_channels[idx - 1]}"
        )
    enc = model.encoder
    print(
        f"latent_channels={enc.final_channels}  "
        f"gate_proj={enc.gate_proj}  "
        f"mag_proj={enc.mag_proj}  "
        f"use_gate_ste={enc.use_gate_ste}"
    )


def check_sparsity(
    model: GatedSparseConvAutoencoder, x1: torch.Tensor, x2: torch.Tensor
) -> None:
    print("\n== Sparsity Checks (gate L1) ==")
    model.eval()
    with torch.no_grad():
        _, sp1 = model(x1)
        _, sp2 = model(x2)

    print(f"batch1 gate_l1={float(sp1):.6f}")
    print(f"batch2 gate_l1={float(sp2):.6f}")
    print(f"sparsity_finite_batch1={torch.isfinite(sp1).item()}")
    print(f"sparsity_finite_batch2={torch.isfinite(sp2).item()}")


def check_latent_decomposition(
    model: GatedSparseConvAutoencoder, x: torch.Tensor
) -> None:
    """Show gate and magnitude activation statistics separately."""
    print("\n== Gated Latent Decomposition ==")
    model.eval()
    with torch.no_grad():
        latent, info = model.encode(x)

    print(f"latent_shape={tuple(latent.shape)}")
    if "gate" in info and "magnitude" in info:
        gate  = info["gate"]
        mag   = info["magnitude"]
        total = latent.numel()
        print(
            f"gate:      min={gate.min().item():.4f}  max={gate.max().item():.4f}  "
            f"mean={gate.mean().item():.4f}  zeros={float((gate == 0).sum()) / total:.3f}"
        )
        print(
            f"magnitude: min={mag.min().item():.4f}  max={mag.max().item():.4f}  "
            f"mean={mag.mean().item():.4f}"
        )
    zeros     = (latent == 0).sum().item()
    near_zero = (latent.abs() < 1e-3).sum().item()
    total     = latent.numel()
    print(
        f"latent:    min={latent.min().item():.4f}  max={latent.max().item():.4f}  "
        f"mean={latent.mean().item():.4f}  "
        f"exact_zero={zeros/total:.4f}  near_zero={near_zero/total:.4f}"
    )


def print_architecture_diagram(model: GatedSparseConvAutoencoder, details: dict) -> None:
    print("\n== Architecture Diagram (Shape Trace) ==")
    print(f"Input: {details['input_shape']}")
    for name, shape in details["encoder"].get("shape_trace", []):
        print(f"  -> Encoder {name}: {shape}")
    print(f"Latent: {details['latent_shape']}")
    for name, shape in details["decoder"].get("shape_trace", []):
        print(f"  -> Decoder {name}: {shape}")
    print(f"Reconstruction: {details['recon_shape']}")


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    model = GatedSparseConvAutoencoder(
        in_channels=3,
        resnet_variant=args.resnet_variant,
        stage_channels=tuple(args.stage_channels),
        stage_strides=tuple(args.stage_strides),
        use_gate_ste=args.use_gate_ste,
        use_stem=True,
        stem_channels=64,
        stem_stride=args.stem_stride,
        use_stem_maxpool=args.use_stem_maxpool,
        output_activation="none",
    ).to(device)

    if args.checkpoint:
        ckpt       = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    total_params, trainable_params = count_parameters(model)
    print("== Model Summary ==")
    print(model)
    print(f"\nparameter_count_total={total_params:,}")
    print(f"parameter_count_trainable={trainable_params:,}")

    print_encoder_structure(model)

    x1 = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    x2 = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

    model.eval()
    with torch.no_grad():
        recon, sparsity, details = model(x1, return_details=True)

    print("\n== Forward Pass ==")
    print(f"input_shape={tuple(x1.shape)}")
    print(f"recon_shape={tuple(recon.shape)}")
    print(f"gate_l1_sparsity={float(sparsity):.6f}")
    print(f"shapes_match={tuple(x1.shape) == tuple(recon.shape)}")

    print_architecture_diagram(model, details)
    check_sparsity(model, x1, x2)
    check_latent_decomposition(model, x1)


if __name__ == "__main__":
    main()
