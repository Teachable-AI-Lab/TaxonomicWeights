#!/usr/bin/env python3
"""Architecture test for JumpReLUSparseConvAutoencoder.

Mirrors src/test/test_topk_sae.py — validates shapes, forward pass, parameter
counts, and JumpReLU sparsity behaviour for both CelebA-HQ (256×256) and
CIFAR-10 (32×32) configurations.
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

from src.model.jumprelu_sae import JumpReLUSparseConvAutoencoder
from src.model.encoder import resolve_resnet_stage_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test JumpReLUSparseConvAutoencoder architecture and sparsity behaviour"
    )
    parser.add_argument("--checkpoint",       type=str, default="", help="Optional checkpoint path")
    parser.add_argument("--image-size",       type=int, default=256)
    parser.add_argument("--batch-size",       type=int, default=2)
    parser.add_argument("--resnet-variant",   type=str, default="18")
    parser.add_argument("--stage-channels",   type=int, nargs=4, default=[64, 128, 256, 512])
    parser.add_argument("--stage-strides",    type=int, nargs=4, default=[1, 2, 2, 2])
    parser.add_argument("--target-l0",        type=float, default=64.0)
    parser.add_argument("--bandwidth",        type=float, default=0.001)
    parser.add_argument("--theta-init",       type=float, default=0.1)
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


def print_encoder_structure(model: JumpReLUSparseConvAutoencoder) -> None:
    print("\n== JumpReLU SAE Encoder Stage Structure ==")
    for idx, stage in enumerate(model.encoder.stages, start=1):
        print(
            f"stage{idx}: blocks={len(stage.blocks)}  "
            f"in_ch={model.encoder.stage_input_channels[idx - 1]}  "
            f"out_ch={model.encoder.stage_channels[idx - 1]}"
        )
    print(
        f"latent_channels={model.encoder.final_channels}  "
        f"target_l0={model.encoder.target_l0}  "
        f"bandwidth={model.encoder.bandwidth}  "
        f"theta_init={model.encoder.theta_init}"
    )
    theta = model.encoder.log_theta.exp()
    print(
        f"theta_mean={float(theta.mean()):.5f}  "
        f"theta_min={float(theta.min()):.5f}  "
        f"theta_max={float(theta.max()):.5f}"
    )


def check_sparsity(
    model: JumpReLUSparseConvAutoencoder, x1: torch.Tensor, x2: torch.Tensor
) -> None:
    print("\n== Sparsity Checks (JumpReLU L0 surrogate) ==")
    model.train()
    with torch.no_grad():
        _, sp1 = model(x1)
        _, sp2 = model(x2)

    # l0_hat (actual count, via encode)
    _, info1 = model.encode(x1)
    _, info2 = model.encode(x2)

    print(f"batch1 sparsity_loss=(l0_hat−target)²={float(sp1):.4f}  "
          f"l0_hat={float(info1['l0_hat']):.2f}  target_l0={model.encoder.target_l0}")
    print(f"batch2 sparsity_loss=(l0_hat−target)²={float(sp2):.4f}  "
          f"l0_hat={float(info2['l0_hat']):.2f}  target_l0={model.encoder.target_l0}")

    model.eval()
    with torch.no_grad():
        _, sp_eval = model(x1)
        _, info_eval = model.encode(x1)
    print(f"eval sparsity_loss={float(sp_eval):.4f}  l0_hat={float(info_eval['l0_hat']):.2f}")


def check_latent_properties(
    model: JumpReLUSparseConvAutoencoder, x: torch.Tensor
) -> None:
    print("\n== Latent Properties (JumpReLU) ==")
    model.eval()
    with torch.no_grad():
        latent, info = model.encode(x)

    total    = latent.numel()
    zeros    = (latent == 0).sum().item()
    B, C, H, W = latent.shape
    frac_zero = zeros / total

    print(f"latent_shape={tuple(latent.shape)}")
    print(f"zero_fraction={frac_zero:.4f}  "
          f"expected_approx={(C - model.encoder.target_l0) / C:.4f} "
          f"from target_l0={model.encoder.target_l0}/C={C}")
    print(f"latent_min={float(latent.min()):.5f}  latent_max={float(latent.max()):.5f}  "
          f"latent_mean={float(latent.mean()):.5f}  latent_std={float(latent.std()):.5f}")
    print(f"l0_hat (mean active/pos)={float(info['l0_hat']):.2f}  "
          f"target_l0={model.encoder.target_l0}")

    theta = model.encoder.log_theta.exp()
    print(f"theta_mean={float(theta.mean()):.5f}  "
          f"theta_min={float(theta.min()):.5f}  "
          f"theta_max={float(theta.max()):.5f}")


def print_architecture_diagram(model: JumpReLUSparseConvAutoencoder, details: dict) -> None:
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

    model = JumpReLUSparseConvAutoencoder(
        in_channels=3,
        resnet_variant=args.resnet_variant,
        stage_channels=tuple(args.stage_channels),
        stage_strides=tuple(args.stage_strides),
        target_l0=args.target_l0,
        bandwidth=args.bandwidth,
        theta_init=args.theta_init,
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
    print(f"sparsity_loss (l0_hat−target)²={float(sparsity):.5f}")
    print(f"shapes_match={tuple(x1.shape) == tuple(recon.shape)}")

    print_architecture_diagram(model, details)
    check_sparsity(model, x1, x2)
    check_latent_properties(model, x1)


if __name__ == "__main__":
    main()
