#!/usr/bin/env python3
"""Detailed command-line test for TaxonAutoencoder architecture and taxonomy behavior."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

import sys

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.taxon_ae import TaxonAutoencoder
from src.model.encoder import resolve_resnet_stage_blocks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test TaxonAutoencoder architecture and taxonomy constraints")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hard", action="store_true", help="Use hard straight-through routing")
    parser.add_argument("--resnet-variant", type=str, default="18")
    parser.add_argument("--stage-taxonomy-layers", type=int, nargs=4, default=[5, 6, 7, 8])
    parser.add_argument("--stage-strides", type=int, nargs=4, default=[1, 2, 2, 2])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_taxonomy_structure(model: TaxonAutoencoder) -> None:
    print("\n== Taxonomy Stage Structure ==")
    for idx, stage in enumerate(model.encoder.taxon_stages, start=1):
        channels = ", ".join(str(c) for c in stage.layer_channels)
        print(
            f"stage{idx}: blocks={stage.n_blocks} stride={stage.stride} "
            f"taxonomy_depth={stage.n_taxonomy_layers} layer_channels=[{channels}] "
            f"total_out={stage.total_out_channels}"
        )


def print_architecture_diagram(model: TaxonAutoencoder, details: dict) -> None:
    print("\n== Architecture Diagram (Shape Trace) ==")
    print(f"Input: {details['input_shape']}")

    enc_trace = details["encoder"].get("shape_trace", [])
    for name, shape in enc_trace:
        print(f"  -> Encoder {name}: {shape}")

    print(f"Latent: {details['latent_shape']}")

    dec_trace = details["decoder"].get("shape_trace", [])
    for name, shape in dec_trace:
        print(f"  -> Decoder {name}: {shape}")

    print(f"Reconstruction: {details['recon_shape']}")


def check_resnet_alignment(model: TaxonAutoencoder, variant: str) -> None:
    expected = resolve_resnet_stage_blocks(variant)
    actual = tuple(model.encoder.stage_blocks)
    print("\n== ResNet Alignment Check ==")
    print(f"variant={variant} expected_stage_blocks={expected} actual_stage_blocks={actual}")
    print(f"is_resnet_aligned={expected == actual}")
    print("Interpretation: encoder keeps ResNet stage block counts; taxonomy is applied to each stage output.")


def check_taxonomy_mass(model: TaxonAutoencoder, x: torch.Tensor, hard: bool) -> None:
    print("\n== Taxonomy Probability Checks ==")
    model.eval()

    with torch.no_grad():
        stage_input = model.encoder.stem(x)

        for stage_idx, stage in enumerate(model.encoder.taxon_stages, start=1):
            stage_output, stage_logp, _ = stage(stage_input, hard=hard)
            split_logps = torch.split(stage_logp, stage.layer_channels, dim=1)
            prev = None
            depth_mass_err_max = 0.0
            parent_err_max = 0.0
            for depth_idx, depth_logp in enumerate(split_logps, start=1):
                p = depth_logp.exp()
                depth_mass_err = (p.sum(dim=1) - 1.0).abs().max().item()
                depth_mass_err_max = max(depth_mass_err_max, depth_mass_err)
                if prev is not None:
                    pair_mass = p.view(p.shape[0], -1, 2, p.shape[2], p.shape[3]).sum(dim=2)
                    parent = prev.exp()
                    parent_err = (pair_mass - parent).abs().max().item()
                    parent_err_max = max(parent_err_max, parent_err)
                prev = depth_logp

            print(
                f"stage{stage_idx}: max_depth|sum_c p(c|x)-1|={depth_mass_err_max:.3e}, "
                f"max|child_pair-parent|={parent_err_max:.3e}, logp_shape={tuple(stage_logp.shape)}"
            )

            report = stage.evaluate_path_encoding(stage_input, hard=hard)
            print(
                f"  stage{stage_idx} detailed_ok={report['ok']} "
                f"dkl={report['dkl']:.6f} entropy={report['entropy']:.6f}"
            )

            stage_input = stage_output


def compare_hard_dkl(model: TaxonAutoencoder, x1: torch.Tensor, x2: torch.Tensor) -> None:
    print("\n== Hard/Soft DKL Comparison ==")
    model.eval()
    with torch.no_grad():
        _, dkl_soft_1, ent_soft_1 = model(x1, hard=False)
        _, dkl_soft_2, ent_soft_2 = model(x2, hard=False)
        _, dkl_hard_1, ent_hard_1 = model(x1, hard=True)
        _, dkl_hard_2, ent_hard_2 = model(x2, hard=True)

    print(f"soft batch1: dkl={float(dkl_soft_1):.6f}, entropy={float(ent_soft_1):.6f}")
    print(f"soft batch2: dkl={float(dkl_soft_2):.6f}, entropy={float(ent_soft_2):.6f}")
    print(f"hard batch1: dkl={float(dkl_hard_1):.6f}, entropy={float(ent_hard_1):.6f}")
    print(f"hard batch2: dkl={float(dkl_hard_2):.6f}, entropy={float(ent_hard_2):.6f}")
    print("Note: aggregated DKL should vary across batches even with hard=True.")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = TaxonAutoencoder(
        in_channels=3,
        resnet_variant=args.resnet_variant,
        stage_taxonomy_layers=tuple(args.stage_taxonomy_layers),
        stage_strides=tuple(args.stage_strides),
        temperature=args.temperature,
        hard=args.hard,
        use_stem=True,
        stem_channels=64,
        stem_stride=2,
        use_stem_maxpool=True,
        output_activation="tanh",
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")

    total_params, trainable_params = count_parameters(model)

    print("== Model Summary ==")
    print(model)
    print(f"\nparameter_count_total={total_params:,}")
    print(f"parameter_count_trainable={trainable_params:,}")

    check_resnet_alignment(model, args.resnet_variant)
    print_taxonomy_structure(model)

    x1 = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    x2 = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

    with torch.no_grad():
        recon, dkl, entropy, details = model(x1, hard=args.hard, return_details=True)

    print("\n== Forward Pass ==")
    print(f"input_shape={tuple(x1.shape)}")
    print(f"recon_shape={tuple(recon.shape)}")
    print(f"dkl={float(dkl):.6f}")
    print(f"entropy={float(entropy):.6f}")

    print_architecture_diagram(model, details)
    check_taxonomy_mass(model, x1, hard=args.hard)
    compare_hard_dkl(model, x1, x2)


if __name__ == "__main__":
    main()
