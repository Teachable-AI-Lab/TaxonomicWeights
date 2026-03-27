#!/usr/bin/env python3
"""Run SAEBench evaluations on trained TaxonSAE / MultiTaxonSAE checkpoints.

Loads our custom SAEs and runs them through SAEBench's eval suite alongside
pre-trained baselines from HuggingFace for direct comparison.

Usage:
    python src/eval/run_saebench_evals.py \
        --checkpoint outputs/saebench/taxon_sae_L8_dkl1e-02_t0p5/checkpoints/best.pt \
        --variant taxon \
        --eval-types core sparse_probing scr tpp absorption
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.core.main as core
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.sparse_probing_sae_probes.main as sparse_probing_sae_probes
import sae_bench.sae_bench_utils.general_utils as general_utils
from sae_bench.custom_saes.run_all_evals_dictionary_learning_saes import (
    get_all_hf_repo_autoencoders,
    load_dictionary_learning_sae,
)

from src.model.saebench.taxon_sae import TaxonSAE
from src.model.saebench.multi_taxon_sae import MultiTaxonSAE

RANDOM_SEED = 42

# SAEBench baseline repos per model
BASELINE_REPOS = {
    "pythia-160m-deduped": {
        "4k": "adamkarvonen/saebench_pythia-160m-deduped_width-2pow12_date-0108",
        "16k": "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108",
    },
    "gemma-2-2b": {
        "4k": "adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108",
        "16k": "canrager/saebench_gemma-2-2b_width-2pow14_date-0107",
    },
}

MODEL_CONFIGS = {
    "pythia-160m-deduped": {
        "batch_size": 256,
        "dtype": "float32",
        "layers": [8],
        "d_model": 768,
    },
    "pythia-70m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
        "layers": [3, 4],
        "d_model": 512,
    },
    "gemma-2-2b": {
        "batch_size": 32,
        "dtype": "bfloat16",
        "layers": [12],
        "d_model": 2304,
    },
}


def load_custom_sae(
    checkpoint: str,
    variant: str,
    device: torch.device,
    dtype: torch.dtype,
) -> TaxonSAE | MultiTaxonSAE:
    """Load a trained TaxonSAE or MultiTaxonSAE from checkpoint."""
    if variant == "taxon":
        sae = TaxonSAE.from_checkpoint(checkpoint, device, dtype)
    elif variant == "multi_taxon":
        sae = MultiTaxonSAE.from_checkpoint(checkpoint, device, dtype)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    sae.eval()
    normalized = sae.check_decoder_norms()
    if not normalized:
        print("Normalizing decoder vectors ...")
        sae.normalize_decoder()

    return sae


def load_baselines(
    model_name: str,
    hook_layer: int,
    device: torch.device,
    dtype: torch.dtype,
    width: str = "4k",
    max_baselines: int = 6,
) -> list[tuple[str, object]]:
    """Load SAEBench baseline SAEs from HuggingFace for comparison."""
    model_repos = BASELINE_REPOS.get(model_name, {})
    if width not in model_repos:
        print(f"No baseline repo for {model_name} width={width}, skipping baselines")
        return []

    repo_id = model_repos[width]
    print(f"Loading baselines from {repo_id} ...")

    try:
        locations = get_all_hf_repo_autoencoders(repo_id)
    except Exception as e:
        print(f"Could not download baseline repo: {e}")
        return []

    # Filter to the correct layer
    layer_str = f"layer_{hook_layer}"
    locations = [loc for loc in locations if layer_str in loc]

    # Limit number of baselines
    locations = locations[:max_baselines]

    selected = []
    for loc in locations:
        try:
            sae = load_dictionary_learning_sae(
                repo_id=repo_id,
                location=loc,
                model_name=model_name,
                device=str(device),
                dtype=dtype,
                layer=hook_layer,
            )
            name = f"baseline_{loc.split('/')[-1] if '/' in loc else loc}"
            selected.append((name, sae))
            print(f"  Loaded baseline: {name}")
        except Exception as e:
            print(f"  Failed to load {loc}: {e}")

    return selected


def run_evals(
    model_name: str,
    selected_saes: list[tuple[str, object]],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    force_rerun: bool = False,
    save_activations: bool = False,
    output_base: str = "eval_results",
) -> None:
    """Run selected SAEBench evaluations."""

    eval_runners = {
        "absorption": (
            lambda: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                f"{output_base}/absorption",
                force_rerun,
            )
        ),
        "core": (
            lambda: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder=f"{output_base}/core",
                verbose=True,
                dtype=llm_dtype,
                device=device,
            )
        ),
        "scr": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    perform_scr=True,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                output_base,
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "tpp": (
            lambda: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    perform_scr=False,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                output_base,
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "sparse_probing": (
            lambda: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=RANDOM_SEED,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                f"{output_base}/sparse_probing",
                force_rerun,
                clean_up_activations=True,
                save_activations=save_activations,
            )
        ),
        "sparse_probing_sae_probes": (
            lambda: sparse_probing_sae_probes.run_eval(
                sparse_probing_sae_probes.SparseProbingSaeProbesEvalConfig(
                    model_name=model_name,
                ),
                selected_saes,
                device,
                f"{output_base}/sparse_probing_sae_probes",
                force_rerun,
            )
        ),
    }

    for eval_type in eval_types:
        if eval_type not in eval_runners:
            print(f"Unknown eval type: {eval_type}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"Running {eval_type} evaluation")
        print(f"{'='*60}\n")
        os.makedirs(f"{output_base}/{eval_type}", exist_ok=True)
        eval_runners[eval_type]()


def checkpoint_from_config(config_path: str) -> tuple[str, str, str]:
    """Infer checkpoint path, variant, and model_name from a training config.

    Returns (checkpoint, variant, model_name).
    """
    with open(config_path) as f:
        cfg = json.load(f)

    model_variant = cfg["model"]["model_variant"]       # "taxon_sae" or "multi_taxon_sae"
    variant = model_variant.replace("_sae", "")          # "taxon" or "multi_taxon"
    model_name = cfg["data"]["model_name"]
    output_dir = cfg["output"]["output_dir"]

    L = cfg["model"]["n_taxonomy_layers"]
    dkl = cfg["training"]["dkl_weight"]
    temp = cfg["model"]["temperature"]
    hard = cfg["model"].get("hard", False)

    dkl_suffix = f"_dkl{dkl:.0e}"
    temp_str = f"{temp:g}".replace(".", "p")
    hard_str = "_hard" if hard else ""

    if variant == "multi_taxon":
        K = cfg["model"]["n_hierarchies"]
        suffix = f"_L{L}_K{K}{dkl_suffix}_t{temp_str}{hard_str}"
    else:
        suffix = f"_L{L}{dkl_suffix}_t{temp_str}{hard_str}"

    checkpoint = f"{output_dir}{suffix}/checkpoints/best.pt"
    return checkpoint, variant, model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAEBench evaluations on TaxonSAE / MultiTaxonSAE"
    )
    parser.add_argument("--config", type=str, default="",
                        help="Training config JSON; infers --checkpoint, --variant, --model-name")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to trained TaxonSAE/MultiTaxonSAE checkpoint")
    parser.add_argument("--variant", type=str, default="",
                        choices=["taxon", "multi_taxon", ""],
                        help="Model variant")
    parser.add_argument("--sae-name", type=str, default="",
                        help="Name for this SAE in results (auto-generated if empty)")
    parser.add_argument("--model-name", type=str, default="",
                        help="TransformerLens model name")
    parser.add_argument("--eval-types", nargs="+",
                        default=["core", "sparse_probing", "scr", "tpp", "absorption"],
                        help="Which SAEBench evals to run")
    parser.add_argument("--include-baselines", action="store_true", default=False,
                        help="Also load and evaluate SAEBench baseline SAEs for comparison")
    parser.add_argument("--baseline-width", type=str, default="4k",
                        choices=["4k", "16k"],
                        help="Width of baseline SAEs to compare against")
    parser.add_argument("--max-baselines", type=int, default=6,
                        help="Max number of baseline SAEs to load")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Base directory for eval result JSONs")
    parser.add_argument("--force-rerun", action="store_true", default=False)
    parser.add_argument("--save-activations", action="store_true", default=False)

    args = parser.parse_args()

    # Infer from config if provided
    if args.config:
        ckpt, variant, model_name = checkpoint_from_config(args.config)
        if not args.checkpoint:
            args.checkpoint = ckpt
        if not args.variant:
            args.variant = variant
        if not args.model_name:
            args.model_name = model_name

    # Validate required fields
    if not args.checkpoint:
        parser.error("--checkpoint is required (or provide --config to infer it)")
    if not args.variant:
        parser.error("--variant is required (or provide --config to infer it)")
    if not args.model_name:
        args.model_name = "pythia-160m-deduped"

    return args


def main() -> None:
    args = parse_args()

    device = general_utils.setup_environment()

    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {args.model_name}")

    mc = MODEL_CONFIGS[args.model_name]
    llm_batch_size = mc["batch_size"]
    llm_dtype = mc["dtype"]
    dtype = general_utils.str_to_dtype(llm_dtype)

    # Load our custom SAE
    print(f"\nLoading {args.variant} SAE from {args.checkpoint} ...")
    sae = load_custom_sae(args.checkpoint, args.variant, torch.device(device), dtype)
    sae_name = args.sae_name or f"{args.variant}_sae_{Path(args.checkpoint).parent.parent.name}"
    sae = sae.to(dtype=dtype)
    sae.cfg.dtype = llm_dtype

    selected_saes = [(sae_name, sae)]
    print(f"  Custom SAE: {sae_name} (d_sae={sae.cfg.d_sae})")

    # Optionally load baselines
    if args.include_baselines:
        hook_layer = sae.cfg.hook_layer
        baselines = load_baselines(
            args.model_name, hook_layer, torch.device(device), dtype,
            width=args.baseline_width, max_baselines=args.max_baselines,
        )
        for bname, bsae in baselines:
            bsae = bsae.to(dtype=dtype)
            bsae.cfg.dtype = llm_dtype
        selected_saes.extend(baselines)

    print(f"\nEvaluating {len(selected_saes)} SAEs: {[n for n, _ in selected_saes]}")

    run_evals(
        model_name=args.model_name,
        selected_saes=selected_saes,
        llm_batch_size=llm_batch_size,
        llm_dtype=llm_dtype,
        device=device,
        eval_types=args.eval_types,
        force_rerun=args.force_rerun,
        save_activations=args.save_activations,
        output_base=args.output_dir,
    )

    print(f"\nAll evaluations complete. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
