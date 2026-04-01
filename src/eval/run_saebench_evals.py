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
import math
import os
import sys
from pathlib import Path

import numpy as np
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
from src.model.saebench.topk_taxon_sae import TopKTaxonSAE
from src.model.saebench.topk_multi_taxon_sae import TopKMultiTaxonSAE

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
) -> TaxonSAE | MultiTaxonSAE | TopKTaxonSAE | TopKMultiTaxonSAE:
    """Load a trained TaxonSAE or MultiTaxonSAE from checkpoint."""
    if variant == "taxon":
        sae = TaxonSAE.from_checkpoint(checkpoint, device, dtype)
    elif variant == "multi_taxon":
        sae = MultiTaxonSAE.from_checkpoint(checkpoint, device, dtype)
    elif variant == "topk_taxon":
        sae = TopKTaxonSAE.from_checkpoint(checkpoint, device, dtype)
    elif variant == "topk_multi_taxon":
        sae = TopKMultiTaxonSAE.from_checkpoint(checkpoint, device, dtype)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    sae.eval()
    normalized = sae.check_decoder_norms()
    if not normalized:
        print("Normalizing decoder vectors ...")
        sae.normalize_decoder()

    return sae


_TRAINER_DISPLAY: dict[str, str] = {
    "BatchTopKTrainer": "BatchTopK",
    "TopKTrainer": "TopK",
    "MatryoshkaBatchTopKTrainer": "MatryoshkaBatchTopK",
    "JumpReluTrainer": "JumpReLU",
    "StandardTrainerAprilUpdate": "Standard",
    "PAnnealTrainer": "PAnneal",
    "GatedSAETrainer": "GatedSAE",
}


def _baseline_name_from_config(config_path: str, fallback: str) -> str:
    """Derive a human-readable baseline name from a DL-SAE config.json."""
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        t = cfg.get("trainer", {})
        tc = t.get("trainer_class", "")
        display = _TRAINER_DISPLAY.get(tc) or tc.removesuffix("Trainer") or tc
        # k-based trainers (TopK, JumpReLU target_l0)
        k = t.get("k") if t.get("k") is not None else t.get("target_l0")
        if k is not None:
            return f"baseline_{display}_k{k}"
        # L1-penalty-based trainers
        l1 = t.get("l1_penalty") if t.get("l1_penalty") is not None else t.get("sparsity_penalty")
        if l1 is not None:
            return f"baseline_{display}_l1_{l1}"
        return f"baseline_{display}"
    except Exception:
        return fallback


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
            config_path = os.path.join(
                "downloaded_saes", repo_id.replace("/", "_"), loc, "config.json"
            )
            fallback = f"baseline_{loc.split('/')[-1] if '/' in loc else loc}"
            name = _baseline_name_from_config(config_path, fallback)
            selected.append((name, sae))
            print(f"  Loaded baseline: {name}")
        except Exception as e:
            print(f"  Failed to load {loc}: {e}")

    return selected


# ── Taxonomy-specific activation analysis ─────────────────────────────────


@torch.no_grad()
def run_taxonomy_analysis(
    selected_saes: list[tuple[str, object]],
    model_name: str,
    device: str,
    output_base: str,
    n_batches: int = 200,
    context_size: int = 128,
) -> None:
    """Analyse per-depth activation statistics for TaxonSAE / MultiTaxonSAE.

    For each taxonomy SAE in *selected_saes*, computes:
      - per-depth L0 (mean #active features per token)
      - per-depth feature firing rate
      - routing path entropy (how decisive the tree is)
      - overall L0 with hard vs soft routing comparison
      - per-depth reconstruction contribution

    Writes a JSON report to ``output_base/taxonomy/<sae_name>.json``.
    Non-taxonomy SAEs (baselines) are silently skipped.
    """
    from sae_lens.training.activations_store import ActivationsStore
    from transformer_lens import HookedTransformer

    out_dir = Path(output_base) / "taxonomy"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model once for all SAEs
    mcfg = MODEL_CONFIGS.get(model_name, {})
    dtype = getattr(torch, mcfg.get("dtype", "float32"))
    print(f"[taxonomy] Loading {model_name} ({dtype}) for activation streaming ...")
    llm = HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)

    for sae_name, sae in selected_saes:
        encoder = getattr(sae, "encoder", None)
        if encoder is None:
            continue  # skip baselines

        # Determine if it's multi-hierarchy
        hierarchies = getattr(encoder, "hierarchies", None)
        is_multi = hierarchies is not None

        # The single-tree stage used for stats
        stage = hierarchies[0] if is_multi else encoder
        n_layers = stage.n_taxonomy_layers
        layer_channels = stage.layer_channels  # [2, 4, 8, ..., 2^L]
        d_sae = sae.cfg.d_sae

        print(f"\n[taxonomy] Analysing {sae_name}  (d_sae={d_sae}, "
              f"n_layers={n_layers}, multi={is_multi})")

        # Activation store for streaming batches
        store_batch = max(1, mcfg.get("batch_size", 256) // context_size)
        act_store = ActivationsStore(
            model=llm,
            dataset="Skylion007/openwebtext",
            streaming=True,
            hook_name=sae.cfg.hook_name,
            hook_head_index=None,
            context_size=context_size,
            d_in=sae.cfg.d_in,
            n_batches_in_buffer=32,
            total_training_tokens=n_batches * mcfg.get("batch_size", 256),
            store_batch_size_prompts=store_batch,
            train_batch_size_tokens=mcfg.get("batch_size", 256),
            prepend_bos=True,
            normalize_activations="none",
            device=torch.device(device),
            dtype=dtype,
        )

        # For multi-taxon, z_hard is [B, K * sum(layer_channels)], so build
        # a flat split list that covers all K hierarchies.
        n_hierarchies = len(hierarchies) if is_multi else 1
        flat_split_sizes = layer_channels * n_hierarchies  # repeat K times

        # Accumulators
        total_tokens = 0
        # Per-depth: aggregate across hierarchies
        depth_active_counts = [0.0 for _ in layer_channels]
        depth_magnitude_sums = [0.0 for _ in layer_channels]
        depth_firing_any = [torch.zeros(ch * n_hierarchies, device=device)
                           for ch in layer_channels]
        overall_l0_hard = 0.0
        overall_l0_soft = 0.0
        routing_entropy_sum = 0.0
        recon_mse_total = 0.0

        for batch_idx in range(n_batches):
            x = act_store.next_batch().to(device=device, dtype=sae.W_dec.dtype)
            flat = x.reshape(-1, x.shape[-1])
            B = flat.shape[0]
            total_tokens += B

            centred = flat - sae.b_dec

            # Hard-routing encode (for L0)
            z_hard, _ = encoder(centred, hard=True)
            overall_l0_hard += (z_hard > 0).float().sum(dim=-1).sum().item()

            # Soft-routing encode (for comparison)
            z_soft, _ = encoder(centred, hard=False)
            overall_l0_soft += (z_soft > 0).float().sum(dim=-1).sum().item()

            # Reconstruction MSE
            x_hat = z_hard @ sae.W_dec + sae.b_dec
            recon_mse_total += ((flat - x_hat) ** 2).sum().item()

            # Per-depth analysis (on hard-routed output)
            # Split into [n_layers * n_hierarchies] chunks, then aggregate
            # by depth across hierarchies.
            all_splits = torch.split(z_hard, flat_split_sizes, dim=1)
            for d in range(n_layers):
                # Gather the d-th depth from each hierarchy
                chunks = [all_splits[d + k * n_layers] for k in range(n_hierarchies)]
                z_d = torch.cat(chunks, dim=1)  # [B, ch * n_hierarchies]
                active = (z_d > 0).float()
                depth_active_counts[d] += active.sum(dim=-1).sum().item()
                depth_magnitude_sums[d] += z_d.sum().item()
                depth_firing_any[d] += active.any(dim=0).float()

            # Routing entropy: run encoder internals to get path probs
            if not is_multi:
                all_logits = encoder.linear(centred)
                d_logits = torch.split(all_logits, layer_channels, dim=1)
                is_topk_stage = not hasattr(encoder, "_pairwise_log_softmax")
                prev_logp = None
                for d, lg in enumerate(d_logits):
                    if is_topk_stage:
                        # TopK stage: compute entropy from soft probs directly
                        probs = encoder._pairwise_softmax(lg, encoder.temperature, False)
                        probs = probs.clamp(min=1e-8)
                        logp = probs.log()
                    else:
                        logp = encoder._pairwise_log_softmax(lg, encoder.temperature, False)
                    logp = logp if prev_logp is None else logp + prev_logp.repeat_interleave(2, dim=1)
                    prob = logp.exp()
                    ent = -(prob * logp).sum(dim=1).mean().item()
                    routing_entropy_sum += ent
                    prev_logp = logp

        # ── Summarise ──────────────────────────────────────────────────
        mean_l0_hard = overall_l0_hard / total_tokens
        mean_l0_soft = overall_l0_soft / total_tokens
        mean_recon_mse = recon_mse_total / total_tokens

        per_depth = []
        for d, ch in enumerate(layer_channels):
            total_ch = ch * n_hierarchies
            l0_d = depth_active_counts[d] / total_tokens
            mean_mag = depth_magnitude_sums[d] / max(depth_active_counts[d], 1)
            alive_pct = (depth_firing_any[d] > 0).float().mean().item() * 100
            per_depth.append({
                "depth": d,
                "channels": ch,
                "total_channels": total_ch,
                "l0": round(l0_d, 3),
                "mean_magnitude": round(mean_mag, 5),
                "alive_features_pct": round(alive_pct, 1),
            })

        report = {
            "sae_name": sae_name,
            "d_sae": d_sae,
            "n_taxonomy_layers": n_layers,
            "is_multi_hierarchy": is_multi,
            "n_hierarchies": n_hierarchies,
            "total_tokens": total_tokens,
            "mean_l0_hard_routing": round(mean_l0_hard, 2),
            "mean_l0_soft_routing": round(mean_l0_soft, 2),
            "mean_recon_mse": round(mean_recon_mse, 6),
            "mean_routing_entropy": round(routing_entropy_sum / (n_batches * n_layers), 5)
                if not is_multi else None,
            "per_depth": per_depth,
        }

        out_path = out_dir / f"{sae_name}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  → saved {out_path}")

        # Print summary
        print(f"  L0 (hard): {mean_l0_hard:.1f}  |  L0 (soft): {mean_l0_soft:.1f}  "
              f"|  MSE: {mean_recon_mse:.5f}")
        for d_info in per_depth:
            print(f"    depth {d_info['depth']:2d}  "
                  f"(ch={d_info['channels']:5d}): "
                  f"L0={d_info['l0']:.2f}  "
                  f"alive={d_info['alive_features_pct']:.0f}%  "
                  f"mag={d_info['mean_magnitude']:.4f}")


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

    # Pre-create artifacts directories that SAEBench SCR/TPP expects
    if save_activations and selected_saes:
        _, first_sae = selected_saes[0]
        hook_name = getattr(getattr(first_sae, "cfg", None), "hook_name", None)
        if hook_name:
            for eval_type in ("scr", "tpp"):
                os.makedirs(
                    os.path.join("artifacts", eval_type, model_name, hook_name),
                    exist_ok=True,
                )

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
        "taxonomy": (
            lambda: run_taxonomy_analysis(
                selected_saes=selected_saes,
                model_name=model_name,
                device=device,
                output_base=output_base,
            )
        ),
    }

    for eval_type in eval_types:
        if eval_type not in eval_runners:
            print(f"Unknown eval type: {eval_type}, skipping")
            continue
        try:
            print(f"\n{'='*60}")
            print(f"Running {eval_type} evaluation")
            print(f"{'='*60}\n")
        except OSError:
            pass
        os.makedirs(f"{output_base}/{eval_type}", exist_ok=True)
        try:
            eval_runners[eval_type]()
        except OSError as e:
            if e.errno in (39, 116):  # 39=Dir not empty (NFS), 116=Stale file handle
                print(f"  Non-fatal OSError (errno {e.errno}) during {eval_type}, continuing: {e}")
            else:
                raise


def checkpoint_from_config(config_path: str) -> tuple[str, str, str]:
    """Infer checkpoint path, variant, and model_name from a training config.

    Returns (checkpoint, variant, model_name).
    """
    with open(config_path) as f:
        cfg = json.load(f)

    model_variant = cfg["model"]["model_variant"]       # "taxon_sae" / "multi_taxon_sae" / "topk_taxon_sae" / "topk_multi_taxon_sae"
    variant = model_variant.replace("_sae", "")          # "taxon" / "multi_taxon" / "topk_taxon" / "topk_multi_taxon"
    model_name = cfg["data"]["model_name"]
    output_dir = cfg["output"]["output_dir"]

    L = cfg["model"]["n_taxonomy_layers"]
    temp = cfg["model"]["temperature"]
    hard = cfg["model"].get("hard", False)
    temp_str = f"{temp:g}".replace(".", "p")
    hard_str = "_hard" if hard else ""

    is_topk = variant.startswith("topk_")

    if is_topk:
        # TopK variants: no dkl suffix
        if variant == "topk_multi_taxon":
            K = cfg["model"]["n_hierarchies"]
            gate_k = cfg["model"].get("gate_k", 4)
            suffix = f"_L{L}_K{K}_gk{gate_k}_t{temp_str}{hard_str}"
        else:
            suffix = f"_L{L}_t{temp_str}{hard_str}"
    else:
        # DKL variants
        dkl = cfg["training"]["dkl_weight"]
        dkl_suffix = f"_dkl{dkl:.0e}"
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
                        choices=["taxon", "multi_taxon", "topk_taxon", "topk_multi_taxon", ""],
                        help="Model variant")
    parser.add_argument("--sae-name", type=str, default="",
                        help="Name for this SAE in results (auto-generated if empty)")
    parser.add_argument("--model-name", type=str, default="",
                        help="TransformerLens model name")
    parser.add_argument("--eval-types", nargs="+",
                        default=["core", "sparse_probing", "scr", "tpp", "absorption", "taxonomy"],
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
        # Auto-derive output dir from checkpoint path
        if args.output_dir == "eval_results":
            args.output_dir = str(
                Path(args.checkpoint).parent.parent / "eval_results"
            )

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
    sae_name = args.sae_name or Path(args.checkpoint).parent.parent.name
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
