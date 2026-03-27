#!/usr/bin/env python3
"""Compare SAEBench evaluation results across SAE variants with graphics.

Loads multiple trained SAEs from their training configs, optionally adds
pre-trained baselines, runs SAEBench evaluations, and generates comparison
graphics.

Usage:
    # Pythia comparison (our models + baselines):
    python src/eval/compare_saebench.py \
        --configs configs/saebench/taxon_sae_pythia160m.json \
                  configs/saebench/multi_taxon_sae_pythia160m.json \
        --include-baselines --baseline-width 4k \
        --eval-types core sparse_probing scr tpp absorption

    # Generate graphics only (skip re-evaluation):
    python src/eval/compare_saebench.py \
        --configs configs/saebench/taxon_sae_pythia160m.json \
                  configs/saebench/multi_taxon_sae_pythia160m.json \
        --include-baselines --baseline-width 4k \
        --skip-eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sae_bench.sae_bench_utils.general_utils as general_utils

from src.eval.run_saebench_evals import (
    load_custom_sae,
    load_baselines,
    run_evals,
    checkpoint_from_config,
    MODEL_CONFIGS,
)

RANDOM_SEED = 42

# ── Colour palette ───────────────────────────────────────────────────────────

_CUSTOM_COLOURS = {
    "taxon": "#1f77b4",
    "multi_taxon": "#9467bd",
}
_BASELINE_COLOURS = [
    "#ff7f0e", "#2ca02c", "#d62728", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _sae_colour(name: str, baseline_idx: int = 0) -> str:
    nl = name.lower()
    if "multi_taxon" in nl:
        return _CUSTOM_COLOURS["multi_taxon"]
    if "taxon" in nl:
        return _CUSTOM_COLOURS["taxon"]
    return _BASELINE_COLOURS[baseline_idx % len(_BASELINE_COLOURS)]


# ── Result collection ───────────────────────────────────────────────────────

def _deep_get(d: dict, *keys, default=None):
    """Nested dict access."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def collect_results(output_base: str, sae_names: list[str]) -> dict[str, dict]:
    """Scan eval result JSONs and aggregate metrics per SAE per eval type.

    Returns: {sae_name: {eval_type: parsed_json_dict}}
    """
    base = Path(output_base)
    results: dict[str, dict] = {}

    for json_path in sorted(base.rglob("*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        # Determine eval type from the JSON itself
        eval_type = data.get("eval_type_id", "")
        sae_id = data.get("sae_lens_id", json_path.stem)

        # Try to match sae_id to a known sae_name
        matched_name = None
        for name in sae_names:
            if name == sae_id or name in sae_id or sae_id in name:
                matched_name = name
                break
        if matched_name is None:
            # Fall back to stem matching
            for name in sae_names:
                if json_path.stem == name or name in json_path.stem:
                    matched_name = name
                    break
        if matched_name is None:
            continue

        if matched_name not in results:
            results[matched_name] = {}
        results[matched_name][eval_type] = data

    return results


def _extract_metric(data: dict, *paths) -> float | None:
    """Try multiple paths into a result dict, return first found float."""
    for path in paths:
        val = _deep_get(data, *path)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


# ── Graphics ─────────────────────────────────────────────────────────────────

def _bar_chart(ax, names, values, colours, title, ylabel="", fmt=".4f"):
    bars = ax.bar(range(len(names)), values, color=colours,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:{fmt}}", ha="center", va="bottom", fontsize=7)


def _short_name(name: str) -> str:
    """Shorten SAE name for display."""
    n = name
    for prefix in ("taxon_sae_", "multi_taxon_sae_", "baseline_"):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    # Capitalize type prefix
    nl = name.lower()
    if "multi_taxon" in nl:
        return f"MultiTaxon ({n})" if n != name else "MultiTaxonSAE"
    elif "taxon" in nl:
        return f"Taxon ({n})" if n != name else "TaxonSAE"
    else:
        return name


def generate_graphics(
    results: dict[str, dict],
    sae_names: list[str],
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Generate comparison graphics from collected eval results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to SAEs with results and assign colours
    display_names, short_names, colours = [], [], []
    baseline_idx = 0
    for name in sae_names:
        if name not in results or not results[name]:
            continue
        display_names.append(name)
        short_names.append(_short_name(name))
        c = _sae_colour(name, baseline_idx)
        if "baseline" in name.lower():
            baseline_idx += 1
        colours.append(c)

    if not display_names:
        print("No results found to plot.")
        return

    # ── CORE metrics ─────────────────────────────────────────────────────
    core_data = {n: results[n].get("core", {}) for n in display_names}
    core_metrics = [
        ("CE Loss Score", ("eval_result_metrics", "model_performance_preservation", "ce_loss_score"), ".4f"),
        ("KL Div Score", ("eval_result_metrics", "model_behavior_preservation", "kl_div_score"), ".4f"),
        ("Explained Variance", ("eval_result_metrics", "reconstruction_quality", "explained_variance"), ".4f"),
        ("L0 Sparsity", ("eval_result_metrics", "sparsity", "l0"), ".1f"),
    ]
    available_core = []
    for label, path, fmt in core_metrics:
        vals = [_deep_get(core_data[n], *path) for n in display_names]
        if any(v is not None for v in vals):
            available_core.append((label, [v if v is not None else 0.0 for v in vals], fmt))

    if available_core:
        ncols = len(available_core)
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
        if ncols == 1:
            axes = [axes]
        for ax, (label, vals, fmt) in zip(axes, available_core):
            _bar_chart(ax, short_names, vals, colours, label, fmt=fmt)
        fig.suptitle(f"{title_prefix}Core Evaluation Metrics", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(output_dir / "comparison_core.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> comparison_core.png")

    # ── SPARSE PROBING ───────────────────────────────────────────────────
    sp_data = {n: results[n].get("sparse_probing", {}) for n in display_names}
    sp_metrics = [
        ("SAE Accuracy (k=1)", ("eval_result_metrics", "sae", "sae_test_accuracy"), ".4f"),
        ("SAE Accuracy (k=5)", ("eval_result_metrics", "sae", "sae_test_accuracy_k5"), ".4f"),
        ("SAE Accuracy (k=10)", ("eval_result_metrics", "sae", "sae_test_accuracy_k10"), ".4f"),
    ]
    available_sp = []
    for label, path, fmt in sp_metrics:
        vals = [_deep_get(sp_data[n], *path) for n in display_names]
        if any(v is not None for v in vals):
            available_sp.append((label, [v if v is not None else 0.0 for v in vals], fmt))

    if available_sp:
        ncols = len(available_sp)
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
        if ncols == 1:
            axes = [axes]
        for ax, (label, vals, fmt) in zip(axes, available_sp):
            _bar_chart(ax, short_names, vals, colours, label, fmt=fmt)
        fig.suptitle(f"{title_prefix}Sparse Probing", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(output_dir / "comparison_sparse_probing.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> comparison_sparse_probing.png")

    # ── SCR & TPP ────────────────────────────────────────────────────────
    scr_tpp_types = [("scr", "SCR"), ("tpp", "TPP")]
    scr_tpp_available = []
    for eval_id, label_prefix in scr_tpp_types:
        et_data = {n: results[n].get(eval_id, {}) for n in display_names}
        metric_path = ("eval_result_metrics", eval_id, f"{eval_id}_dir1_threshold_2")
        vals = [_deep_get(et_data[n], *metric_path) for n in display_names]
        if any(v is not None for v in vals):
            scr_tpp_available.append(
                (f"{label_prefix} (dir1, top-2)", [v if v is not None else 0.0 for v in vals], ".4f")
            )

    if scr_tpp_available:
        ncols = len(scr_tpp_available)
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
        if ncols == 1:
            axes = [axes]
        for ax, (label, vals, fmt) in zip(axes, scr_tpp_available):
            _bar_chart(ax, short_names, vals, colours, label, fmt=fmt)
        fig.suptitle(f"{title_prefix}SCR & TPP", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(output_dir / "comparison_scr_tpp.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> comparison_scr_tpp.png")

    # ── ABSORPTION ───────────────────────────────────────────────────────
    abs_data = {n: results[n].get("absorption", {}) for n in display_names}
    abs_path = ("eval_result_metrics", "mean", "mean_absorption_fraction_score")
    abs_vals = [_deep_get(abs_data[n], *abs_path) for n in display_names]
    if any(v is not None for v in abs_vals):
        fig, ax = plt.subplots(figsize=(6, 5))
        _bar_chart(ax, short_names,
                   [v if v is not None else 0.0 for v in abs_vals],
                   colours, "Mean Absorption Fraction Score", fmt=".4f")
        fig.suptitle(f"{title_prefix}Absorption", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(output_dir / "comparison_absorption.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> comparison_absorption.png")

    # ── SUMMARY (all key metrics on one page) ────────────────────────────
    summary_metrics = []
    # Gather one key metric per eval type
    for name_label, data_dict, path, fmt in [
        ("CE Loss Score", core_data, ("eval_result_metrics", "model_performance_preservation", "ce_loss_score"), ".4f"),
        ("L0 Sparsity", core_data, ("eval_result_metrics", "sparsity", "l0"), ".0f"),
        ("Explained Var.", core_data, ("eval_result_metrics", "reconstruction_quality", "explained_variance"), ".4f"),
        ("Absorption", abs_data, ("eval_result_metrics", "mean", "mean_absorption_fraction_score"), ".4f"),
    ]:
        vals = [_deep_get(data_dict[n], *path) for n in display_names]
        if any(v is not None for v in vals):
            summary_metrics.append((name_label, [v if v is not None else 0.0 for v in vals], fmt))

    if summary_metrics:
        ncols = len(summary_metrics)
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        if ncols == 1:
            axes = [axes]
        for ax, (label, vals, fmt) in zip(axes, summary_metrics):
            _bar_chart(ax, short_names, vals, colours, label, fmt=fmt)
        fig.suptitle(f"{title_prefix}Summary Comparison", fontsize=15, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(output_dir / "comparison_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> comparison_summary.png")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAEBench evaluations across SAE variants"
    )
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Training config JSONs for our models")
    parser.add_argument("--include-baselines", action="store_true",
                        help="Also load SAEBench baseline SAEs for comparison")
    parser.add_argument("--baseline-width", type=str, default="4k",
                        choices=["4k", "16k"])
    parser.add_argument("--max-baselines", type=int, default=6)
    parser.add_argument("--eval-types", nargs="+",
                        default=["core", "sparse_probing", "scr", "tpp", "absorption"])
    parser.add_argument("--output-dir", type=str, default="",
                        help="Where to save results + graphics (auto from model name)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation, only generate graphics from existing results")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve all configs
    configs_info = []
    model_name = None
    for cfg_path in args.configs:
        ckpt, variant, mname = checkpoint_from_config(cfg_path)
        if model_name is None:
            model_name = mname
        elif mname != model_name:
            print(f"Warning: mixed models ({model_name} vs {mname}), using {model_name}")
        configs_info.append((ckpt, variant, mname))

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    mc = MODEL_CONFIGS[model_name]
    llm_batch_size = mc["batch_size"]
    llm_dtype = mc["dtype"]
    dtype = general_utils.str_to_dtype(llm_dtype)

    # Determine output directory
    if args.output_dir:
        output_base = args.output_dir
    else:
        model_short = model_name.replace("-deduped", "").replace("-", "_")
        output_base = f"eval_results/{model_short}_comparison"

    device = general_utils.setup_environment()

    # Load our custom SAEs
    sae_names = []
    selected_saes = []
    for ckpt, variant, _ in configs_info:
        print(f"\nLoading {variant} SAE from {ckpt} ...")
        sae = load_custom_sae(ckpt, variant, torch.device(device), dtype)
        sae_name = f"{variant}_sae_{Path(ckpt).parent.parent.name}"
        sae = sae.to(dtype=dtype)
        sae.cfg.dtype = llm_dtype
        selected_saes.append((sae_name, sae))
        sae_names.append(sae_name)
        print(f"  {sae_name} (d_sae={sae.cfg.d_sae})")

    # Load baselines
    if args.include_baselines:
        hook_layer = selected_saes[0][1].cfg.hook_layer
        baselines = load_baselines(
            model_name, hook_layer, torch.device(device), dtype,
            width=args.baseline_width, max_baselines=args.max_baselines,
        )
        for bname, bsae in baselines:
            bsae = bsae.to(dtype=dtype)
            bsae.cfg.dtype = llm_dtype
        selected_saes.extend(baselines)
        sae_names.extend([n for n, _ in baselines])

    print(f"\nSAEs to compare: {sae_names}")
    print(f"Output: {output_base}")

    # Run evaluations
    if not args.skip_eval:
        print(f"\nRunning SAEBench evaluations ...")
        run_evals(
            model_name=model_name,
            selected_saes=selected_saes,
            llm_batch_size=llm_batch_size,
            llm_dtype=llm_dtype,
            device=device,
            eval_types=args.eval_types,
            force_rerun=args.force_rerun,
            save_activations=args.save_activations,
            output_base=output_base,
        )
    else:
        print("\nSkipping evaluation (--skip-eval), loading existing results ...")

    # Collect results from disk
    print(f"\nCollecting results from {output_base} ...")
    results = collect_results(output_base, sae_names)

    n_with_results = sum(1 for n in sae_names if n in results and results[n])
    print(f"  Found results for {n_with_results}/{len(sae_names)} SAEs")

    if n_with_results == 0:
        print("No results found. Run without --skip-eval first.")
        return

    # Generate graphics
    gfx_dir = Path(output_base) / "graphics"
    model_short = model_name.replace("-deduped", "").replace("-", "_")
    title_prefix = f"{model_short} — "
    print(f"\nGenerating comparison graphics in {gfx_dir} ...")
    generate_graphics(results, sae_names, gfx_dir, title_prefix)

    # Save a summary CSV
    _save_summary_csv(results, sae_names, Path(output_base))

    print(f"\nDone. All outputs in {output_base}/")


def _save_summary_csv(results: dict, sae_names: list[str], out_dir: Path) -> None:
    """Save a CSV of key metrics per SAE."""
    import csv

    metric_extractors = [
        ("ce_loss_score", lambda d: _deep_get(d.get("core", {}), "eval_result_metrics", "model_performance_preservation", "ce_loss_score")),
        ("kl_div_score", lambda d: _deep_get(d.get("core", {}), "eval_result_metrics", "model_behavior_preservation", "kl_div_score")),
        ("explained_variance", lambda d: _deep_get(d.get("core", {}), "eval_result_metrics", "reconstruction_quality", "explained_variance")),
        ("l0", lambda d: _deep_get(d.get("core", {}), "eval_result_metrics", "sparsity", "l0")),
        ("absorption", lambda d: _deep_get(d.get("absorption", {}), "eval_result_metrics", "mean", "mean_absorption_fraction_score")),
    ]

    fieldnames = ["sae_name"] + [m[0] for m in metric_extractors]
    rows = []
    for name in sae_names:
        if name not in results:
            continue
        row = {"sae_name": name}
        for metric_name, extractor in metric_extractors:
            val = extractor(results[name])
            row[metric_name] = f"{val:.6f}" if val is not None else ""
        rows.append(row)

    if not rows:
        return

    csv_path = out_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> comparison_summary.csv ({len(rows)} SAEs)")


if __name__ == "__main__":
    main()
