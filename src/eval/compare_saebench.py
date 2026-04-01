#!/usr/bin/env python3
"""Compare SAEBench evaluation results across SAE variants with graphics.

Auto-discovers all trained SAEs in a directory, collects their per-model
eval results, optionally evaluates baselines, and generates comparison
graphics.

Usage:
    # Pythia comparison (auto-discover models + baselines):
    python src/eval/compare_saebench.py \
        --models-dir outputs/saebench/pythia160m_layer8 \
        --model-name pythia-160m-deduped \
        --include-baselines --baseline-width 4k

    # Generate graphics only (skip baseline evaluation):
    python src/eval/compare_saebench.py \
        --models-dir outputs/saebench/pythia160m_layer8 \
        --model-name pythia-160m-deduped \
        --include-baselines --baseline-width 4k \
        --skip-eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    MODEL_CONFIGS,
)

RANDOM_SEED = 42

# ── Colour palette ───────────────────────────────────────────────────────────

_CUSTOM_COLOURS = {
    "taxon": "#1f77b4",
    "multi_taxon": "#9467bd",
    "topk_taxon": "#2ca02c",
    "topk_multi_taxon": "#d62728",
}
_BASELINE_COLOURS = [
    "#ff7f0e", "#2ca02c", "#d62728", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _sae_colour(name: str, baseline_idx: int = 0) -> str:
    nl = name.lower()
    if "topk_multi_taxon" in nl:
        return _CUSTOM_COLOURS["topk_multi_taxon"]
    if "topk_taxon" in nl:
        return _CUSTOM_COLOURS["topk_taxon"]
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


def discover_models(models_dir: str) -> list[tuple[str, str, str]]:
    """Scan *models_dir* for subdirectories containing ``checkpoints/best.pt``.

    Returns list of ``(checkpoint_path, variant, sae_name)``.
    """
    base = Path(models_dir)
    found = []
    for ckpt in sorted(base.glob("*/checkpoints/best.pt")):
        training_dir = ckpt.parent.parent
        dirname = training_dir.name

        if "topk_multi_taxon_sae" in dirname:
            variant = "topk_multi_taxon"
        elif "topk_taxon_sae" in dirname:
            variant = "topk_taxon"
        elif "multi_taxon_sae" in dirname:
            variant = "multi_taxon"
        elif "taxon_sae" in dirname:
            variant = "taxon"
        else:
            print(f"  Skipping {dirname}: unrecognised variant")
            continue

        sae_name = dirname  # e.g. "taxon_sae_L10_dkl1e-02_t0p01"
        found.append((str(ckpt), variant, sae_name))

    return found


def collect_results(
    custom_entries: list[tuple[Path, str]],
    baseline_dir: Path | None = None,
    baseline_names: list[str] | None = None,
) -> dict[str, dict]:
    """Collect eval results from per-model ``eval_results/`` directories.

    For custom models every JSON in their ``eval_results/`` belongs to them.
    For baselines the shared *baseline_dir* is scanned and JSONs are matched
    by ``sae_lens_id``.

    Returns: ``{sae_name: {eval_type: parsed_json_dict}}``
    """
    results: dict[str, dict] = {}

    # Custom models — each has its own eval_results/ subdirectory
    for model_dir, sae_name in custom_entries:
        eval_dir = model_dir / "eval_results"
        if not eval_dir.exists():
            print(f"  No eval results for {sae_name} (expected at {eval_dir})")
            continue
        for json_path in sorted(eval_dir.rglob("*.json")):
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            eval_type = data.get("eval_type_id", "")
            if not eval_type:
                continue
            results.setdefault(sae_name, {})[eval_type] = data

    # Baselines — shared directory, match by sae_lens_id / stem
    if baseline_dir and baseline_names and baseline_dir.exists():
        for json_path in sorted(baseline_dir.rglob("*.json")):
            try:
                with open(json_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
            eval_type = data.get("eval_type_id", "")
            sae_id = data.get("sae_lens_id", json_path.stem)
            matched_name = None
            for name in baseline_names:
                if name == sae_id or name in sae_id or sae_id in name:
                    matched_name = name
                    break
            if matched_name is None:
                for name in baseline_names:
                    if json_path.stem == name or name in json_path.stem:
                        matched_name = name
                        break
            if matched_name is None:
                continue
            results.setdefault(matched_name, {})[eval_type] = data

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
    for prefix in ("topk_multi_taxon_sae_", "topk_taxon_sae_", "multi_taxon_sae_", "taxon_sae_", "baseline_"):
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    # Capitalize type prefix
    nl = name.lower()
    if "topk_multi_taxon" in nl:
        return f"TopKMultiTaxon ({n})" if n != name else "TopKMultiTaxonSAE"
    elif "topk_taxon" in nl:
        return f"TopKTaxon ({n})" if n != name else "TopKTaxonSAE"
    elif "multi_taxon" in nl:
        return f"MultiTaxon ({n})" if n != name else "MultiTaxonSAE"
    elif "taxon" in nl:
        return f"Taxon ({n})" if n != name else "TaxonSAE"
    elif name.lower().startswith("baseline_"):
        # Descriptive baseline names like baseline_BatchTopK_k20 or baseline_Standard_l1_0.04
        m = re.match(r'^(.+?)_k(\d+)$', n)
        if m:
            return f"{m.group(1)} (k={m.group(2)})"
        m = re.match(r'^(.+?)_l1_([\d.eE+-]+)$', n)
        if m:
            return f"{m.group(1)} (L1={m.group(2)})"
        return n
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
        description="Compare SAEBench evaluations across all trained SAE variants"
    )
    parser.add_argument("--models-dir", required=True,
                        help="Directory to scan for trained models "
                             "(e.g., outputs/saebench/pythia160m_layer8)")
    parser.add_argument("--model-name", required=True,
                        help="TransformerLens model name "
                             "(e.g., pythia-160m-deduped, gemma-2-2b)")
    parser.add_argument("--include-baselines", action="store_true",
                        help="Include SAEBench baseline SAEs for comparison")
    parser.add_argument("--baseline-width", type=str, default="4k",
                        choices=["4k", "16k"])
    parser.add_argument("--max-baselines", type=int, default=6)
    parser.add_argument("--eval-types", nargs="+",
                        default=["core", "sparse_probing", "scr", "tpp", "absorption"])
    parser.add_argument("--output-dir", type=str, default="",
                        help="Where to save comparison graphics "
                             "(default: {models-dir}/comparison)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip baseline evaluation; only generate graphics "
                             "from existing results")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {args.model_name}")

    mc = MODEL_CONFIGS[args.model_name]
    llm_batch_size = mc["batch_size"]
    llm_dtype = mc["dtype"]
    dtype = general_utils.str_to_dtype(llm_dtype)

    device = general_utils.setup_environment()

    # ── Discover all trained models ──────────────────────────────────────
    print(f"\nScanning {args.models_dir} for trained models ...")
    discovered = discover_models(args.models_dir)
    if not discovered:
        print("No trained models found. Expected subdirectories with "
              "checkpoints/best.pt")
        return

    print(f"  Found {len(discovered)} model(s):")
    for ckpt, variant, sae_name in discovered:
        print(f"    {sae_name} ({variant}) -> {ckpt}")

    # Build custom-model entries for result collection
    sae_names: list[str] = []
    custom_entries: list[tuple[Path, str]] = []
    for ckpt, variant, sae_name in discovered:
        model_dir = Path(ckpt).parent.parent
        custom_entries.append((model_dir, sae_name))
        sae_names.append(sae_name)

    # ── Handle baselines ─────────────────────────────────────────────────
    baseline_dir: Path | None = None
    baseline_names: list[str] = []
    if args.include_baselines:
        # Determine hook_layer from first discovered model
        print(f"\nLoading first model to determine hook_layer ...")
        first_ckpt, first_variant, _ = discovered[0]
        first_sae = load_custom_sae(
            first_ckpt, first_variant, torch.device(device), dtype,
        )
        hook_layer = first_sae.cfg.hook_layer
        del first_sae
        torch.cuda.empty_cache()

        baselines = load_baselines(
            args.model_name, hook_layer, torch.device(device), dtype,
            width=args.baseline_width, max_baselines=args.max_baselines,
        )

        if baselines:
            baseline_dir = (
                Path(args.models_dir) / f"baselines_{args.baseline_width}"
            )
            baseline_output = str(baseline_dir / "eval_results")

            for bname, bsae in baselines:
                bsae = bsae.to(dtype=dtype)
                bsae.cfg.dtype = llm_dtype
            baseline_names = [n for n, _ in baselines]
            sae_names.extend(baseline_names)

            if not args.skip_eval:
                print(f"\nEvaluating {len(baselines)} baselines ...")
                run_evals(
                    model_name=args.model_name,
                    selected_saes=baselines,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    device=device,
                    eval_types=args.eval_types,
                    force_rerun=args.force_rerun,
                    save_activations=args.save_activations,
                    output_base=baseline_output,
                )

            # Free baseline SAEs
            del baselines
            torch.cuda.empty_cache()

    print(f"\nSAEs to compare: {sae_names}")

    # ── Collect results ──────────────────────────────────────────────────
    print(f"\nCollecting results ...")
    bl_eval_dir = baseline_dir / "eval_results" if baseline_dir else None
    results = collect_results(
        custom_entries=custom_entries,
        baseline_dir=bl_eval_dir,
        baseline_names=baseline_names,
    )

    n_with_results = sum(1 for n in sae_names if n in results and results[n])
    print(f"  Found results for {n_with_results}/{len(sae_names)} SAEs")

    if n_with_results == 0:
        print("No results found. Run eval scripts first.")
        return

    # ── Generate graphics ────────────────────────────────────────────────
    output_dir = args.output_dir or str(
        Path(args.models_dir) / "comparison"
    )
    gfx_dir = Path(output_dir) / "graphics"
    model_short = args.model_name.replace("-deduped", "").replace("-", "_")
    title_prefix = f"{model_short} — "
    print(f"\nGenerating comparison graphics in {gfx_dir} ...")
    generate_graphics(results, sae_names, gfx_dir, title_prefix)

    # Save a summary CSV
    _save_summary_csv(results, sae_names, Path(output_dir))

    print(f"\nDone. All outputs in {output_dir}/")


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
