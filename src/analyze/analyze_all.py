#!/usr/bin/env python3
"""Dispatch the appropriate analysis script for every run directory found under
``outputs/``, skipping comparison directories.

Model-type detection (directory name prefix):
  ``taxon_ae_*``       → src/analyze/analyze_celeba_hq_ae.py
  ``multi_taxon_ae_*`` → src/analyze/analyze_multi_taxon_ae_celeba_hq.py
  ``baseline_ae_*``    → src/analyze/analyze_baseline_ae.py
  ``sae_*``            → src/analyze/analyze_sae.py  (handles l1/topk/gated)

For each matching run directory the script:
  1. Verifies ``checkpoints/best.pt`` exists.
  2. Reads ``args`` from the checkpoint to recover all training hyperparameters.
  3. Builds a synthetic JSON config (or CLI flags) for the target analyze script.
  4. Invokes that script as a subprocess, continuing on failure.

Usage::

    # Analyze everything
    python src/analyze/analyze_all.py

    # Only taxon / multi-taxon runs
    python src/analyze/analyze_all.py --taxon-only

    # Skip runs that already have an analysis/ sub-directory
    python src/analyze/analyze_all.py --skip-existing

    # Preview without executing
    python src/analyze/analyze_all.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── model-type detection ─────────────────────────────────────────────────────

def detect_model_type(run_name: str) -> Optional[str]:
    """Return one of 'taxon', 'multi_taxon', 'topk_taxon', 'topk_multi_taxon',
    'bias_taxon', 'bias_multi_taxon', 'baseline', 'sae', or None."""
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
    if run_name.startswith("baseline_ae_"):
        return "baseline"
    if (
        run_name.startswith("sae_")
        or run_name.startswith("sae_topk_")
        or run_name.startswith("sae_gated_")
    ):
        return "sae"
    return None


# ── data-param inference ─────────────────────────────────────────────────────

def _infer_data_params(run_name: str, a: dict) -> dict:
    """Return a data-config dict, preferring checkpoint args, falling back to
    directory-name heuristics."""
    if "cifar10" in run_name.lower():
        default_root, default_size = "./data", 32
    elif "imagenet" in run_name.lower():
        default_root, default_size = "./data/imagenet", 224
    else:
        default_root, default_size = "./data/celeba_hq", 256

    return {
        "data_root":   a.get("data_root",  default_root),
        "image_size":  a.get("image_size", default_size),
        "batch_size":  a.get("batch_size", 16),
        "num_workers": a.get("num_workers", 4),
        "val_split":   a.get("val_split", 0.05),
    }


# ── config builders ──────────────────────────────────────────────────────────

def _build_taxon_config(a: dict, run_dir: Path) -> dict:
    """Synthetic config for analyze_celeba_hq_ae.py.

    Sets ``output_dir`` to the *base* path (without run suffix) so that the
    analyze script can reconstruct the correct run suffix from the hyperparams
    and locate both the checkpoint and the analysis save directory.
    """
    base_out = a.get("output_dir", str(run_dir))
    data = _infer_data_params(run_dir.name, a)
    return {
        # experiment_name prevents the analyze script from appending a datetime
        # suffix to the analysis directory name.
        "experiment_name": run_dir.name,
        "model": {
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
        },
        "training": {
            "dkl_weight":     a.get("dkl_weight", 1e-2),
            "entropy_weight": a.get("entropy_weight", 0.0),
            "seed":           a.get("seed", 42),
        },
        "data": data,
        "output": {
            # Use the base output dir (without run suffix); the analyze script
            # appends run_suffix itself based on the hyperparams above.
            "output_dir":        base_out,
            "analysis_save_dir": base_out + "/analysis",
        },
        "analysis": {
            # Base checkpoint path — the analyze script also appends run_suffix
            # before looking for the file.
            "checkpoint_path": base_out + "/checkpoints/best.pt",
        },
    }


def _build_minimal_config(run_name: str, a: dict) -> dict:
    """Config that only supplies data loading parameters.

    Used for baseline and SAE analyze scripts when ``--checkpoint`` and
    ``--save-dir`` are passed explicitly on the command line.
    """
    data = _infer_data_params(run_name, a)
    return {
        "data":   data,
        "model":  {},
        "output": {"output_dir": "", "analysis_save_dir": ""},
    }


# ── per-run dispatch ─────────────────────────────────────────────────────────

def _write_temp_config(cfg: dict, prefix: str) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix=prefix
    ) as f:
        json.dump(cfg, f, indent=2)
        return f.name


def _cleanup_temp(cmd: List[str]) -> None:
    for arg in cmd:
        if arg.endswith(".json") and os.path.dirname(arg) == tempfile.gettempdir():
            try:
                os.unlink(arg)
            except OSError:
                pass


def run_one(run_dir: Path, model_type: str, dry_run: bool = False,
            skip_partonomy: bool = False) -> bool:
    """Build the subprocess command for one run directory and optionally execute it.

    Returns True on success or dry-run, False on checkpoint-not-found or
    non-zero exit code.
    """
    ckpt_path = run_dir / "checkpoints" / "best.pt"

    print(f"\n{'=' * 72}")
    print(f"  Run dir   : {run_dir.name}")
    print(f"  Model type: {model_type}")
    print(f"  Checkpoint: {ckpt_path}")

    if not ckpt_path.exists():
        print(f"  SKIP: checkpoint not found.")
        return False

    a = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("args", {})

    if model_type == "taxon":
        cfg = _build_taxon_config(a, run_dir)
        cfg_path = _write_temp_config(cfg, "analyze_all_taxon_")
        cmd = [
            sys.executable,
            str(ROOT / "src" / "analyze" / "analyze_celeba_hq_ae.py"),
            "--config",     cfg_path,
            "--checkpoint", str(ckpt_path),
        ]

    elif model_type == "multi_taxon":
        base_out = a.get("output_dir", str(run_dir))
        data = _infer_data_params(run_dir.name, a)
        cmd = [
            sys.executable,
            str(ROOT / "src" / "analyze" / "analyze_multi_taxon_ae_celeba_hq.py"),
            "--output-dir",          base_out,
            "--dkl-weight",          str(a.get("dkl_weight", 1e-2)),
            "--temperature",         str(a.get("temperature", 1.0)),
            "--n-hierarchies",       str(a.get("n_hierarchies", 3)),
            "--entropy-weight",      str(a.get("entropy_weight", 0.0)),
            "--gate-dkl-weight",     str(a.get("gate_dkl_weight", 1e-2)),
            "--gate-entropy-weight", str(a.get("gate_entropy_weight", 0.0)),
            "--data-root",           data["data_root"],
            "--image-size",          str(data["image_size"]),
            "--batch-size",          str(data["batch_size"]),
            "--num-workers",         str(data["num_workers"]),
            "--val-split",           str(data["val_split"]),
            "--seed",                str(a.get("seed", 42)),
        ]
        if a.get("hard", False):
            cmd.append("--hard")

    elif model_type == "baseline":
        cfg = _build_minimal_config(run_dir.name, a)
        cfg_path = _write_temp_config(cfg, "analyze_all_baseline_")
        cmd = [
            sys.executable,
            str(ROOT / "src" / "analyze" / "analyze_baseline_ae.py"),
            "--config",     cfg_path,
            "--checkpoint", str(ckpt_path),
            "--save-dir",   str(run_dir / "analysis"),
        ]

    elif model_type == "sae":
        cfg = _build_minimal_config(run_dir.name, a)
        cfg_path = _write_temp_config(cfg, "analyze_all_sae_")
        cmd = [
            sys.executable,
            str(ROOT / "src" / "analyze" / "analyze_sae.py"),
            "--config",     cfg_path,
            "--checkpoint", str(ckpt_path),
            "--save-dir",   str(run_dir / "analysis"),
        ]

    elif model_type == "topk_taxon":
        cfg = _build_taxon_config(a, run_dir)
        cfg["model"]["k"] = a.get("k", None)
        cfg["model"]["k_aux"] = a.get("k_aux", None)
        cfg["model"]["dead_steps"] = a.get("dead_steps", 2000)
        cfg["model"]["model_variant"] = "topk"
        cfg["training"]["auxk_weight"] = a.get("auxk_weight", 0.0)
        # Let the new script compute analysis_save_dir from output_dir + run_suffix
        cfg["output"]["analysis_save_dir"] = ""
        cfg_path = _write_temp_config(cfg, "analyze_all_topk_taxon_")
        cmd = [
            sys.executable,
            str(ROOT / "src" / "analyze" / "analyze_topk_taxon_ae_celeba_hq.py"),
            "--config",     cfg_path,
            "--checkpoint", str(ckpt_path),
            "--output-dir", cfg["output"]["output_dir"],
            "--k", str(a.get("k", "")),
            "--auxk-weight", str(a.get("auxk_weight", 0.0)),
        ]
        # Drop --k if it was None
        if a.get("k") is None:
            cmd = [c for i, c in enumerate(cmd) if not (cmd[max(0,i-1)] == "--k" or c == "--k")]

    elif model_type == "bias_taxon":
        cfg = _build_taxon_config(a, run_dir)
        cfg["model"]["k"] = a.get("k", None)
        cfg["model"]["bias_update_rate"] = a.get("bias_update_rate", 0.001)
        cfg["model"]["bias_ema_decay"] = a.get("bias_ema_decay", 0.99)
        cfg["model"]["model_variant"] = "bias"
        # Let the new script compute analysis_save_dir from output_dir + run_suffix
        cfg["output"]["analysis_save_dir"] = ""
        cfg_path = _write_temp_config(cfg, "analyze_all_bias_taxon_")
        cmd = [
            sys.executable,
            str(ROOT / "src" / "analyze" / "analyze_bias_taxon_ae_celeba_hq.py"),
            "--config",     cfg_path,
            "--checkpoint", str(ckpt_path),
            "--output-dir", cfg["output"]["output_dir"],
            "--bias-update-rate", str(a.get("bias_update_rate", 0.001)),
        ]
        if a.get("k") is not None:
            cmd.extend(["--k", str(a["k"])])

    elif model_type in ("topk_multi_taxon", "bias_multi_taxon"):
        base_out = a.get("output_dir", str(run_dir))
        data = _infer_data_params(run_dir.name, a)
        model_cfg: dict = {
            "in_channels":           a.get("in_channels", 3),
            "resnet_variant":        a.get("resnet_variant", "18"),
            "stage_taxonomy_layers": a.get("stage_taxonomy_layers", [5, 6, 7, 8]),
            "stage_strides":         a.get("stage_strides", [1, 2, 2, 2]),
            "stage_blocks":          a.get("stage_blocks", None),
            "n_hierarchies":         a.get("n_hierarchies", 3),
            "kernel_size":           a.get("kernel_size", 3),
            "use_stem":              a.get("use_stem", True),
            "stem_channels":         a.get("stem_channels", 64),
            "stem_stride":           a.get("stem_stride", 2),
            "use_stem_maxpool":      a.get("use_stem_maxpool", True),
            "output_activation":     a.get("output_activation", "none"),
            "depth_decay":           a.get("depth_decay", 0.5),
        }
        if model_type == "topk_multi_taxon":
            model_cfg["model_variant"] = "topk"
            model_cfg["k"] = a.get("k", None)
            model_cfg["k_aux"] = a.get("k_aux", None)
            model_cfg["dead_steps"] = a.get("dead_steps", 2000)
            model_cfg["gate_k"] = a.get("gate_k", 1)
            script = "analyze_topk_multi_taxon_ae_celeba_hq.py"
        else:
            model_cfg["model_variant"] = "bias"
            model_cfg["k"] = a.get("k", None)
            model_cfg["bias_update_rate"] = a.get("bias_update_rate", 0.001)
            model_cfg["bias_ema_decay"] = a.get("bias_ema_decay", 0.99)
            model_cfg["gate_k"] = a.get("gate_k", 1)
            script = "analyze_bias_multi_taxon_ae_celeba_hq.py"
        cfg = {
            "model": model_cfg,
            "data": data,
            "training": {
                "seed": a.get("seed", 42),
                "auxk_weight": a.get("auxk_weight", 0.0),
            },
            "output": {
                "output_dir": base_out,
                "analysis_save_dir": "",
            },
        }
        cfg_path = _write_temp_config(cfg, f"analyze_all_{model_type}_")
        cmd = [
            sys.executable,
            str(ROOT / "src" / "analyze" / script),
            "--config",              cfg_path,
            "--output-dir",          base_out,
            "--n-hierarchies",       str(a.get("n_hierarchies", 3)),
            "--data-root",           data["data_root"],
            "--image-size",          str(data["image_size"]),
            "--batch-size",          str(data["batch_size"]),
            "--num-workers",         str(data["num_workers"]),
            "--val-split",           str(data["val_split"]),
            "--seed",                str(a.get("seed", 42)),
        ]
        if model_type == "topk_multi_taxon":
            if a.get("k") is not None:
                cmd.extend(["--k", str(a["k"])])
            cmd.extend(["--auxk-weight", str(a.get("auxk_weight", 0.0))])
        else:
            if a.get("k") is not None:
                cmd.extend(["--k", str(a["k"])])
            cmd.extend(["--bias-update-rate", str(a.get("bias_update_rate", 0.001))])

    else:
        print(f"  SKIP: unrecognised model type '{model_type}'.")
        return False

    if skip_partonomy:
        cmd.append("--skip-partonomy")

    print(f"  Command: {' '.join(cmd)}")
    if dry_run:
        print("  DRY RUN — not executing.")
        _cleanup_temp(cmd)
        return True

    result = subprocess.run(cmd, cwd=str(ROOT))
    _cleanup_temp(cmd)
    return result.returncode == 0


# ── entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run analysis for every output directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--outputs-dir", type=str, default="outputs",
        help="Root outputs directory to scan (default: outputs/)",
    )
    p.add_argument(
        "--taxon-only", action="store_true",
        help="Only process taxon_ae_* and multi_taxon_ae_* directories.",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip run directories that already contain an analysis/ sub-directory.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    p.add_argument(
        "--skip-partonomy", action="store_true",
        help="Pass --skip-partonomy to every taxon analyze script.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = ROOT / args.outputs_dir

    if not outputs_dir.exists():
        print(f"ERROR: outputs directory not found: {outputs_dir}")
        sys.exit(1)

    run_dirs = sorted(
        d for d in outputs_dir.iterdir()
        if d.is_dir() and not d.name.startswith("comparison")
    )

    # If a subdirectory is a dataset grouping folder (e.g. celeba_hq/, cifar10/)
    # rather than a model run, expand it to its children.
    expanded: List[Path] = []
    for d in run_dirs:
        if detect_model_type(d.name) is None:
            # Treat as dataset-level folder — scan its children
            expanded.extend(
                sorted(
                    c for c in d.iterdir()
                    if c.is_dir() and not c.name.startswith("comparison")
                )
            )
        else:
            expanded.append(d)
    run_dirs = expanded

    successes: List[str] = []
    failures:  List[str] = []
    skipped:   List[str] = []

    for run_dir in run_dirs:
        model_type = detect_model_type(run_dir.name)
        if model_type is None:
            print(f"\nSkipping unrecognised directory: {run_dir.name}")
            skipped.append(run_dir.name)
            continue

        if args.taxon_only and model_type not in (
            "taxon", "multi_taxon",
            "topk_taxon", "topk_multi_taxon",
            "bias_taxon", "bias_multi_taxon",
        ):
            skipped.append(run_dir.name)
            continue

        if args.skip_existing and (run_dir / "analysis").exists():
            print(f"\nSkipping (analysis/ already exists): {run_dir.name}")
            skipped.append(run_dir.name)
            continue

        ok = run_one(run_dir, model_type, dry_run=args.dry_run,
                     skip_partonomy=args.skip_partonomy)
        (successes if ok else failures).append(run_dir.name)

    print(f"\n{'=' * 72}")
    print(
        f"Summary: {len(successes)} succeeded, {len(failures)} failed, "
        f"{len(skipped)} skipped"
    )
    if failures:
        print("Failed runs:")
        for name in failures:
            print(f"  {name}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
