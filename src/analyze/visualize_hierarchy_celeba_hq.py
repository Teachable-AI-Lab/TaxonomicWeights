#!/usr/bin/env python3
"""Per-taxon-run hierarchy visualisations for CelebA-HQ.

For every taxon model run found under ``--outputs-dir`` (using the same
``discover_runs`` machinery as :mod:`probe_celeba_hq`), this script produces
two kinds of GradCAM tree figures saved under
``<save-dir>/taxon_hierarchy/<short_name>/``:

  * ``single_img{i}_<stage_label>_full.png``  — one fixed image at every node,
    overlaid with that node's GradCAM.  ``--n-fixed`` such figures per
    hierarchy.
  * ``topk_<stage_label>_full.png``           — a 2x2 grid of the top-K
    activating validation images per node, each overlaid with its own
    GradCAM.

All raw activations / top-K indices / image tensors / gradcam heatmaps are
also written under ``<save-dir>/taxon_hierarchy/<short_name>/cache/`` as
.npy files so subsequent runs (without ``--force-recompute``) reload from
cache and only re-render the figures.

Usage::

    python src/analyze/visualize_hierarchy_celeba_hq.py
    python src/analyze/visualize_hierarchy_celeba_hq.py --model-filter bottleneck
    python src/analyze/visualize_hierarchy_celeba_hq.py --force-recompute
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyze._taxon_hierarchy_viz_lib import process_one_run  # noqa: E402
from src.compare.compare_celeba_hq import discover_runs, load_model  # noqa: E402
from src.utils.dataloader import CelebAHQLoader  # noqa: E402


_TAXON_TYPES = {
    "taxon", "multi_taxon",
    "topk_taxon", "topk_multi_taxon",
    "bias_taxon", "bias_multi_taxon",
    "bottleneck_taxon", "bottleneck_multi_taxon",
    "bottleneck_topk_taxon", "bottleneck_topk_multi_taxon",
}


def _load_val_loader(data_root: str, image_size: int, batch_size: int, num_workers: int):
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    loader = CelebAHQLoader(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        val_split=0.05,
        transform=tf,
    )
    _, val_loader = loader.get_loaders()
    return val_loader if val_loader is not None else loader.train_loader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", type=str, default="./outputs/celeba_hq",
                   help="Root outputs directory to scan for trained models")
    p.add_argument("--data-root", type=str, default="./data/celeba_hq")
    p.add_argument("--save-dir", type=str, default="./outputs/analysis_celeba_hq",
                   help="Hierarchy figures are written under <save-dir>/taxon_hierarchy/")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=1000,
                   help="Number of val images to scan for top-K and fixed selection")
    p.add_argument("--n-fixed", type=int, default=4,
                   help="Number of fixed images to overlay across the tree (single-image figures)")
    p.add_argument("--top-k", type=int, default=4,
                   help="Top-K activating images shown per node")
    p.add_argument("--ablations", type=str, default=None, metavar="DIR")
    p.add_argument("--model-filter", type=str, default="",
                   help="Only process runs whose name contains this substring")
    p.add_argument("--force-recompute", action="store_true",
                   help="Ignore cached .npy files and re-extract everything")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose taxon_hierarchy/<short>/ directory already has figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs_dir = Path(args.outputs_dir)
    save_dir = Path(args.save_dir)
    out_root = save_dir / "taxon_hierarchy"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CelebA-HQ Hierarchy Visualisation")
    print("=" * 80)
    print(f"  device      = {device}")
    print(f"  outputs_dir = {outputs_dir}")
    print(f"  save_dir    = {out_root}")
    print(f"  max_samples = {args.max_samples}  (top-K, fixed selection)")
    print(f"  n_fixed     = {args.n_fixed}    top_k = {args.top_k}")
    print("=" * 80)

    runs = discover_runs(outputs_dir, include_ablations=False)
    if args.ablations:
        abl_dir = Path(args.ablations)
        if abl_dir.exists():
            abl_runs = discover_runs(abl_dir, include_ablations=True)
            existing = {r["path"] for r in runs}
            runs.extend(r for r in abl_runs if r["path"] not in existing)
        else:
            print(f"  WARNING: --ablations directory not found: {abl_dir}")

    runs = [r for r in runs if r["type"] in _TAXON_TYPES]
    if args.model_filter:
        runs = [r for r in runs if args.model_filter.lower() in r["name"].lower()]

    print(f"\nFound {len(runs)} taxon runs to visualise:")
    for r in runs:
        print(f"  [{r['type']:32s}] {r['name']}")
    if not runs:
        print("Nothing to do.")
        return

    print("\n[Step 1] Building CelebA-HQ val loader …")
    val_loader = _load_val_loader(
        args.data_root, args.image_size, args.batch_size, args.num_workers,
    )

    failures = []
    for i, run in enumerate(runs):
        print(f"\n[{i + 1}/{len(runs)}] {run['name']}  ({run['type']})")
        out_dir = out_root / run["short"].replace("/", "_").replace(" ", "_")
        if args.skip_existing and out_dir.exists() and any(out_dir.glob("*.png")):
            print(f"  SKIP: {out_dir} already has figures")
            continue
        try:
            model, _ = load_model(run, device)
            process_one_run(
                model=model,
                val_loader=val_loader,
                device=device,
                out_dir=out_dir,
                n_fixed=args.n_fixed,
                top_k=args.top_k,
                max_samples=args.max_samples,
                force_recompute=args.force_recompute,
                short_label=run["short"],
            )
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failures.append(run["name"])
        finally:
            torch.cuda.empty_cache()

    print(f"\nDone. {len(runs) - len(failures)} succeeded, {len(failures)} failed.")
    if failures:
        for n in failures:
            print(f"  FAIL: {n}")
        sys.exit(1)


if __name__ == "__main__":
    main()
