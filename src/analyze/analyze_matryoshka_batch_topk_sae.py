#!/usr/bin/env python3
"""Analyze a trained MatryoshkaBatchTopKSparseConvAutoencoder checkpoint.

Thin wrapper around analyze_sae.py — reuses all analysis functions unchanged.
The model is identified automatically via ``args["model_variant"] ==
"matryoshka_batch_topk"`` in the checkpoint, which dispatches to
:class:`~src.model.cnn.baseline.matryoshka_batch_topk_sae.MatryoshkaBatchTopKSparseConvAutoencoder`
in :func:`~src.analyze.analyze_sae.load_model`.

Analysis uses the largest-k latent (primary forward pass) for all metrics,
matching the inference-time behaviour of the model.

Usage:
    # CelebA-HQ
    python src/analyze/analyze_matryoshka_batch_topk_sae.py \\
        --config configs/celeba_hq/matryoshka_batch_topk_sae_celeba_hq.json

    # ImageNet
    python src/analyze/analyze_matryoshka_batch_topk_sae.py \\
        --config configs/imagenet/matryoshka_batch_topk_sae_imagenet.json

    # Custom checkpoint
    python src/analyze/analyze_matryoshka_batch_topk_sae.py \\
        --config     configs/celeba_hq/matryoshka_batch_topk_sae_celeba_hq.json \\
        --checkpoint ./outputs/celeba_hq/sae_matryoshka_batch_topk_celeba_hq_r18_k8-4-2_sw1e-02/checkpoints/best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-use the full analysis pipeline from analyze_sae.
# Matryoshka Batch-TopK checkpoints are identified automatically via
# args["model_variant"] == "matryoshka_batch_topk".
from src.analyze.analyze_sae import main  # noqa: F401

if __name__ == "__main__":
    main()
