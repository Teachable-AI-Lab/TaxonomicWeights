#!/usr/bin/env python3
"""Analyze a trained JumpReLUSparseConvAutoencoder checkpoint.

Thin wrapper around analyze_sae.py — reuses all analysis functions unchanged.
The only difference is that ``load_model()`` in analyze_sae.py dispatches to
JumpReLUSparseConvAutoencoder when it reads ``args["model_variant"] == "jumprelu"``
from the checkpoint.

Usage:
    # CelebA-HQ
    python src/analyze/analyze_jumprelu_sae.py --config configs/jumprelu_sae_celeba_hq.json

    # CIFAR-10
    python src/analyze/analyze_jumprelu_sae.py --config configs/jumprelu_sae_cifar10.json

    # Custom checkpoint
    python src/analyze/analyze_jumprelu_sae.py \\
        --config     configs/jumprelu_sae_cifar10.json \\
        --checkpoint ./outputs/sae_jumprelu_cifar10_r18_l064_sw1e-04/checkpoints/best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-use the full analysis pipeline from analyze_sae.
# JumpReLU checkpoints are identified automatically via args["model_variant"] == "jumprelu".
from src.analyze.analyze_sae import main  # noqa: F401

if __name__ == "__main__":
    main()
