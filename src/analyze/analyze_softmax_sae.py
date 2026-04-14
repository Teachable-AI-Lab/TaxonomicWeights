#!/usr/bin/env python3
"""Analyze a trained SoftmaxSparseConvAutoencoder checkpoint.

Thin wrapper around :func:`src.analyze.analyze_sae.main` — all analysis
logic is shared with the other SAE variants.

Usage::

    python src/analyze/analyze_softmax_sae.py \\
        --config configs/celeba_hq/softmax_sae_celeba_hq.json

    python src/analyze/analyze_softmax_sae.py \\
        --config configs/imagenet/softmax_sae_imagenet.json
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analyze.analyze_sae import main

if __name__ == "__main__":
    main()
