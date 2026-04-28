"""Helpers to read per-channel deadness from baseline TopK SAE encoders.

Walks ``model.encoder`` looking for any submodule that has registered a
``_steps_since_active`` (or ``_steps_since_active_<i>``) buffer together with
a ``dead_steps`` integer attribute, and returns the global dead fraction.

Compatible with:
  - TopKSAEEncoder                            (single buffer per encoder)
  - MatryoshkaBatchTopKSAEEncoder             (single buffer per encoder)
  - IntermediateTopKSAEEncoder                (one buffer per stage)
  - IntermediateMatryoshkaBatchTopKSAEEncoder (one buffer per stage)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_dead_frac(model: nn.Module) -> float:
    """Fraction of channels whose ``_steps_since_active >= dead_steps``.

    Returns 0.0 if no such buffers exist (e.g. before any training step).
    """
    n_dead = 0
    n_total = 0
    for module in model.modules():
        dead_steps = getattr(module, "dead_steps", None)
        if dead_steps is None:
            continue
        for name, buf in module.named_buffers(recurse=False):
            if not name.startswith("_steps_since_active"):
                continue
            with torch.no_grad():
                n_dead += int((buf >= int(dead_steps)).sum().item())
                n_total += int(buf.numel())
    if n_total == 0:
        return 0.0
    return n_dead / n_total


__all__ = ["compute_dead_frac"]
