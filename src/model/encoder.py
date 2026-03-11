"""Taxonomy-constrained ResNet-style encoder modules.

This module contains the encoder-side building blocks used by the taxon
autoencoder:
- A pre-activation residual block (ResNet basic-block style).
- A taxonomic stage that converts stage activations into hierarchical path
  distributions through pairwise softmax.
- A ResNet-like multi-stage encoder that applies the taxonomic stage at each
  residual stage.

Regularization terms produced by each taxonomic stage:
- ``entropy``: mean per-spatial entropy of path probabilities (lower => spikier).
- ``dkl``: KL(path-marginal || uniform), where the path marginal is aggregated
  across batch and spatial locations (coverage/diversity term).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def resolve_resnet_stage_blocks(
    resnet_variant: str | int | None = "18",
    stage_blocks: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Resolve stage block counts from a ResNet variant.

    Args:
        resnet_variant: ResNet variant identifier (e.g., ``"18"``, ``"34"``).
        stage_blocks: Explicit block counts. If provided, it is returned directly.

    Returns:
        Tuple of block counts per stage.
    """
    if stage_blocks is not None:
        return tuple(int(v) for v in stage_blocks)

    presets = {
        "18": (2, 2, 2, 2),
        "34": (3, 4, 6, 3),
        "50": (3, 4, 6, 3),
        "101": (3, 4, 23, 3),
    }

    key = "18" if resnet_variant is None else str(resnet_variant).lower().replace("resnet", "").strip()
    if key not in presets:
        raise ValueError(f"Unsupported resnet_variant={resnet_variant}. Supported: {sorted(presets)}")
    return presets[key]


class ResidualConvBlock(nn.Module):
    """Pre-activation residual block used inside taxonomic encoder stages.

    Structure: ``BN -> Conv -> ReLU -> BN -> Conv + skip``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if (in_channels != out_channels) or (stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class TaxonResNetStage(nn.Module):
    """One ResNet stage with taxonomy-constrained hierarchical routing.

    The stage first computes residual features, then interprets them as logits for
    a binary-tree taxonomy per depth, with channel counts ``[2, 4, 8, ...]``.
    """

    def __init__(
        self,
        in_channels: int,
        n_taxonomy_layers: int,
        n_blocks: int,
        stride: int = 1,
        kernel_size: int = 3,
        temperature: float = 1.0,
        hard: bool = False,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()
        if n_taxonomy_layers < 1:
            raise ValueError(f"n_taxonomy_layers must be >= 1, got {n_taxonomy_layers}")

        self.in_channels = in_channels
        self.n_taxonomy_layers = n_taxonomy_layers
        self.n_blocks = n_blocks
        self.stride = stride
        self.temperature = float(temperature)
        self.default_hard = bool(hard)
        self.depth_decay = float(depth_decay)

        self.layer_channels: List[int] = [1 << (i + 1) for i in range(self.n_taxonomy_layers)]
        self.total_out_channels = self.output_channels(self.n_taxonomy_layers)

        blocks: List[nn.Module] = []
        for idx in range(self.n_blocks):
            block_in = self.in_channels if idx == 0 else self.total_out_channels
            block_stride = self.stride if idx == 0 else 1
            blocks.append(
                ResidualConvBlock(
                    in_channels=block_in,
                    out_channels=self.total_out_channels,
                    kernel_size=kernel_size,
                    stride=block_stride,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    @staticmethod
    def output_channels(n_taxonomy_layers: int) -> int:
        """Return total output channels for depths ``[2, 4, ..., 2^L]``."""
        return (1 << (n_taxonomy_layers + 1)) - 2

    def _stage_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = x
        for block in self.blocks:
            feat = block(feat)
        return feat

    def _taxon_logits_per_depth(self, x: torch.Tensor) -> List[torch.Tensor]:
        stage_logits = self._stage_features(x)
        if stage_logits.shape[1] != self.total_out_channels:
            raise ValueError(
                f"Stage output channel mismatch: got {stage_logits.shape[1]}, expected {self.total_out_channels}."
            )
        return list(torch.split(stage_logits, self.layer_channels, dim=1))

    def _pairwise_log_softmax(
        self,
        logits: torch.Tensor,
        tau: Optional[float] = None,
        hard: Optional[bool] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Apply pairwise (sibling) softmax and return log-conditionals.

        When ``hard=True``, a straight-through one-hot estimator is used.
        """
        if tau is None:
            tau = self.temperature
        if hard is None:
            hard = self.default_hard

        if logits.shape[1] % 2 != 0:
            raise ValueError(f"Expected even channel count for pairwise softmax, got {logits.shape[1]}")

        bsz, channels, h, w = logits.shape
        pair_logits = logits.view(bsz, channels // 2, 2, h, w)
        y_soft = torch.softmax(pair_logits / tau, dim=2)

        if hard:
            argmax = y_soft.argmax(dim=2, keepdim=True)
            y_hard = torch.zeros_like(pair_logits).scatter_(2, argmax, 1.0)
            probs = y_hard - y_soft.detach() + y_soft
        else:
            probs = y_soft

        return probs.clamp_min(eps).log().view(bsz, channels, h, w)

    def _regularization_terms(
        self,
        prob: torch.Tensor,
        logp: torch.Tensor,
        eps: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute local entropy and aggregated coverage KL for one depth."""
        local_entropy = -(prob * logp).sum(dim=1).mean()

        marginal = prob.mean(dim=(0, 2, 3))
        marginal = marginal / marginal.sum().clamp_min(eps)
        uniform_logp = -math.log(prob.shape[1])
        coverage_kl = (marginal * (marginal.clamp_min(eps).log() - uniform_logp)).sum()

        return local_entropy, coverage_kl

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run stage forward pass.

        Returns:
            - Concatenated taxonomy-gated stage activations.
            - Concatenated log path probabilities.
            - Dict with ``{"entropy", "dkl"}``.
        """
        outputs: List[torch.Tensor] = []
        logps: List[torch.Tensor] = []
        prev: Optional[torch.Tensor] = None

        total_entropy = x.new_zeros(())
        total_dkl = x.new_zeros(())

        for depth_idx, logits in enumerate(self._taxon_logits_per_depth(x)):
            log_cond = self._pairwise_log_softmax(logits, tau=self.temperature, hard=hard)
            logp = log_cond if prev is None else log_cond + prev.repeat_interleave(2, dim=1)
            prob = logp.exp()
            out = logits * prob

            entropy_i, dkl_i = self._regularization_terms(prob, logp)
            depth_weight = self.depth_decay ** depth_idx
            total_entropy = total_entropy + depth_weight * entropy_i
            total_dkl = total_dkl + depth_weight * dkl_i

            outputs.append(out)
            logps.append(logp)
            prev = logp

        return (
            torch.cat(outputs, dim=1),
            torch.cat(logps, dim=1),
            {"entropy": total_entropy, "dkl": total_dkl},
        )

    @torch.no_grad()
    def evaluate_path_encoding(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        atol: float = 1e-5,
        rtol: float = 1e-4,
    ) -> Dict[str, object]:
        """Validate path-mass constraints for this stage on input ``x``."""
        out_all, logp_all, regs = self.forward(x, hard=hard)
        out_splits = torch.split(out_all, self.layer_channels, dim=1)
        logp_splits = torch.split(logp_all, self.layer_channels, dim=1)

        depth_reports: List[Dict[str, object]] = []
        prev_logp: Optional[torch.Tensor] = None
        all_ok = True

        for depth_idx, (out, logp) in enumerate(zip(out_splits, logp_splits)):
            probs = logp.exp()
            mass_err = (probs.sum(dim=1) - 1.0).abs().max().item()
            mass_ok = mass_err <= (atol + rtol)

            if prev_logp is None:
                parent_err = 0.0
                parent_ok = True
            else:
                pair_mass = probs.view(probs.shape[0], -1, 2, probs.shape[2], probs.shape[3]).sum(dim=2)
                parent = prev_logp.exp()
                parent_err = (pair_mass - parent).abs().max().item()
                parent_ok = parent_err <= (atol + rtol)

            finite_ok = bool(torch.isfinite(out).all().item() and torch.isfinite(logp).all().item())
            depth_ok = bool(mass_ok and parent_ok and finite_ok)
            all_ok = all_ok and depth_ok

            depth_reports.append(
                {
                    "depth": depth_idx + 1,
                    "channels": int(out.shape[1]),
                    "mass_error": float(mass_err),
                    "parent_error": float(parent_err),
                    "finite_ok": finite_ok,
                    "ok": depth_ok,
                }
            )
            prev_logp = logp

        return {
            "ok": all_ok,
            "n_depths": self.n_taxonomy_layers,
            "layer_channels": list(self.layer_channels),
            "dkl": float(regs["dkl"].detach().cpu()),
            "entropy": float(regs["entropy"].detach().cpu()),
            "depth_reports": depth_reports,
        }


class TaxonResNetEncoder(nn.Module):
    """ResNet-like encoder whose stage activations are taxonomy-constrained.

    The structure mirrors standard ResNet staging:
    ``stem -> stage1 -> stage2 -> stage3 -> stage4``
    with stage block counts from a ResNet preset (e.g., 18 => ``[2,2,2,2]``).

    Each stage output is not a plain feature map; it is transformed into
    taxonomy-gated activations via :class:`TaxonResNetStage`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_taxonomy_layers: Sequence[int] = (5, 6, 7, 8),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        temperature: float = 1.0,
        hard: bool = False,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        depth_decay: float = 0.5,
    ) -> None:
        super().__init__()

        resolved_stage_blocks = resolve_resnet_stage_blocks(
            resnet_variant=resnet_variant,
            stage_blocks=stage_blocks,
        )

        if not (len(resolved_stage_blocks) == len(stage_taxonomy_layers) == len(stage_strides)):
            raise ValueError(
                "stage_blocks, stage_taxonomy_layers, and stage_strides must have equal length. "
                f"Got {len(resolved_stage_blocks)}, {len(stage_taxonomy_layers)}, {len(stage_strides)}"
            )

        self.in_channels = in_channels
        self.resnet_variant = str(resnet_variant)
        self.stage_blocks = tuple(int(v) for v in resolved_stage_blocks)
        self.stage_taxonomy_layers = tuple(int(v) for v in stage_taxonomy_layers)
        self.stage_strides = tuple(int(v) for v in stage_strides)
        self.temperature = float(temperature)
        self.default_hard = bool(hard)
        self.use_stem = bool(use_stem)
        self.stem_stride = int(stem_stride)
        self.use_stem_maxpool = bool(use_stem_maxpool)

        if self.use_stem:
            stem_ops: List[nn.Module] = [
                nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=self.stem_stride, padding=3, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            ]
            if self.use_stem_maxpool:
                stem_ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.stem = nn.Sequential(*stem_ops)
            current_channels = stem_channels
            self.stem_total_stride = self.stem_stride * (2 if self.use_stem_maxpool else 1)
        else:
            self.stem = nn.Identity()
            current_channels = in_channels
            self.stem_total_stride = 1

        self.stage_input_channels: List[int] = []
        self.taxon_stages = nn.ModuleList()

        for blocks, n_layers, stride in zip(self.stage_blocks, self.stage_taxonomy_layers, self.stage_strides):
            self.stage_input_channels.append(current_channels)
            stage = TaxonResNetStage(
                in_channels=current_channels,
                n_taxonomy_layers=n_layers,
                n_blocks=blocks,
                stride=stride,
                kernel_size=kernel_size,
                temperature=self.temperature,
                hard=self.default_hard,
                depth_decay=depth_decay,
            )
            self.taxon_stages.append(stage)
            current_channels = stage.total_out_channels

        self.final_channels = current_channels

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode input image into taxonomy-aware latent features."""
        if hard is None:
            hard = self.default_hard

        details: Dict[str, object] = {
            "input_shape": tuple(x.shape),
            "shape_trace": [],
            "stages": [],
            "resnet_variant": self.resnet_variant,
            "stage_blocks": self.stage_blocks,
            "stage_taxonomy_layers": self.stage_taxonomy_layers,
            "stage_strides": self.stage_strides,
        }

        x = self.stem(x)
        if return_details:
            details["shape_trace"].append(("stem", tuple(x.shape)))

        total_dkl = x.new_zeros(())
        total_entropy = x.new_zeros(())

        for stage_idx, stage in enumerate(self.taxon_stages, start=1):
            x, stage_logp, regs = stage(x, hard=hard)
            total_dkl = total_dkl + regs["dkl"]
            total_entropy = total_entropy + regs["entropy"]

            if return_details:
                details["shape_trace"].append((f"stage{stage_idx}", tuple(x.shape)))
                details["stages"].append(
                    {
                        "name": f"stage{stage_idx}",
                        "output": x,
                        "logp": stage_logp,
                        "output_shape": tuple(x.shape),
                        "logp_shape": tuple(stage_logp.shape),
                        "layer_channels": list(stage.layer_channels),
                        "taxonomy_depth": stage.n_taxonomy_layers,
                        "n_blocks": stage.n_blocks,
                        "stride": stage.stride,
                        "dkl": regs["dkl"],
                        "entropy": regs["entropy"],
                    }
                )

        details["latent_shape"] = tuple(x.shape)
        details["dkl"] = total_dkl
        details["entropy"] = total_entropy

        return x, details




class TaxonResNetStageWithAttention(TaxonResNetStage):
    """ResNet stage that applies cross-depth attention to taxonomy logits.

    After computing the usual per-depth logits we treat each "pair" of
    channels as a 2‑dimensional token and run a simple multi‑head
    self‑attention sequence across depths.  The idea is to let later depths
    condition their logits on the routing decisions made at earlier depths.

    The attention is very lightweight: by default it uses a single head in an
    embedding space of dimension two (one logit per sibling).  Only a small
    amount of extra computation is incurred, and the existing entropy/DKL
    regularizers still operate on the final, attended probabilities.
    """

    def __init__(
        self,
        *args,
        attn_heads: int = 1,
        **kwargs,
    ) -> None:
        # ``args`` and ``kwargs`` are passed through to the base class
        super().__init__(*args, **kwargs)
        if attn_heads < 1 or attn_heads > 2:
            raise ValueError("attn_heads currently must be 1 or 2 (embed_dim=2)")
        # each token is a pair of logits => embed_dim==2
        self.attn_heads = attn_heads
        self.attn = nn.MultiheadAttention(embed_dim=2, num_heads=attn_heads, batch_first=False)

    def _apply_cross_depth_attention(
        self, logits_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Run sequential attention across depths.

        Args:
            logits_list: list of tensors ``[B, C_d, H, W]`` for each depth.

        Returns: new list of logits with the same shapes.
        """
        # build tokens per depth: shape (B*H*W, pairs, 2)
        b, _, h, w = logits_list[0].shape
        tokens_per_depth: List[torch.Tensor] = []
        for logits in logits_list:
            c = logits.shape[1]
            pairs = c // 2
            # [B, pairs, 2, H, W]
            t = logits.view(b, pairs, 2, h, w)
            # [B, H, W, pairs, 2]
            t = t.permute(0, 3, 4, 1, 2)
            tokens_per_depth.append(t.reshape(b * h * w, pairs, 2))

        prev_kv: Optional[torch.Tensor] = None
        attended_tokens: List[torch.Tensor] = []
        for t in tokens_per_depth:
            if prev_kv is None:
                attended = t
            else:
                # attention expects seq,len-first: (seq_len, batch, embed_dim)
                q = t.permute(1, 0, 2)  # [pairs, N, 2]
                kv = prev_kv.permute(1, 0, 2)  # [K, N, 2]
                att_out, _ = self.attn(q, kv, kv)
                attended = att_out.permute(1, 0, 2)  # [N, pairs, 2]
            attended_tokens.append(attended)
            prev_kv = attended if prev_kv is None else torch.cat([prev_kv, attended], dim=1)

        # convert tokens back into logits
        new_logits_list: List[torch.Tensor] = []
        for idx, attended in enumerate(attended_tokens):
            pairs = attended.shape[1]
            c = pairs * 2
            out = attended.view(b, h, w, pairs, 2).permute(0, 3, 4, 1, 2).reshape(b, c, h, w)
            new_logits_list.append(out)
        return new_logits_list

    def forward(
        self,
        x: torch.Tensor,
        hard: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # replicate parent logic but insert attention step
        outputs: List[torch.Tensor] = []
        logps: List[torch.Tensor] = []
        prev: Optional[torch.Tensor] = None

        total_entropy = x.new_zeros(())
        total_dkl = x.new_zeros(())

        # calculate raw logits per depth then run attention
        logits_list = self._taxon_logits_per_depth(x)
        logits_list = self._apply_cross_depth_attention(logits_list)

        for depth_idx, logits in enumerate(logits_list):
            log_cond = self._pairwise_log_softmax(logits, tau=self.temperature, hard=hard)
            logp = log_cond if prev is None else log_cond + prev.repeat_interleave(2, dim=1)
            prob = logp.exp()
            out = logits * prob

            entropy_i, dkl_i = self._regularization_terms(prob, logp)
            depth_weight = self.depth_decay ** depth_idx
            total_entropy = total_entropy + depth_weight * entropy_i
            total_dkl = total_dkl + depth_weight * dkl_i

            outputs.append(out)
            logps.append(logp)
            prev = logp

        return (
            torch.cat(outputs, dim=1),
            torch.cat(logps, dim=1),
            {"entropy": total_entropy, "dkl": total_dkl},
        )


class TaxonResNetEncoderWithAttention(TaxonResNetEncoder):
    """Encoder using the attention‑enhanced stages.

    Parameters are identical to :class:`TaxonResNetEncoder` plus an
    ``attn_heads`` argument passed through to the underlying stages.
    """

    def __init__(
        self,
        *args,
        attn_heads: int = 1,
        **kwargs,
    ) -> None:
        # copy most of TaxonResNetEncoder.__init__ but create
        # TaxonResNetStageWithAttention instances instead of the vanilla ones.
        in_channels = kwargs.get("in_channels", 3) if "in_channels" in kwargs else args[0] if args else 3
        resnet_variant = kwargs.get("resnet_variant", "18")
        stage_taxonomy_layers = kwargs.get("stage_taxonomy_layers", (5, 6, 7, 8))
        stage_strides = kwargs.get("stage_strides", (1, 2, 2, 2))
        stage_blocks = kwargs.get("stage_blocks", None)
        temperature = kwargs.get("temperature", 1.0)
        hard = kwargs.get("hard", False)
        kernel_size = kwargs.get("kernel_size", 3)
        use_stem = kwargs.get("use_stem", True)
        stem_channels = kwargs.get("stem_channels", 64)
        stem_stride = kwargs.get("stem_stride", 2)
        use_stem_maxpool = kwargs.get("use_stem_maxpool", True)
        depth_decay = kwargs.get("depth_decay", 0.5)

        super().__init__(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_taxonomy_layers=stage_taxonomy_layers,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            temperature=temperature,
            hard=hard,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
            depth_decay=depth_decay,
        )

        # override the stages created by the base class
        self.taxon_stages = nn.ModuleList()
        current_channels = self.stage_input_channels[0] if self.stage_input_channels else in_channels
        for blocks, n_layers, stride in zip(self.stage_blocks, self.stage_taxonomy_layers, self.stage_strides):
            stage = TaxonResNetStageWithAttention(
                in_channels=current_channels,
                n_taxonomy_layers=n_layers,
                n_blocks=blocks,
                stride=stride,
                kernel_size=kernel_size,
                temperature=self.temperature,
                hard=self.default_hard,
                depth_decay=depth_decay,
                attn_heads=attn_heads,
            )
            self.taxon_stages.append(stage)
            current_channels = stage.total_out_channels

        self.final_channels = current_channels


__all__ = [
    "resolve_resnet_stage_blocks",
    "ResidualConvBlock",
    "TaxonResNetStage",
    "TaxonResNetEncoder",
    "TaxonResNetStageWithAttention",
    "TaxonResNetEncoderWithAttention",
]
