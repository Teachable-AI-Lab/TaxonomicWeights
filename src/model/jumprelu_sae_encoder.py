"""JumpReLU Convolutional SAE encoder.

Implements the **JumpReLU SAE** architecture
(Rajamanoharan et al., Google DeepMind 2024 —
"Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse
Autoencoders").

Core idea: replace the L1 penalty / hard TopK / gating with a *learned
per-channel threshold* θ.  Each latent channel fires if and only if its
pre-activation exceeds θ:

    f_c(z) = relu(z_c) · H(z_c − θ_c)

where H is the Heaviside step function.  θ is stored in log-space
(``log_theta``) so it is always positive.

Sparsity is controlled by targeting a desired mean L0 count (active features
per spatial position).  The sparsity loss

    L_sparse = (L̂₀ − target_l0)²

penalises deviations from ``target_l0``.  Gradients through the
non-differentiable Heaviside are estimated with a Straight-Through Estimator
(STE) using a rectangle-kernel window of width ``bandwidth``.

Interface is identical to the other SAE encoders::

    latent, info = encoder(x)

where ``info["sparsity"]`` is the raw sparsity penalty scalar
``(l0_hat − target_l0)²`` and ``info["l0_hat"]`` is the current mean L0.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .encoder import resolve_resnet_stage_blocks
from .sae_encoder import ConvSAEStage


# ---------------------------------------------------------------------------
# STE autograd functions
# ---------------------------------------------------------------------------

class _JumpReLUActivation(Function):
    """Applies relu(z) * H(z − θ) with STE gradient through the threshold.

    Forward:  latent_c = relu(z_c) · H(z_c − θ_c)
    Backward:
      * grad_z: relu'(z) · H(z − θ)  — gate, straight-through for relu
      * grad_log_theta: STE via rectangle-kernel approximation of ∂H/∂θ
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        z: torch.Tensor,
        log_theta: torch.Tensor,
        bandwidth: float,
    ) -> torch.Tensor:
        theta = log_theta.exp().view(1, -1, 1, 1)
        ctx.save_for_backward(z, log_theta)
        ctx.bandwidth = bandwidth
        gate = (z > theta).to(z.dtype)
        return F.relu(z) * gate

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        z, log_theta = ctx.saved_tensors
        bw = ctx.bandwidth
        theta = log_theta.exp()
        t = theta.view(1, -1, 1, 1)

        # Gradient w.r.t. z: relu'(z) * H(z - theta)
        relu_deriv = (z > 0).to(z.dtype)
        gate = (z > t).to(z.dtype)
        grad_z = grad_output * relu_deriv * gate

        # Gradient w.r.t. log_theta via STE:
        #   d(relu(z) · H(z-θ)) / dθ  ≈  -relu(z) · K_bw(z − θ)
        #   d(...)              / d(log θ) = d(...)/dθ · θ
        window = ((z - t).abs() <= bw / 2).to(z.dtype) / bw
        grad_log_theta = (grad_output * (-F.relu(z)) * window * t).sum(dim=(0, 2, 3))

        return grad_z, grad_log_theta, None


class _L0SurrogateFunction(Function):
    """STE-based L0 count for the sparsity penalty.

    Forward:  mean number of active channels per spatial position.
    Backward: STE gradient of the count w.r.t. ``log_theta`` only.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        z: torch.Tensor,
        log_theta: torch.Tensor,
        bandwidth: float,
    ) -> torch.Tensor:
        theta = log_theta.exp().view(1, -1, 1, 1)
        ctx.save_for_backward(z, log_theta)
        ctx.bandwidth = bandwidth
        # Mean active features per (batch, H, W) position
        return (z > theta).to(z.dtype).sum(dim=1).mean()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        z, log_theta = ctx.saved_tensors
        bw = ctx.bandwidth
        theta = log_theta.exp()
        t = theta.view(1, -1, 1, 1)
        B, C, H, W = z.shape
        n_pos = float(B * H * W)

        # d L0_mean / dθ_c ≈ −mean_{b,h,w} K_bw(z_c − θ_c)
        # d L0_mean / d(log θ_c) = d L0_mean / dθ_c · θ_c
        window = ((z - t).abs() <= bw / 2).to(z.dtype) / bw
        grad_log_theta = grad_output * ((-window * t).sum(dim=(0, 2, 3)) / n_pos)

        return None, grad_log_theta, None


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class JumpReLUSAEEncoder(nn.Module):
    """ResNet-style encoder for the JumpReLU Sparse Convolutional Autoencoder.

    Identical backbone to the other SAE encoders.  The final activation uses a
    *learned per-channel threshold* θ = exp(log_theta) stored as an
    ``nn.Parameter``.  Sparsity is encouraged by penalising the mean
    active-feature count (L̂₀) deviating from ``target_l0``.

    Args:
        in_channels: Input image channels.
        resnet_variant: ResNet preset for stage block counts.
        stage_channels: Output channel count per stage.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit per-stage block counts (overrides
            ``resnet_variant`` if provided).
        target_l0: Desired mean number of active channels per spatial
            position (analogous to ``topk_k`` in TopK SAE).
        bandwidth: STE window width used to approximate ∂H/∂θ.
            Typically 0.001 (from Rajamanoharan et al. 2024).
        theta_init: Initial threshold value; stored as ``log(theta_init)``.
        kernel_size: Conv kernel size throughout.
        use_stem: Include the 7×7 conv stem.
        stem_channels: Output channels of the stem.
        stem_stride: Stride of the stem conv.
        use_stem_maxpool: Append 3×3 max-pool after stem.
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        target_l0: float = 64.0,
        bandwidth: float = 0.001,
        theta_init: float = 0.1,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
    ) -> None:
        super().__init__()

        resolved_blocks = resolve_resnet_stage_blocks(
            resnet_variant=resnet_variant,
            stage_blocks=stage_blocks,
        )

        if not (len(resolved_blocks) == len(stage_channels) == len(stage_strides)):
            raise ValueError(
                "stage_blocks, stage_channels, and stage_strides must have equal length. "
                f"Got {len(resolved_blocks)}, {len(stage_channels)}, {len(stage_strides)}."
            )

        self.in_channels = in_channels
        self.resnet_variant = str(resnet_variant)
        self.stage_blocks = tuple(int(v) for v in resolved_blocks)
        self.stage_channels = tuple(int(v) for v in stage_channels)
        self.stage_strides = tuple(int(v) for v in stage_strides)
        self.target_l0 = float(target_l0)
        self.bandwidth = float(bandwidth)
        self.theta_init = float(theta_init)
        self.sparsity_type = "jumprelu"
        self.latent_activation = "jumprelu"
        self.use_stem = bool(use_stem)

        # ----- Stem --------------------------------------------------------
        if use_stem:
            stem_ops: List[nn.Module] = [
                nn.Conv2d(
                    in_channels, stem_channels,
                    kernel_size=7, stride=stem_stride, padding=3, bias=False,
                ),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True),
            ]
            if use_stem_maxpool:
                stem_ops.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.stem = nn.Sequential(*stem_ops)
            current_channels = stem_channels
            self.stem_total_stride = stem_stride * (2 if use_stem_maxpool else 1)
        else:
            self.stem = nn.Identity()
            current_channels = in_channels
            self.stem_total_stride = 1

        self.stem_channels_out = stem_channels if use_stem else in_channels

        # ----- Residual stages ---------------------------------------------
        self.stage_input_channels: List[int] = []
        self.stages = nn.ModuleList()

        for n_blocks, out_ch, stride in zip(
            self.stage_blocks, self.stage_channels, self.stage_strides
        ):
            self.stage_input_channels.append(current_channels)
            self.stages.append(
                ConvSAEStage(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    n_blocks=n_blocks,
                    stride=stride,
                    kernel_size=kernel_size,
                )
            )
            current_channels = out_ch

        self.final_channels = current_channels

        # ----- Learned threshold -------------------------------------------
        # Stored as log(θ) so that θ = exp(log_theta) is always positive.
        init_log = float(torch.tensor(theta_init).log().item())
        self.log_theta = nn.Parameter(torch.full((self.final_channels,), init_log))

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode an image batch to a JumpReLU-sparse latent feature map.

        Args:
            x: Input image tensor ``[B, C, H, W]``.
            return_details: If ``True``, populate ``shape_trace`` in details.

        Returns:
            ``(latent, info)`` where ``info["sparsity"]`` is the raw
            ``(l0_hat − target_l0)²`` scalar and ``info["l0_hat"]`` is the
            current mean active-feature count per spatial position.
        """
        details: Dict[str, object] = {"input_shape": tuple(x.shape), "shape_trace": []}

        x = self.stem(x)
        if return_details:
            details["shape_trace"].append(("stem", tuple(x.shape)))

        for idx, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if return_details:
                details["shape_trace"].append((f"stage{idx}", tuple(x.shape)))

        # JumpReLU activation with STE for log_theta
        latent = _JumpReLUActivation.apply(x, self.log_theta, self.bandwidth)

        # STE-based L0 estimate for the sparsity loss
        l0_hat = _L0SurrogateFunction.apply(x, self.log_theta, self.bandwidth)

        # Sparsity penalty: squared deviation from target (used in training loss)
        sparsity = (l0_hat - self.target_l0) ** 2

        theta_vals = self.log_theta.exp()
        details["latent_shape"]  = tuple(latent.shape)
        details["sparsity"]      = sparsity
        details["l0_hat"]        = l0_hat
        details["theta_mean"]    = float(theta_vals.mean().detach().cpu())
        details["theta_min"]     = float(theta_vals.min().detach().cpu())
        details["theta_max"]     = float(theta_vals.max().detach().cpu())

        return latent, details


__all__ = [
    "_JumpReLUActivation",
    "_L0SurrogateFunction",
    "JumpReLUSAEEncoder",
]
