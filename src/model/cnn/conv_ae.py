"""Strided-conv backbone (no residual connections) for conv_topk_sae_ae and
conv_bottleneck_topk_taxon_ae.

Encoder:  3 → 64 → 128 → 256 → 512 → 1024
          each via Conv2d(k=4, s=2, pad=1) + BN + LeakyReLU(0.2)

Decoder:  in_channels → 512 → 256 → 128 → 64 → out_channels
          each via ConvTranspose2d(k=4, s=2, pad=1) + BN + ReLU
          (final block: no BN, optional output activation)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StridedConvEncoder(nn.Module):
    """Five strided-conv blocks with no residual connections.

    Spatial path (for 256×256 input):
      256 → 128 → 64 → 32 → 16 → 8
    Channel path (default):
      3 → 64 → 128 → 256 → 512 → 1024
    """

    DEFAULT_CHANNELS: Sequence[int] = (3, 64, 128, 256, 512, 1024)

    def __init__(
        self,
        in_channels: int = 3,
        channel_list: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        ch = list(channel_list) if channel_list is not None else list(self.DEFAULT_CHANNELS)
        ch[0] = in_channels

        layers: list[nn.Module] = []
        for c_in, c_out in zip(ch[:-1], ch[1:]):
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.net = nn.Sequential(*layers)
        self.out_channels: int = ch[-1]
        self.channel_list: list[int] = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StridedConvDecoder(nn.Module):
    """Five transposed-conv blocks with no residual connections.

    Spatial path (for 8×8 latent → 256×256 output):
      8 → 16 → 32 → 64 → 128 → 256
    Channel path (default, decoder mirrors encoder):
      in_channels → 512 → 256 → 128 → 64 → out_channels

    The intermediate channel sizes are fixed at [512, 256, 128, 64] so the
    architecture is independent of the latent channel count (the first block
    projects from ``in_channels`` to 512).

    The final ConvTranspose2d has no BN; output activation is applied if
    ``output_activation`` is not ``"none"``/``"identity"``.
    """

    INNER_CHANNELS: Sequence[int] = (512, 256, 128, 64)

    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 3,
        output_activation: str = "none",
    ) -> None:
        super().__init__()
        self.output_activation = output_activation.lower()

        ch = [in_channels, *self.INNER_CHANNELS, out_channels]
        layers: list[nn.Module] = []
        for i, (c_in, c_out) in enumerate(zip(ch[:-1], ch[1:])):
            is_last = i == len(ch) - 2
            layers.append(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=is_last)
            )
            if not is_last:
                layers += [nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.in_channels = in_channels

    def forward(
        self,
        z: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        x = self.net(z)
        if output_size is not None and tuple(x.shape[-2:]) != tuple(output_size):
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        if self.output_activation == "tanh":
            x = torch.tanh(x)
        elif self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x
