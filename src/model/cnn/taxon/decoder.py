"""ResNet-style decoder modules for the taxon autoencoder.

The decoder mirrors encoder stages in reverse order with residual
resize-convolution blocks.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDeconvBlock(nn.Module):
    """Pre-activation residual upsampling block.

    Structure: ``BN -> Upsample(stride) -> Conv -> ReLU -> BN -> Conv + skip``.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2) -> None:
        super().__init__()
        padding = kernel_size // 2
        upsample = nn.Upsample(scale_factor=stride, mode="nearest") if stride != 1 else nn.Identity()

        self.main = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            upsample,
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
        )

        if (in_channels != out_channels) or (stride != 1):
            self.skip = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest") if stride != 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) + self.skip(x)


class TaxonDecodeStage(nn.Module):
    """One decoder stage built from residual upsampling blocks."""

    def __init__(self, in_channels: int, out_channels: int, n_blocks: int, stride: int, kernel_size: int = 3) -> None:
        super().__init__()
        blocks: List[nn.Module] = []
        for idx in range(n_blocks):
            block_in = in_channels if idx == 0 else out_channels
            block_stride = stride if idx == 0 else 1
            blocks.append(
                ResidualDeconvBlock(
                    in_channels=block_in,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=block_stride,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TaxonResNetDecoder(nn.Module):
    """Mirror decoder for :class:`taxon_resnet_encoder.TaxonResNetEncoder`."""

    def __init__(
        self,
        latent_channels: int,
        stage_input_channels: Sequence[int],
        stage_blocks: Sequence[int],
        stage_strides: Sequence[int],
        out_channels: int = 3,
        use_stem: bool = True,
        stem_total_stride: int = 4,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        if not (len(stage_input_channels) == len(stage_blocks) == len(stage_strides)):
            raise ValueError(
                "stage_input_channels, stage_blocks, and stage_strides must have equal lengths. "
                f"Got {len(stage_input_channels)}, {len(stage_blocks)}, {len(stage_strides)}"
            )

        self.stage_input_channels = tuple(int(v) for v in stage_input_channels)
        self.stage_blocks = tuple(int(v) for v in stage_blocks)
        self.stage_strides = tuple(int(v) for v in stage_strides)
        self.use_stem = bool(use_stem)
        self.stem_total_stride = int(stem_total_stride)

        current_channels = int(latent_channels)
        self.decoder_stages = nn.ModuleList()

        for rev_idx in reversed(range(len(self.stage_input_channels))):
            stage = TaxonDecodeStage(
                in_channels=current_channels,
                out_channels=self.stage_input_channels[rev_idx],
                n_blocks=self.stage_blocks[rev_idx],
                stride=self.stage_strides[rev_idx],
                kernel_size=kernel_size,
            )
            self.decoder_stages.append(stage)
            current_channels = self.stage_input_channels[rev_idx]

        if self.use_stem:
            self.stem_decoder = ResidualDeconvBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=max(1, self.stem_total_stride),
            )
        else:
            self.stem_decoder = (
                nn.Conv2d(current_channels, out_channels, kernel_size=1, stride=1)
                if current_channels != out_channels
                else nn.Identity()
            )

    def forward(
        self,
        z: torch.Tensor,
        output_size: Tuple[int, int] | None = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Decode latent feature map to image space."""
        x = z
        details: Dict[str, object] = {"shape_trace": [], "stages": []}

        for idx, stage in enumerate(self.decoder_stages, start=1):
            x = stage(x)
            if return_details:
                details["shape_trace"].append((f"dec_stage{idx}", tuple(x.shape)))
                details["stages"].append({"name": f"dec_stage{idx}", "shape": tuple(x.shape)})

        x = self.stem_decoder(x)
        if return_details:
            details["shape_trace"].append(("stem_decoder", tuple(x.shape)))

        if output_size is not None and x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
            if return_details:
                details["shape_trace"].append(("resize", tuple(x.shape)))

        return x, details


__all__ = [
    "ResidualDeconvBlock",
    "TaxonDecodeStage",
    "TaxonResNetDecoder",
]
