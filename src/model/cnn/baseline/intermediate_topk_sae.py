"""Per-Position Intermediate TopK Sparse Convolutional Autoencoder.

Like :class:`IntermediateMatryoshkaBatchTopKSparseConvAutoencoder` (TopK applied at
every stage output + per-stage decoders + matryoshka-style weighted loss) but
uses **per-position** TopK instead of batch-global TopK.

Per-position TopK forces *exactly* ``k_values[i]`` active channels at every
spatial location independently, giving deterministic sparsity regardless of
batch contents.  Batch-global TopK lets the total active count float per
spatial location based on global statistics.

Architecture is otherwise identical:
- Stem → N residual stages
- ``topk_activation(stage_out, k_values[i])`` after each stage
- Sparse activation flows into the next stage
- One :class:`TaxonResNetDecoder` per stage (covers stages 0..i + stem)
- Training: ``Σ_i loss_weights[i] * MSE(decode_i(z_i), x)  +  sparsity_weight * Σ_i auxk_i``
- Inference: deepest stage decoder
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ..taxon.encoder import resolve_resnet_stage_blocks
from ..taxon.decoder import TaxonResNetDecoder
from .sae_encoder import ConvSAEStage
from .topk_sae_encoder import topk_activation, auxk_loss, make_dead_latent


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class IntermediateTopKSAEEncoder(nn.Module):
    """Encoder that applies per-position TopK at *every* stage output."""

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        k_values: Sequence[int] = (32, 16, 8, 4),
        k_aux: Optional[int] = None,
        use_aux_loss: bool = True,
        dead_threshold: float = 1e-3,
        dead_steps: int = 200,
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
        if not (len(resolved_blocks) == len(stage_channels) == len(stage_strides) == len(k_values)):
            raise ValueError(
                "stage_blocks, stage_channels, stage_strides and k_values must all have equal length. "
                f"Got {len(resolved_blocks)}, {len(stage_channels)}, {len(stage_strides)}, {len(k_values)}."
            )

        self.in_channels = int(in_channels)
        self.resnet_variant = str(resnet_variant)
        self.stage_blocks = tuple(int(v) for v in resolved_blocks)
        self.stage_channels = tuple(int(v) for v in stage_channels)
        self.stage_strides = tuple(int(v) for v in stage_strides)
        self.k_values = tuple(int(v) for v in k_values)
        self.k_aux = int(k_aux) if k_aux is not None else int(self.k_values[-1])
        self.use_aux_loss = bool(use_aux_loss)
        self.dead_threshold = float(dead_threshold)
        self.dead_steps = int(dead_steps)
        self.use_stem = bool(use_stem)
        self.sparsity_type = "intermediate_topk"

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

        # ----- Stages ------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[List[torch.Tensor], Dict[str, object]]:
        """Run stem + stages, sparsifying every stage output via per-position TopK.

        Returns:
            ``(latents, info)`` where ``latents[i]`` is the sparse activation
            after stage *i* and ``info["sparsity"]`` is the summed AuxK loss
            across stages (zero unless ``self.training and use_aux_loss``).
        """
        details: Dict[str, object] = {
            "input_shape": tuple(x.shape),
            "shape_trace": [],
            "stage_shapes": [],
        }

        h = self.stem(x)
        if return_details:
            details["shape_trace"].append(("stem", tuple(h.shape)))

        latents: List[torch.Tensor] = []
        dead_latents: List[Optional[torch.Tensor]] = []

        for i, stage in enumerate(self.stages):
            pre = stage(h)
            sparse = topk_activation(pre, self.k_values[i])
            latents.append(sparse)

            # Per-stage dead-channel tracking + AuxK dead latent
            C = int(sparse.shape[1])
            buf_name = f"_steps_since_active_{i}"
            if not hasattr(self, buf_name) or getattr(self, buf_name).numel() != C:
                self.register_buffer(
                    buf_name,
                    torch.zeros(C, dtype=torch.long, device=sparse.device),
                    persistent=False,
                )
            buf = getattr(self, buf_name)
            if self.training:
                with torch.no_grad():
                    any_active = sparse.amax(dim=(0, 2, 3)) > self.dead_threshold
                    buf[any_active] = 0
                    buf[~any_active] += 1
            if self.use_aux_loss and self.training:
                dead_latents.append(make_dead_latent(
                    pre, buf, k_aux=self.k_aux,
                    k=self.k_values[i], dead_steps=self.dead_steps,
                ))
            else:
                dead_latents.append(None)

            h = sparse  # sparsified activation flows into next stage
            if return_details:
                details["shape_trace"].append((f"stage{i}_topk_k{self.k_values[i]}", tuple(sparse.shape)))
                details["stage_shapes"].append(tuple(sparse.shape))

        details["sparsity"] = h.new_zeros(())   # back-compat (model computes real aux_loss)
        details["dead_latents"] = dead_latents
        details["k_values"] = self.k_values
        details["latent_shape"] = tuple(latents[-1].shape)
        return latents, details


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class IntermediateTopKSparseConvAutoencoder(nn.Module):
    """SAE with per-position TopK sparsity at every stage, decoded per-stage.

    Each per-stage decoder covers only the stages that downsampled the
    corresponding latent (stages 0..i), so spatial sizes are correctly
    inverted by each decoder head.

    Inference uses the deepest decoder.
    Training uses :meth:`forward_matryoshka` with gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        k_values: Sequence[int] = (32, 16, 8, 4),
        k_aux: Optional[int] = None,
        use_aux_loss: bool = True,
        dead_threshold: float = 1e-3,
        dead_steps: int = 200,
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
    ) -> None:
        super().__init__()

        self.output_activation = output_activation.lower()

        self.encoder = IntermediateTopKSAEEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_channels=stage_channels,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            k_values=k_values,
            k_aux=k_aux,
            use_aux_loss=use_aux_loss,
            dead_threshold=dead_threshold,
            dead_steps=dead_steps,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
        )

        # Decoder i mirrors stages 0..i + stem — same convention as the
        # intermediate matryoshka SAE.
        self.decoders = nn.ModuleList()
        N = len(self.encoder.stages)
        for i in range(N):
            self.decoders.append(
                TaxonResNetDecoder(
                    latent_channels=self.encoder.stage_channels[i],
                    stage_input_channels=self.encoder.stage_input_channels[: i + 1],
                    stage_blocks=self.encoder.stage_blocks[: i + 1],
                    stage_strides=self.encoder.stage_strides[: i + 1],
                    out_channels=in_channels,
                    use_stem=use_stem,
                    stem_total_stride=self.encoder.stem_total_stride,
                    kernel_size=kernel_size,
                )
            )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation in {"identity", "none", "linear"}:
            return x
        raise ValueError(f"Unsupported output_activation='{self.output_activation}'.")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[List[torch.Tensor], Dict[str, object]]:
        return self.encoder(x, return_details=return_details)

    def decode_stage(
        self,
        z: torch.Tensor,
        stage_idx: int,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        recon, _ = self.decoders[stage_idx](z, output_size=output_size)
        return self._apply_output_activation(recon)

    # -----------------------------------------------------------------------
    # Inference forward (deepest stage)
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ):
        latents, enc_details = self.encode(x, return_details=return_details)
        deepest = len(latents) - 1
        recon, dec_details = self.decoders[deepest](
            latents[deepest], output_size=x.shape[-2:], return_details=return_details
        )
        recon = self._apply_output_activation(recon)

        # Decoder-based AuxK on deepest stage (matches inference path).
        dead_latents = enc_details.get("dead_latents", [None] * len(latents))
        dl = dead_latents[deepest] if dead_latents else None
        if dl is not None and self.training:
            dead_recon, _ = self.decoders[deepest](dl, output_size=x.shape[-2:])
            dead_recon = self._apply_output_activation(dead_recon)
            aux_loss = F.mse_loss(dead_recon, (x - recon).detach())
        else:
            aux_loss = enc_details["sparsity"]

        if not return_details:
            return recon, aux_loss

        details = {
            "input_shape":  tuple(x.shape),
            "latent_shape": tuple(latents[deepest].shape),
            "recon_shape":  tuple(recon.shape),
            "encoder":      enc_details,
            "decoder":      dec_details,
        }
        return recon, aux_loss, details

    # -----------------------------------------------------------------------
    # Training forward (matryoshka per-stage decodes)
    # -----------------------------------------------------------------------

    def forward_matryoshka(
        self,
        x: torch.Tensor,
        use_checkpoint: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Decode every intermediate sparse latent back to image space.

        Shallow decoders operate on large spatial maps; gradient checkpointing
        is used to avoid holding all decoder activations in memory at once.

        Returns:
            ``(recons, aux_loss)`` where ``recons[i]`` is the reconstruction
            from stage *i*'s sparse latent and ``aux_loss`` is the summed
            per-position AuxK loss across stages.
        """
        latents, info = self.encoder(x)
        H, W = x.shape[-2:]
        recons: List[torch.Tensor] = []

        def _decode(z: torch.Tensor, idx: int) -> torch.Tensor:
            recon, _ = self.decoders[idx](z, output_size=(H, W))
            return self._apply_output_activation(recon)

        for i, z in enumerate(latents):
            if use_checkpoint and self.training and torch.is_grad_enabled():
                recon = torch.utils.checkpoint.checkpoint(
                    _decode, z, i, use_reentrant=False
                )
            else:
                recon = _decode(z, i)
            recons.append(recon)

        # Per-stage decoder-based AuxK summed across stages.
        dead_latents = info.get("dead_latents", [None] * len(latents))
        aux_loss = recons[0].new_zeros(())
        if self.training:
            for i, dl in enumerate(dead_latents):
                if dl is None:
                    continue
                dead_recon, _ = self.decoders[i](dl, output_size=(H, W))
                dead_recon = self._apply_output_activation(dead_recon)
                aux_loss = aux_loss + F.mse_loss(
                    dead_recon, (x - recons[i]).detach()
                )

        return recons, aux_loss


__all__ = [
    "IntermediateTopKSAEEncoder",
    "IntermediateTopKSparseConvAutoencoder",
]
