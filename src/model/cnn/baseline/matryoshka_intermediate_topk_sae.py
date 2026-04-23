"""Intermediate-Stage Matryoshka Batch-TopK Sparse Convolutional Autoencoder.

Same backbone as :class:`MatryoshkaBatchTopKSparseConvAutoencoder`
(plain ResNet-18 stem + 4 residual stages + mirror decoder), but sparsity is
applied at **every encoder stage's output** rather than only at the final
latent — analogous to how the taxon AE family enforces taxonomic routing at
every stage.

Differences vs. the standard matryoshka SAE:

1. **Per-stage Batch-TopK**: after stage *i* runs, its activation is passed
   through ``batch_topk_activation(·, k_values[i])`` and the sparsified map
   is what flows into stage *i+1*.  Deeper stages therefore see (and must
   reconstruct from) sparse inputs.
2. **Per-stage decode**: each intermediate sparse activation is reconstructed
   to image space by its own mirror decoder
   (a :class:`TaxonResNetDecoder` covering stages ``i..N-1``).
3. **Matryoshka loss across stages**: training minimises a weighted sum of
   per-stage MSEs ``Σ_i loss_weights[i] * MSE(recon_i, x)`` plus per-stage
   AuxK ``sparsity_weight * Σ_i auxk_i``.

Inference uses the deepest decoder (largest receptive field, most processed
features), matching the convention used by the rest of the SAE family.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..taxon.encoder import resolve_resnet_stage_blocks
from ..taxon.decoder import TaxonResNetDecoder
from .sae_encoder import ConvSAEStage
from .matryoshka_batch_topk_sae_encoder import batch_topk_activation, batch_auxk_loss


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class IntermediateMatryoshkaBatchTopKSAEEncoder(nn.Module):
    """Encoder that applies batch-global TopK at *every* stage output."""

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
        self.use_stem = bool(use_stem)
        self.sparsity_type = "matryoshka_intermediate_batch_topk"

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
        """Run stem + stages, sparsifying every stage output via Batch-TopK.

        Returns:
            ``(latents, info)`` where ``latents[i]`` is the sparse activation
            after stage *i* (``[B, stage_channels[i], H_i, W_i]``) and
            ``info["sparsity"]`` is the *summed* AuxK loss across stages
            (zero unless ``self.training and use_aux_loss``).
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
        sparsity = h.new_zeros(())

        for i, stage in enumerate(self.stages):
            pre = stage(h)
            sparse = batch_topk_activation(pre, self.k_values[i])
            latents.append(sparse)
            if self.use_aux_loss and self.training:
                sparsity = sparsity + batch_auxk_loss(
                    pre, sparse, self.k_values[i], self.k_aux, self.dead_threshold
                )
            h = sparse  # sparsified activation flows forward
            if return_details:
                details["shape_trace"].append((f"stage{i}_topk_k{self.k_values[i]}", tuple(sparse.shape)))
                details["stage_shapes"].append(tuple(sparse.shape))

        details["sparsity"] = sparsity
        details["k_values"] = self.k_values
        details["latent_shape"] = tuple(latents[-1].shape)
        return latents, details


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class IntermediateMatryoshkaBatchTopKSparseConvAutoencoder(nn.Module):
    """SAE with Batch-TopK sparsity at every stage, decoded per-stage.

    Architecture (channels / strides) is identical to
    :class:`MatryoshkaBatchTopKSparseConvAutoencoder`; only the placement of
    sparsity and the multi-headed decoder differ.
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
        kernel_size: int = 3,
        use_stem: bool = True,
        stem_channels: int = 64,
        stem_stride: int = 2,
        use_stem_maxpool: bool = True,
        output_activation: str = "none",
    ) -> None:
        super().__init__()

        self.output_activation = output_activation.lower()

        self.encoder = IntermediateMatryoshkaBatchTopKSAEEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_channels=stage_channels,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            k_values=k_values,
            k_aux=k_aux,
            use_aux_loss=use_aux_loss,
            dead_threshold=dead_threshold,
            kernel_size=kernel_size,
            use_stem=use_stem,
            stem_channels=stem_channels,
            stem_stride=stem_stride,
            use_stem_maxpool=use_stem_maxpool,
        )

        # Per-stage decoders.  Decoder i mirrors stages 0..i (and the stem)
        # because the i-th latent has only been downsampled by stages 0..i.
        # Mirroring stages i..N-1 would apply the wrong strides and blow up
        # the spatial extent for shallow stages.
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
        raise ValueError(
            f"Unsupported output_activation='{self.output_activation}'."
        )

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
    # Inference forward (uses deepest stage)
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
    # Training forward (per-stage decodes)
    # -----------------------------------------------------------------------

    def forward_matryoshka(
        self,
        x: torch.Tensor,
        use_checkpoint: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Decode every intermediate sparse activation back to image space.

        Each per-stage decoder is run under
        :func:`torch.utils.checkpoint.checkpoint` (non-reentrant) so its
        activations are recomputed during the backward pass instead of being
        held in memory.  This is essential here because the shallow-stage
        decoders operate on large spatial feature maps and would otherwise
        cause OOM when all N decoders are held in memory simultaneously.

        Returns:
            ``(recons, aux_loss)`` where ``recons[i]`` is the reconstruction
            decoded from the ``i``-th stage's sparse latent and ``aux_loss``
            is the *summed* AuxK loss across stages.
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
        return recons, info["sparsity"]


__all__ = [
    "IntermediateMatryoshkaBatchTopKSAEEncoder",
    "IntermediateMatryoshkaBatchTopKSparseConvAutoencoder",
]
