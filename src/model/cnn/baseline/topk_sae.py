"""TopK Sparse Convolutional Autoencoder wrapper.

Composes:
- :class:`~src.model.topk_sae_encoder.TopKSAEEncoder`: TopK-activated encoder.
- :class:`~src.model.decoder.TaxonResNetDecoder`: mirror decoder (reused).

The forward signature mirrors :class:`SparseConvAutoencoder`:

    ``recon, aux_loss = model(x)``

where ``aux_loss`` is the optional AuxK auxiliary loss (0 when disabled).
Scale it with an external ``sparsity_weight``; for pure TopK without AuxK,
set ``sparsity_weight=0`` in the training config.

For full debugging info::

    recon, aux_loss, details = model(x, return_details=True)
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .topk_sae_encoder import TopKSAEEncoder
from ..taxon.decoder import TaxonResNetDecoder


class TopKSparseConvAutoencoder(nn.Module):
    """TopK Sparse Convolutional Autoencoder.

    Same architecture as :class:`SparseConvAutoencoder` but with a hard
    TopK latent activation (Gao et al. 2024) instead of L1/KL penalty.
    Sparsity is controlled exclusively by ``topk_k`` (number of active
    channels per spatial position); no penalty weight is required.

    Args:
        in_channels: Input image channels (default 3).
        resnet_variant: ResNet preset for stage block counts.
        stage_channels: Output channel count per encoder stage.
        stage_strides: Spatial stride per stage (first block only).
        stage_blocks: Explicit block counts (overrides ``resnet_variant``).
        topk_k: Active channels to retain per spatial position.
        k_aux: AuxK features for the auxiliary dead-neuron loss.
        use_aux_loss: Enable the AuxK auxiliary loss.
        dead_threshold: Activation threshold for dead-neuron detection.
        kernel_size: Conv kernel size throughout.
        use_stem: Include 7×7 stem conv.
        stem_channels: Stem output channels.
        stem_stride: Stem conv stride.
        use_stem_maxpool: Append 3×3 max-pool after stem.
        output_activation: Activation on reconstruction output
            (``"none"`` / ``"tanh"`` / ``"sigmoid"``).
    """

    def __init__(
        self,
        in_channels: int = 3,
        resnet_variant: str | int = "18",
        stage_channels: Sequence[int] = (64, 128, 256, 512),
        stage_strides: Sequence[int] = (1, 2, 2, 2),
        stage_blocks: Optional[Sequence[int]] = None,
        topk_k: int = 64,
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

        self.encoder = TopKSAEEncoder(
            in_channels=in_channels,
            resnet_variant=resnet_variant,
            stage_channels=stage_channels,
            stage_strides=stage_strides,
            stage_blocks=stage_blocks,
            topk_k=topk_k,
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

        self.decoder = TaxonResNetDecoder(
            latent_channels=self.encoder.final_channels,
            stage_input_channels=self.encoder.stage_input_channels,
            stage_blocks=self.encoder.stage_blocks,
            stage_strides=self.encoder.stage_strides,
            out_channels=in_channels,
            use_stem=use_stem,
            stem_total_stride=self.encoder.stem_total_stride,
            kernel_size=kernel_size,
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
            f"Unsupported output_activation='{self.output_activation}'. "
            "Use 'none'/'identity'/'linear', 'tanh', or 'sigmoid'."
        )

    # -----------------------------------------------------------------------
    # Public encode / decode API (mirrors SparseConvAutoencoder)
    # -----------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Encode an image batch to a TopK-sparse latent feature map."""
        return self.encoder(x, return_details=return_details)

    def decode(
        self,
        z: torch.Tensor,
        output_size: Optional[Tuple[int, int]] = None,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Decode a latent feature map to image space."""
        return self.decoder(z, output_size=output_size, return_details=return_details)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ):
        """Full autoencoder pass.

        Args:
            x: Input image tensor ``[B, C, H, W]``.
            return_details: Return shape-trace details dict.

        Returns:
            ``(recon, aux_loss)`` normally, or
            ``(recon, aux_loss, details)`` when ``return_details=True``.

            ``aux_loss`` is the raw (unweighted) AuxK auxiliary scalar.
            It is 0 when ``use_aux_loss=False`` or during evaluation.
        """
        latent, enc_details = self.encode(x, return_details=return_details)
        recon, dec_details = self.decode(
            latent, output_size=x.shape[-2:], return_details=return_details
        )
        recon = self._apply_output_activation(recon)

        # Decoder-based AuxK: decode the dead-only sparse latent and MSE
        # against the residual.  See encoder.make_dead_latent docstring.
        dead_latent = enc_details.get("dead_latent")
        if dead_latent is not None and self.training:
            dead_recon, _ = self.decoder(dead_latent, output_size=x.shape[-2:])
            error = (x - recon).detach()
            aux_loss = F.mse_loss(dead_recon, error)
        else:
            aux_loss = enc_details["sparsity"]

        if not return_details:
            return recon, aux_loss

        details = {
            "input_shape":  tuple(x.shape),
            "latent_shape": tuple(latent.shape),
            "recon_shape":  tuple(recon.shape),
            "encoder":      enc_details,
            "decoder":      dec_details,
        }
        return recon, aux_loss, details


__all__ = ["TopKSparseConvAutoencoder"]
