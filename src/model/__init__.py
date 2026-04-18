"""TaxonomicWeights model package.

Subpackages:
    cnn/taxon     — CNN autoencoders with taxonomy routing
    cnn/baseline  — CNN sparse autoencoders (L1, TopK, Gated, JumpReLU)
"""

# ── CNN taxon models ──────────────────────────────────────────────────────
from .cnn.taxon import (
    TaxonResNetEncoder,
    TaxonResNetStage,
    ResidualConvBlock,
    resolve_resnet_stage_blocks,
    MultiTaxonResNetStage,
    MultiTaxonResNetEncoder,
    TaxonResNetDecoder,
    TaxonDecodeStage,
    ResidualDeconvBlock,
    TaxonAutoencoder,
    MultiTaxonAutoencoder,
    TopKTaxonAutoencoder,
    TopKMultiTaxonAutoencoder,
    BiasTaxonAutoencoder,
    BiasMultiTaxonAutoencoder,
)

# ── CNN baseline SAE models ──────────────────────────────────────────────
from .cnn.baseline import (
    ConvSAEStage,
    ConvSAEEncoder,
    SparseConvAutoencoder,
    BaselineConvAutoencoder,
    TopKSAEEncoder,
    TopKSparseConvAutoencoder,
    GatedSAEEncoder,
    GatedSparseConvAutoencoder,
    JumpReLUSAEEncoder,
    JumpReLUSparseConvAutoencoder,
)

__all__ = [
    # CNN taxon
    "TaxonAutoencoder",
    "TaxonResNetEncoder",
    "TaxonResNetDecoder",
    "TaxonResNetStage",
    "TaxonDecodeStage",
    "ResidualConvBlock",
    "ResidualDeconvBlock",
    "resolve_resnet_stage_blocks",
    "MultiTaxonResNetStage",
    "MultiTaxonResNetEncoder",
    "MultiTaxonAutoencoder",
    "TopKTaxonAutoencoder",
    "TopKMultiTaxonAutoencoder",
    "BiasTaxonAutoencoder",
    "BiasMultiTaxonAutoencoder",
    # CNN baseline
    "SparseConvAutoencoder",
    "ConvSAEEncoder",
    "ConvSAEStage",
    "BaselineConvAutoencoder",
    "TopKSAEEncoder",
    "TopKSparseConvAutoencoder",
    "GatedSAEEncoder",
    "GatedSparseConvAutoencoder",
    "JumpReLUSAEEncoder",
    "JumpReLUSparseConvAutoencoder",
]
