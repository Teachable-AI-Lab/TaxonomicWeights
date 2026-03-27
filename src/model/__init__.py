"""TaxonomicWeights model package.

Subpackages:
    cnn/taxon     — CNN autoencoders with taxonomy routing
    cnn/baseline  — CNN sparse autoencoders (L1, TopK, Gated, JumpReLU)
    saebench      — SAEBench-compatible linear SAEs for LLM residual streams
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

# ── SAEBench linear SAE models ───────────────────────────────────────────
from .saebench.taxon_sae import TaxonSAE
from .saebench.multi_taxon_sae import MultiTaxonSAE

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
    # SAEBench linear
    "TaxonSAE",
    "MultiTaxonSAE",
]
