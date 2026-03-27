"""CNN taxon autoencoder models — hierarchical pairwise-softmax routing."""

from .encoder import (
    TaxonResNetEncoder,
    TaxonResNetStage,
    ResidualConvBlock,
    resolve_resnet_stage_blocks,
    MultiTaxonResNetStage,
    MultiTaxonResNetEncoder,
)
from .decoder import TaxonResNetDecoder, TaxonDecodeStage, ResidualDeconvBlock
from .taxon_ae import TaxonAutoencoder
from .multi_taxon_ae import MultiTaxonAutoencoder
from .topk_taxon_ae import TopKTaxonAutoencoder
from .topk_multi_taxon_ae import TopKMultiTaxonAutoencoder
from .bias_taxon_ae import BiasTaxonAutoencoder
from .bias_multi_taxon_ae import BiasMultiTaxonAutoencoder

__all__ = [
    "TaxonResNetEncoder",
    "TaxonResNetStage",
    "ResidualConvBlock",
    "resolve_resnet_stage_blocks",
    "MultiTaxonResNetStage",
    "MultiTaxonResNetEncoder",
    "TaxonResNetDecoder",
    "TaxonDecodeStage",
    "ResidualDeconvBlock",
    "TaxonAutoencoder",
    "MultiTaxonAutoencoder",
    "TopKTaxonAutoencoder",
    "TopKMultiTaxonAutoencoder",
    "BiasTaxonAutoencoder",
    "BiasMultiTaxonAutoencoder",
]
