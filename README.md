# Taxonomic Weights

## Karthik Branch!!

New plan - minimize KL-divergence from uniform distribution while also minimizing KL-divergence for individual instances from the path!!

We can do this by minimizing the entropy of each distribution

## Notes on Zekun's prior code

*   `taxon-conv-weight.ipynb`
    *   This is the things with alphas (also found in `intuitions.pdf`) - each higher-level node is a direct mixing of its lower-level nodes, as a result of an alpha parameterization
*   `taxon-conv.ipynb`
    *   This is more similar to how the Deep Taxonomic Networks functioned - a KL divergence is imposed on the hierarchy of the weights
