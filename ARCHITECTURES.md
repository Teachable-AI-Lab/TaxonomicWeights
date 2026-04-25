# Architecture Descriptions

## TopK Sparse Convolutional Autoencoder (TopK SAE)

**Overview.** A convolutional autoencoder whose latent is forced to be maximally sparse by a hard TopK gate. It is the baseline from which all other variants extend.

**Encoder.** A standard ResNet-style encoder: a 7×7 stem (BN + ReLU + optional MaxPool) followed by $N$ residual stages, each stage containing several `ResidualConvBlock`s with strided downsampling. All stages are dense — no sparsity until the final latent map.

**Sparsity mechanism (TopK).** After the last encoder stage, the latent tensor $z \in \mathbb{R}^{B \times C \times H \times W}$ is processed by a TopK gate. The gate first applies ReLU, then at each spatial position $(b, h, w)$ keeps only the $k$ largest values and zeros the rest (per-position TopK). The threshold is computed with a straight-through estimator so gradients flow through it. Alternatively, *batch TopK* (`use_batch_topk=True`) applies a single global threshold across all $B \times H \times W$ positions, holding the total number of active channels constant at $k \times B \times H \times W$ while allowing variable per-position sparsity.

**Dead neuron revival (AuxK).** Channels that have not been activated in any sample for `dead_steps` consecutive training steps are considered dead. An auxiliary loss (`auxk_loss`) selects the top-$k_\text{aux}$ dead channels by their pre-ReLU value and penalises their non-activation via MSE against the current reconstruction error, forcing them to revive and take on useful representations.

**Decoder.** A mirrored `TaxonResNetDecoder`: bilinear upsampling + transposed-strided residual blocks mirroring each encoder stage in reverse, followed by a stem upsample to restore the original spatial resolution.

**Training objective.**
$$\mathcal{L} = \mathrm{MSE}(x, \hat{x}) + \lambda_\text{aux} \cdot \mathcal{L}_\text{auxk}$$

**Key hyperparameters.** `topk_k` (number of active channels), `k_aux`, `dead_steps`, `use_batch_topk`.

---

## Matryoshka Intermediate Batch-TopK SAE

**Overview.** Extends the TopK SAE with two ideas: (1) *intermediate* sparsity — TopK is applied at every encoder stage, not just the last; (2) *Matryoshka* nesting — each intermediate sparse latent is independently decoded and contributes to a nested weighted training loss, encouraging coarser representations at earlier stages to be self-sufficient.

**Encoder.** Same stem + $N$ residual stages. After each stage $i$, a batch-global TopK gate is applied with budget $k_i$ (`k_values[i]`). The sparse activation then flows into stage $i+1$, so each subsequent stage receives an already-sparse input.

**Per-stage decoders.** A dedicated `TaxonResNetDecoder` is attached at each stage $i$. Decoder $i$ covers only the remaining upsampling path from stage $i$ to the original resolution (i.e., it reverses stages $i, i+1, \ldots, N$ plus the stem upsample). This gives a nested family of reconstructions from coarse (early stage, low $k$) to fine (last stage, high $k$).

**Matryoshka loss.** At training time `forward_matryoshka(x)` decodes every intermediate sparse latent and returns one reconstruction per stage. The loss is:
$$\mathcal{L} = \sum_i w_i \cdot \mathrm{MSE}(x, \hat{x}_i) + \lambda_\text{sparsity} \cdot \mathcal{L}_\text{auxk}$$
where $w_i$ are per-stage loss weights (typically decaying, giving more weight to deeper stages). At inference, only the deepest decoder is used.

**Sparsity mechanism.** Identical to the TopK SAE but batch-global: across the batch a single global threshold keeps exactly $k_i \cdot (B \cdot H_i \cdot W_i)$ activations per stage.

**AuxK.** One AuxK buffer per stage; auxiliary losses are summed.

**Key hyperparameters.** `k_values` (per-stage active counts), `loss_weights`, `k_aux`, `dead_threshold`, `use_batch_topk`.

---

## Bottleneck TopK Taxon Autoencoder

**Overview.** Introduces *structured hierarchical sparsity* into the latent. Instead of selecting arbitrary top-k channels, the model organises channels into a binary taxonomy tree and enforces that only complete root-to-leaf paths survive. This makes every non-zero channel directly interpretable as a node in a shared conceptual hierarchy.

**Encoder.** A stem + $P$ plain (dense) ResNet stages reduce the spatial resolution and extract rich features. The final *bottleneck stage* is a `TopKTaxonResNetStage` with depth $L$ (`bottleneck_n_taxonomy_layers`).

**Taxonomy routing (hierarchical pairwise softmax).** The bottleneck stage maps its input through residual blocks to produce $C = 2^{L+1} - 2$ feature channels, split by depth into groups of $[2, 4, 8, \ldots, 2^L]$ channels. At each depth $d$, a pairwise softmax is applied to sibling pairs:
$$p_{d,2i},\ p_{d,2i+1} = \mathrm{softmax}\!\left(\frac{[l_{d,2i},\ l_{d,2i+1}]}{\tau}\right)$$
The full path probability to a depth-$d$ node is the product of all pairwise probabilities along the path from the root. The stage output is $\text{out}_{d,c} = l_{d,c} \times p_{d,c}$ — feature magnitude weighted by the routing probability, so only well-supported paths carry energy.

**TopK path selection.** After routing, a TopK gate (per-position or batch-global) keeps only the $k$ largest activations. With `k_leaves=1`, the gate is constrained so that exactly one complete root-to-leaf path (all $L$ depth nodes on a single branch) is active per spatial position, giving $L$ non-zero channels per location. `topk_k_multiplier` scales the effective $k$.

**AuxK revival.** Dead nodes (not activated for `dead_steps` steps) are identified per channel. An auxiliary convolution (`auxk_proj`) projects dead features to pixel space and penalises the MSE against the reconstruction residual, providing a strong per-pixel gradient signal to revive them.

**Decoder.** A single `TaxonResNetDecoder` inverts all stages back to the input resolution.

**Matryoshka-forward.** `forward_matryoshka(x)` produces $L$ reconstructions, one per depth prefix of the taxonomy (masking channels beyond depth $d$), enabling analysis of coarse-to-fine hierarchical decomposition.

**Key hyperparameters.** `bottleneck_n_taxonomy_layers` $L$ (tree depth; total channels $= 2^{L+1}-2$), `k_leaves` (0 = unconstrained TopK, 1 = one full path), `topk_k_multiplier`, `use_batch_topk`, `temperature`, `depth_decay` (per-depth regularisation weight decay), `dead_steps`.

---

## Bottleneck TopK Multi-Taxon Autoencoder

**Overview.** Generalises the single-hierarchy taxon AE to $K$ independent parallel taxonomy trees at the bottleneck. Each hierarchy independently routes and selects paths, allowing the model to represent $K$ separate conceptual "viewpoints" simultaneously per spatial location.

**Encoder.** Identical stem + $P$ plain stages as the single-taxon variant. The bottleneck is a `TopKMultiTaxonResNetStage` containing $K$ independent `TopKTaxonResNetStage` modules (`n_hierarchies=K`), each with depth $L$.

**Multi-hierarchy latent.** Each hierarchy $k$ produces its own sparse taxonomy latent of $2^{L+1}-2$ channels, and all $K$ outputs are concatenated: total latent width $= K \times (2^{L+1}-2)$. With `k_leaves=1`, each hierarchy contributes $L$ active channels per position, giving $K \times L$ total non-zero channels per spatial location.

**Inter-hierarchy gate (optional).** When `use_inter_hierarchy_gate=True`, a learned 1×1 conv (stride-matched to the bottleneck) produces $K$ logits per spatial location. A TopK selection with budget `gate_k` picks the `gate_k` most relevant hierarchies, zeroing the outputs of all others via a straight-through masked softmax. This provides a second level of sparsity — across hierarchies in addition to within-hierarchy path selection. With `gate_k=1` only the single best-matching hierarchy is active per location; with `gate_k=K` the gate is effectively disabled.

**AuxK.** Dead-node revival runs independently inside each hierarchy's `TopKTaxonResNetStage`; the AuxK losses are summed over all $K$ hierarchies.

**Decoder.** A single shared `TaxonResNetDecoder` receives the concatenated $K$-hierarchy latent and reconstructs the image.

**Matryoshka-forward.** `forward_matryoshka(x)` masks each hierarchy simultaneously to depth-$d$ prefix and returns $L$ reconstructions, exactly as in the single-taxon variant.

**Key hyperparameters.** `n_hierarchies` $K$, `bottleneck_n_taxonomy_layers` $L$, `k_leaves`, `use_inter_hierarchy_gate`, `gate_k`, `topk_k_multiplier`, `use_batch_topk`, `temperature`, `depth_decay`, `dead_steps`.
