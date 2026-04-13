# TaxonSAE Training Log

All runs on **Pythia-160M-deduped, Layer 8, resid_post**, dataset: Skylion007/openwebtext.

Last updated: 2026-03-29

---

## 1. Ablation Sweeps (48k steps = 100M tokens)

Purpose: explore hyperparameter space before formal training.

### sweep_temp — Temperature sweep
| trainer | n_layers | kl | temp | loss_space | hier_dec | eval | finding |
|---------|----------|----|------|------------|----------|------|---------|
| 0 | 13 | 0.0 | 0.1 | normalized | no | core+sp+hier | |
| 1 | 13 | 0.0 | 0.5 | normalized | no | core+sp+hier | **best CE diff** |
| 2 | 13 | 0.0 | 1.0 | normalized | no | core+sp+hier | |
| 3 | 13 | 0.0 | 2.0 | normalized | no | core+sp+hier | |
| 4 | 13 | 0.0 | 5.0 | normalized | no | core+sp+hier | |

**Conclusion**: temp=0.5 optimal (lowest CE diff 2.39, highest CosSim 0.875).

### sweep_kl — KL coefficient sweep
| trainer | n_layers | kl | temp | loss_space | hier_dec | eval | finding |
|---------|----------|----|------|------------|----------|------|---------|
| 0 | 13 | 0.0 | 1.0 | normalized | no | core+sp+hier | |
| 1 | 13 | 0.001 | 1.0 | normalized | no | core+sp+hier | **best CE diff** |
| 2 | 13 | 0.01 | 1.0 | normalized | no | core+sp+hier | |
| 3 | 13 | 0.1 | 1.0 | normalized | no | core+sp+hier | harmful |
| 4 | 13 | 1.0 | 1.0 | normalized | no | core+sp+hier | harmful |

**Conclusion**: kl=0.001 marginally best. kl≥0.1 degrades quality.

---

## 2. Phase 1 Formal Training (244k steps = 500M tokens)

Purpose: full SAEBench-standard training for baseline comparison. Config informed by ablation: temp=0.5.

### pythia160m_L8_4k — 4k width (n=11, dict_size=4094, L0=11)
| trainer | kl | CE diff | EV | MSE | CosSim | eval |
|---------|------|---------|------|---------|--------|------|
| 0 | 0.0 | 2.2213 | 0.9050 | 0.0638 | 0.8810 | core+sp |
| 1 | 0.001 | 2.2435 | 0.9018 | 0.0659 | 0.8803 | core+sp |
| 2 | 0.01 | 2.2542 | 0.9044 | 0.0642 | 0.8802 | core+sp |

### pythia160m_L8_16k — 16k width (n=13, dict_size=16382, L0=13)
| trainer | kl | CE diff | EV | MSE | CosSim | eval |
|---------|------|---------|------|---------|--------|------|
| 0 | 0.0 | 2.0787 | 0.9086 | 0.0613 | 0.8859 | core |
| 1 | 0.001 | 2.0628 | 0.9089 | 0.0612 | 0.8864 | core |
| 2 | 0.01 | 2.0959 | 0.9079 | 0.0618 | 0.8852 | core |

**Key finding**: CE diff ~2.0-2.2, far worse than baseline (~0.3-0.5 at L0≈19). Root cause: normalized-space loss + L0 information bottleneck. See `docs/eval_report_phase1.md`.

---

## 3. Scheme A+E Validation (original-space loss + hierarchical decoder)

Purpose: test whether switching loss to original space and adding per-depth decoders improves CE preservation.

### pythia160m_L8_4k_v2_quick — Quick validation (48k steps)
| trainer | n_layers | kl | temp | loss_space | hier_dec | CE diff | CosSim |
|---------|----------|----|------|------------|----------|---------|--------|
| 0 | 11 | 0.001 | 0.5 | original | yes | 2.5682 | 0.8674 |

### pythia160m_L8_4k_v2_full — Fair comparison (244k steps)
| trainer | n_layers | kl | temp | loss_space | hier_dec | CE diff | CosSim |
|---------|----------|----|------|------------|----------|---------|--------|
| 0 | 11 | 0.001 | 0.5 | original | yes | 2.2498 | 0.8792 |

**vs V1 (pythia160m_L8_4k trainer_1)**: CE diff 2.25 vs 2.24 — no improvement.

**Diagnostic findings** (diagnose_improvements.py):
- Scheme A does reduce high-norm token error: corr(norm², orig_mse) = -0.12 (V2) vs +0.65 (V1)
- But total MSE unchanged — capacity redistributed, not increased
- Core bottleneck is L0=11 information capacity, not loss space

---

## 4. Depth Sweep (48k steps, in progress)

Purpose: understand how CE diff scales with tree depth (L0). Find what depth matches TopK 4k k=20 (CE diff ≈ 0.49).

Unified config: temp=0.5, kl=0.001, normalized loss, no hier_dec.

| directory | n_layers | L0 | dict_size | params (enc) | status | CE diff | CosSim |
|-----------|----------|----|-----------|-------------|--------|---------|--------|
| depth_sweep_n11 | 11 | 11 | 4,094 | ~3M | ✅ | 2.4049 | 0.8729 |
| depth_sweep_n13 | 13 | 13 | 16,382 | ~13M | ✅ | 2.5313 | 0.8718 |
| depth_sweep_n15 | 15 | 15 | 65,534 | ~50M | ✅ | 2.5235 | 0.8696 |
| depth_sweep_n17 | 17 | 17 | 262,142 | ~200M | ✅ | 2.5304 | 0.8706 |
| ~~depth_sweep_n19~~ | ~~19~~ | ~~19~~ | ~~2,097,150~~ | ~~~800M~~ | ✗ OOM (>48GB) | | |

Note: TopK 4k k=20 has dict_size=4096, ~6M params, CE diff=0.49. At n=19 (L0=19, matching k=20), TaxonSAE would have 2M features and ~3.2B params — 500x more. This demonstrates the fundamental L0-dict_size coupling problem of binary tree routing.

---

## 5. Skip-TaxonSAE (low-rank skip connection)

Purpose: add a rank-8 dense skip connection to offload high-freq background info from the tree.

Based on Phase 1-2 diagnosis: baseline SAEs use ~5-6/20 L0 for high-freq features (rank≈2), TaxonSAE's shallow layers are similarly captured by norm-based routing.

| directory | skip_rank | steps | loss_space | init | status | CE diff | CosSim | notes |
|-----------|-----------|-------|------------|------|--------|---------|--------|-------|
| ~~skip_taxon_r8_4k_quick~~ | 8 | 48k | normalized | **zero** | ✗ deleted | 2.59 | 0.867 | skip dead (zero init gradient lock) |
| ~~skip_taxon_r8_4k_v2~~ | 8 | 48k | normalized | kaiming×0.01 | ✗ deleted | 2.30* | 0.878* | *after adapter fix; MSE plateaued at 50% |
| skip_taxon_r8_4k_full | 8 | 244k | normalized | kaiming×0.01 | ✅ | 1.9416 | 0.8903 | **+13.5% CE diff improvement vs V1** |

**Key fixes applied:**
- Zero init → kaiming×0.01 (broke gradient symmetry dead-lock)
- SAEBench adapter: cache skip_recon in encode(), add it back in decode()

**Skip diagnostic results** (skip_taxon_r8_4k_full, 500M tokens):
- skip_frac = 0.41 (skip handles 41% of reconstruction energy, close to Phase 1 prediction of ~43%)
- corr(norm², mse) reduced from +0.645 (V1) to +0.313 (skip) — partial fix
- Depth-0 corr(route, norm) reduced from -0.16 (V1) to +0.02 (skip) — routing no longer norm-driven
- 71% of leaf tokens follow norm-flat semantic paths; 18% follow norm-cascading paths
- Some genuinely semantic leaf nodes found (be-verbs, action verbs, time words, person nouns)

---

## 6. Soft Routing Sweep (48k steps)

Purpose: test whether soft routing (hard=False) can increase effective L0 to match baselines.

Config: n=11, kl=0.001, normalized loss, no skip.

| directory | temp | L0 | CE diff | CosSim | vs BatchTopK at similar L0 |
|-----------|------|-----|---------|--------|---------------------------|
| soft_temp0.0001_4k | 0.0001 | 12.5 | 2.077 | 0.885 | — |
| soft_temp0.0003_4k | 0.0003 | 16.5 | 2.171 | 0.887 | BatchTopK L0=19: CE=0.489 (**4.4x gap**) |
| soft_temp0.001_4k | 0.001 | 171 | 1.437 | 0.905 | BatchTopK L0=156: CE=0.106 (**13x gap**) |
| soft_temp0.003_4k | 0.003 | 426 | 0.811 | 0.941 | BatchTopK L0=313: CE=0.041 (**20x gap**) |
| soft_temp0.3_4k | 0.3 | 3619 | 0.385 | 0.963 | (near-dense, not comparable) |
| soft_temp0.5_4k | 0.5 | 3991 | 0.371 | 0.964 | (near-dense, not comparable) |

**Conclusion: soft routing is ineffective.** See `docs/soft_routing_analysis.md` for full analysis.
- Hierarchical probability multiplication creates only two stable regimes: near-hard (L0≈11) or near-dense (L0≈3000+)
- At matched L0, TaxonSAE is 4-20x worse than baselines, gap widens with L0
- Extra "soft" L0 is inflated — top-11 features carry 86-100% of latent energy

---

## 7. Multi-leaf TaxonSAE (leaf-level top-k)

Purpose: increase tree's combinatorial space by selecting top-k leaves instead of 1.

### Step 1: Multi-leaf k=20 (no skip, kl=0.001, 48k steps)

| directory | k_leaves | kl | CE diff | CosSim | EV | L0 (total) | L0 (leaf) |
|-----------|----------|----|---------|--------|------|-----------|-----------|
| multileaf_k20_4k | 20 | 0.001 | **0.9168** | 0.9309 | 0.9409 | 148 | 20 |

**vs baselines:**
- vs single-leaf k=1 (CE diff=2.40): **+62% improvement**
- vs BatchTopK k=20 (L0=19, CE diff=0.49): 1.9x gap, but L0 not directly comparable
- Effective leaf L0=20 matches baseline's k=20; ancestor L0≈128 is structural overhead

**Dead leaf problem:** 1632/2048 (80%) leaves are dead (never selected).
- Cause: per-token top-k is winner-take-all; unlike single-leaf pairwise routing which forces 50/50 splits
- Only ~200 leaves survive, but each token selects 20 from this pool → still much better than single-leaf
- KL coeff=0.001 is too weak to prevent concentration; larger KL may help (future experiment)

### Step 2: + Skip r=8 (k=20, kl=0.001, 48k steps)

| directory | CE diff | CosSim | EV | L0 (total) | L0 (leaf) |
|-----------|---------|--------|------|-----------|-----------|
| multileaf_k20_skip8_4k | **0.8518** | 0.9320 | 0.9433 | 87 | 20 |

**vs Step 1**: +7.1% CE diff improvement (0.917→0.852)

**Skip solved the dead leaf problem:**
- Dead leaves: 1632 (Step 1) → **0** (Step 2)
- Top leaf freq: 86% (Step 1) → **6.8%** (Step 2) — near-uniform
- Shallow activation: d0 1.9→1.3, d3 8.8→3.5 — skip absorbs generic reconstruction
- Total L0: 148→87 — fewer ancestors needed

**Leaf semantic quality good:** negation contractions (don't/isn't/wasn't), quotation markers, time words, connective punctuation. More semantic clusters than skip-only single-leaf.

### Step 3: + Matryoshka λ=0.1 (k=20, skip=8, kl=0.001, 48k steps)

| directory | CE diff | CosSim | EV | L0 (total) | L0 (leaf) |
|-----------|---------|--------|------|-----------|-----------|
| multileaf_k20_skip8_matry_4k | **0.7006** | 0.9420 | 0.9518 | 110 | 20 |

**vs Step 2**: +18% CE diff improvement (0.852→0.701)

**Matryoshka improved numerics but degraded leaf quality:**
- Shallow layer d0 incr_norm doubled (0.068→0.136) — forced to contribute more
- Mid-depth prefix MSE improved 16-22%
- But top-5 leaf freq jumped to **89%** — near-monopoly
- High-freq leaves encode "low-norm tokens" (no semantic content), orthogonal to skip direction
- Gradient analysis: leaf gradients decreased 31% due to lower overall loss, but no evidence of direct suppression by prefix loss

**Trade-off:** CE diff 0.70 (best so far) but leaf semantic structure lost. Step 2 (CE diff 0.85) has better leaf quality.

### Summary of progression

| Step | Config | CE diff | vs BatchTopK k=20 | Leaf quality |
|------|--------|---------|-------------------|-------------|
| 0 | single-leaf | 2.405 | 4.9x | norm-driven routing |
| 1 | + multi-leaf k=20 | 0.917 | 1.9x | 80% dead leaves |
| 2 | + skip r=8 | 0.852 | 1.7x | **uniform, semantic** |
| 3 | + matryoshka λ=0.1 | 0.701 | 1.4x | concentrated, generic |

All runs are 48k steps (100M tokens). Full 244k (500M tokens) training pending.

---

## 8. Skip Rank Sweep (multi-leaf k=20, 48k steps)

Purpose: find skip rank saturation point. Does higher rank skip further close the gap to baselines? At what point does tree degrade?

Config: n=11, k_leaves=20, temp=0.5, kl=0.001, loss_space=normalized, 48k steps.

### Core metrics

| directory | skip_rank | CE diff | CosSim | EV | L0 | vs BTK k=20 (0.550) |
|-----------|-----------|---------|--------|------|------|---------------------|
| multileaf_k20_skip8_4k | 8 | 0.852 | 0.932 | 0.943 | 87.5 | 1.55x |
| skip_rank_sweep_r16_4k | 16 | 0.909 | 0.933 | 0.944 | 80.4 | 1.65x (worse) |
| skip_rank_sweep_r32_4k | 32 | 0.624 | 0.945 | 0.954 | 98.5 | 1.13x |
| skip_rank_sweep_r48_4k | 48 | 0.632 | 0.946 | 0.955 | 86.7 | 1.15x |
| skip_rank_sweep_r64_4k | 64 | 0.553 | 0.949 | 0.957 | 83.2 | 1.01x |
| skip_rank_sweep_r96_4k | 96 | 0.461 | 0.955 | 0.962 | 163.6 | 0.84x (beats) |
| skip_rank_sweep_r128_4k | 128 | 0.387 | 0.960 | 0.966 | 89.3 | 0.70x (beats) |
| skip_rank_sweep_r192_4k | 192 | 0.287 | 0.968 | 0.971 | 164.3 | 0.52x (beats) |

### Skip-tree separation diagnostics

| skip_rank | skip_frac | eff_rank | cos(skip,tree) | subspace overlap | corr(norm²,mse) |
|-----------|-----------|----------|----------------|-----------------|-----------------|
| 8 | 0.483 | 7.5 | -0.044 | 0.003 | +0.002 |
| 16 | 0.540 | 15.0 | -0.074 | 0.006 | -0.002 |
| 32 | 0.602 | 31.0 | -0.040 | 0.006 | +0.003 |
| 48 | 0.637 | 46.8 | -0.028 | 0.002 | +0.002 |
| 64 | 0.665 | 62.6 | -0.024 | 0.002 | +0.003 |
| 96 | 0.776 | 85.1 | -0.010 | — | — |
| 128 | 0.751 | 124.2 | -0.017 | — | — |
| 192 | 0.832 | 177.3 | -0.033 | — | — |

### Leaf health

| skip_rank | dead leaves | Gini | top1 freq | top5 freq | top20 freq |
|-----------|------------|------|-----------|-----------|------------|
| 8 | 0 | 0.450 | 0.3% | 1.6% | 5.2% |
| 16 | 0 | 0.644 | 0.4% | 1.9% | 6.3% |
| 32 | 2 | 0.519 | 4.8% | 19.3% | 23.1% |
| 48 | 0 | 0.426 | 0.3% | 1.3% | 4.5% |
| 64 | 0 | **0.334** | 0.2% | 1.0% | 3.5% |
| 96 | **1619** | 0.981 | — | — | — |
| 128 | 0 | **0.322** | — | — | — |
| 192 | **1618** | 0.982 | — | — | — |

### Key findings

1. **CE diff monotonically improves** with rank: 0.852 → 0.287. r=64 matches BTK k=20; r=192 beats TopK 16k k=20 (0.348).
2. **Skip uses all capacity**: eff_rank ≈ skip_rank everywhere, skip_frac unsaturated at r=192 (0.832).
3. **Skip and tree naturally orthogonal**: cos(skip,tree) ≈ 0, subspace overlap < 1%. No explicit orthogonality constraint needed.
4. **Dead leaf problem returns at high rank**: r=96 and r=192 have 79% dead leaves (same pattern as no-skip multi-leaf). Skip absorbs too much → tree gradient weakens → leaf collapse.
5. **r=64 is the reliable optimum**: CE diff competitive (0.553), Gini lowest (0.334), zero dead leaves, stable.
6. **r=128 anomalous**: dead=0 despite being between two dead-leaf runs. May be training stochasticity — needs replication.
7. **r=16 worse than r=8**: insufficient skip capacity actively harms tree.

### Tree contribution (hierarchy eval, pruning curve)

| skip_rank | tree EV (full depth) | tree CosSim (full) |
|-----------|---------------------|-------------------|
| 8 | 0.285 | 0.745 |
| 16 | 0.250 | 0.729 |
| 32 | 0.236 | 0.726 |
| 48 | 0.218 | 0.715 |
| 64 | 0.201 | 0.709 |

Tree explained variance decreases with rank (0.285 → 0.201), confirming skip absorbs info that tree previously handled. But overall CE improves because skip handles it better.

### Depth semantics analysis (token entropy per depth)

Metric: per-node normalized token entropy (0=pure/monosemantic, 1=uniform/no selectivity).

| depth | r=8 | r=16 | r=32 | r=48 | r=64 | r=96 | r=128 | r=192 |
|-------|-----|------|------|------|------|------|-------|-------|
| 0 | 0.758 | 0.774 | 0.752 | 0.756 | 0.757 | 0.746 | 0.755 | 0.746 |
| 5 | 0.819 | 0.842 | 0.818 | 0.816 | 0.818 | 0.794 | 0.812 | 0.811 |
| 10 (leaf) | 0.902 | 0.921 | 0.906 | 0.901 | 0.898 | 0.924 | 0.897 | 0.930 |

**Findings:**
1. **No semantic specialization at any depth.** Entropy 0.75-0.93 everywhere — nodes activate on random high-freq tokens (the, of, ., ,).
2. **Entropy increases with depth (0.75 → 0.90)** — opposite of expected coarse-to-fine. Deeper nodes are MORE uniform, not more specific. Multi-leaf top-k disperses activations rather than concentrating them.
3. **Skip rank has negligible effect on tree semantics.** All healthy ranks (8-128) show nearly identical entropy profiles. CE diff improves but tree interpretability does not.
4. **r=96/192 dead leaf pattern:** only 212-222/2048 leaves active, surviving leaves carry all tokens (entropy ≈ 0.76 but only because few leaves absorb everything).

**Implication:** pure skip rank scaling improves reconstruction but does NOT improve tree interpretability. A directional inductive bias (e.g. tree cosine loss) is needed to push tree toward semantic features.

### Context-window analysis: TaxonSAE vs baselines (2026-03-29)

To rule out analysis methodology issues, ran identical context-window analysis (top-activating tokens ±5 token context, 200 sequences) on both TaxonSAE leaf nodes and baseline BatchTopK features.

**Methodology**: for each feature/leaf, collect top-20 activating positions and their surrounding context. Inspect whether target tokens form semantic clusters (same concept, same POS, same syntactic role). Also computed content-token entropy (stop words filtered).

**BatchTopK k=20 (4k dict) — clear semantic features found:**
- feat 202 (acts=5): "Access, access, entry, entry, mission" — **access/entry concept**
- feat 205 (acts=5): "pain, Pain, ache, joy, joy" — **emotion/sensation words**
- feat 225 (acts=202): "Jama, Hong, Hong, Mong, Scandin, Maced, Somal, Lib" — **country/region names**
- feat 1673 (acts=201): all begin-of-sequence — **paragraph-initial tokens**
- feat 155 (acts=5): "-, –, ~, -, -" — **dash/hyphen punctuation**

**TaxonSAE (all variants) — very few semantic features:**
- ml_skip64 leaf 1870 (acts=19): "Clinton, Trump, Trump, Clinton, Obama" — **political figures** (rare exception)
- ml_skip64 leaf 1834 (acts=25): "yet, yet, later, yet, eventually, later" — **temporal adverbs** (rare exception)
- ml_skip64 leaf 842 (acts=501): "22, 24, 30, 18, 13" — **date numbers**
- Vast majority of leaves show random high-frequency token mixtures with no discernible pattern

**Content entropy comparison** (stop words filtered, 0=pure, 1=uniform):
- All TaxonSAE variants: content entropy > 0.97, zero leaves below 0.85 threshold
- BatchTopK: not computed, but qualitative inspection shows much clearer clustering

**Conclusion: the lack of semantic features is a real TaxonSAE problem, not an analysis artifact.** BatchTopK's flat features naturally learn semantic directions via decoder columns. TaxonSAE's tree routing does not produce equivalent semantic specialization — the binary tree partitions tokens by reconstruction-optimal directions (dominated by activation norm/variance), not by semantic category.

**Correction to earlier findings:** Training log sections 5 and 7 reported "genuinely semantic leaf nodes (be-verbs, negation contractions, time words)." Systematic analysis across all 2048 leaves with context windows could not reproduce these at scale. The earlier claims likely reflect cherry-picked examples from a handful of leaves in a sea of non-semantic nodes, or were based on different analysis methodology not preserved.

### Root cause analysis: why TaxonSAE lacks semantic features (2026-03-29)

**Diagnostic methodology:** compared routing-decoder alignment between TaxonSAE and BatchTopK. For each model, measured: (1) cosine between active feature's decoder direction and the token activating it, (2) overlap between actually-selected features and "oracle" features (those with highest decoder cosine to the token).

**Routing-decoder alignment:**

| Model | Selected-decoder cos | Routing-decoder lift | Enc-dec cos |
|---|---|---|---|
| BatchTopK k=20 | 6.09 (raw) | 0.141 | 0.724 |
| single_leaf_skip8 | 0.119 (norm) | 0.111 | 0.895 |
| ml_k20_skip8 | 0.057 | 0.045 | 0.798 |
| ml_k20_skip64 | 0.037 | 0.028 | 0.850 |

**Feature selection vs oracle overlap:**

| Model | Selected ∩ Oracle / k | Oracle/Selected ratio |
|---|---|---|
| BatchTopK k=20 | **50% (10/20)** | 1.41x |
| TaxonSAE ml_k20_skip64 | 14% (2.8/20) | 2.45x |

**Findings:**

1. **Encoder-decoder alignment is NOT the problem.** TaxonSAE enc-dec cos (0.80-0.90) is actually higher than BatchTopK (0.72). Each node's encoder and decoder point in the same direction.

2. **Cascaded routing is the problem.** The multi-leaf top-k selects leaves via cascaded pairwise softmax probability (product of 11 layers). Depth-0 splits tokens by norm/variance (the MSE-gradient-dominant direction), and this split propagates down — a token entering the left subtree can only reach left-half leaves, regardless of whether those leaves' decoder directions match the token.

3. **Skip connection exacerbates the problem.** Higher skip rank → tree handles less reconstruction energy → weaker gradients to routing → routing-decoder alignment degrades further (lift: 0.111 → 0.045 → 0.028).

4. **Chicken-and-egg:** bad routing → decoder columns don't specialize → even oracle selection doesn't help much (TaxonSAE oracle cos = 0.082 vs BatchTopK = 8.55 raw).

**Root cause chain:**
```
MSE loss gradient ∝ activation magnitude
  → depth-0 pairwise softmax splits by norm/variance (highest gradient direction)
  → multiplicative probability chain propagates this split to all depths
  → leaf assignment determined by ancestor routing, not decoder direction alignment
  → selected leaves' decoder directions uncorrelated with token (overlap 14% vs 50%)
  → decoder columns receive poor gradients, can't specialize semantically
  → vicious cycle: bad routing ↔ bad decoder directions
```

**Contrast with BatchTopK:**
```
feature selection = topk(W_enc @ x + b) — flat, per-feature competition
  → selected feature ↔ high encoder projection ↔ aligned with decoder (tied init)
  → decoder direction gets gradient proportional to how well it reconstructs
  → virtuous cycle: good direction → more selection → better direction
```

**Scripts:** `scripts/diagnose_routing.py`, `scripts/analyze_leaf_contexts.py`, `scripts/analyze_baseline_contexts.py`, `scripts/analyze_leaf_semantics.py`

---

## 9. Activation Decomposition Pre-analysis (2026-03-29)

Purpose: determine whether removing positional/contextual components from activations before SAE training could improve feature quality. Based on Song & Zhong (ICLR 2024) four-way decomposition.

### Exp 0a: Four-way decomposition

`h_{c,t} = μ + pos_t + ctx_c + resid_{c,t}` on 2000 sequences × 128 tokens.

| Component | Variance fraction | Effective rank | Notes |
|---|---|---|---|
| pos_t | **29.2%** | **1** | Single direction, magnitude varies with position |
| ctx_c | 2.2% | 9 (50%), 120 (90%) | Small but real; continuous, no discrete clusters |
| resid_{c,t} | 68.5% | — | Token-level semantic signal |

pos-ctx orthogonality: |cos| mean = 0.105, 57% pairs < 0.1 → near-orthogonal.

### Exp 0b: Skip connection alignment

Skip (rank=64) alignment with each component:

| Component | |cos| | R² |
|---|---|---|
| pos | 0.274 | 9.2% |
| ctx | 0.326 | 14.7% |
| **resid** | **0.507** | **39.3%** |

**Skip primarily fits resid (39%), not pos (9%).** Skip acts as a "dense SAE" compensating for tree weakness, not as an artifact remover. This explains why increasing skip rank improved CE diff without improving tree semantics — skip and tree compete for the same signal.

### Exp 0d: Temporal autocorrelation (ctx_len=1024)

| δ | h_clean | resid | ctx contribution |
|---|---|---|---|
| 1 | 0.298 | 0.255 | 0.044 |
| 32 | 0.090 | 0.032 | 0.058 |
| 128 | 0.071 | 0.007 | 0.064 |
| 256 | 0.064 | ~0 | 0.065 |
| 512 | 0.055 | -0.008 | 0.063 |
| 768 | 0.050 | -0.011 | 0.061 |

- resid decorrelates by δ~128 (token-level signal range ~100 tokens)
- ctx contribution stable at ~0.06 from δ=64 to δ=768 → document-level, near-constant
- Song & Zhong's sequence mean is a valid ctx estimate at ~1000 token scale
- Initial analysis with ctx_len=128 showed artificial dropoff at δ=127 due to sequence boundary truncation

### Key implications

1. **pos is the #1 denoising target**: 29% variance, rank-1, one line of code to remove
2. **ctx is small (2.2%) but real**: stable autocorrelation confirms it exists as a document-level signal
3. **Current skip connection is misallocated**: it mostly learns resid (the signal SAE should handle), not pos/ctx (the artifacts that should be removed)

Full analysis and plan: see `docs/activation_decomposition_plan.md`.

### Exp 0c: BatchTopK on depos activations

Trained BatchTopK k=20 (4k dict) on `h - μ - pos` (pos estimated at ctx_len=1024). Compared against original BatchTopK baseline in original activation space.

**Reconstruction quality (original space):**

| Metric | Original BTK (244k) | Depos BTK (48k) | Depos BTK (244k) |
|---|---|---|---|
| MSE | 62.22 | 56.69 | **54.87** |
| CosSim | 0.942 | 0.948 | **0.949** |
| EV | 0.954 | 0.958 | **0.959** |
| L0 | 19.2 | 18.7 | 18.7 |
| Mean coherence | 0.411 | 0.432 | **0.434** |
| >=0.5 coherence | 23.9% | 28.3% | 27.8% |
| >=0.7 coherence | 9.9% | 12.9% | **13.5%** |

Depos improves all metrics: MSE -12%, coherent features (>=0.7) +36%. Full training (244k) adds marginal improvement over 48k — most gains are immediate.

**Note on ctx_len sensitivity:** pos estimated at ctx_len=128 vs 1024 gives cos≈0.002 between estimates, and pos variance fraction drops from 29% (ctx128) to 11% (ctx1024). However, SAE performance is similar between both estimates. Must use estimates matching training ctx_len.

---

## 10. Embedding Coherence Analysis (2026-03-30)

### New metric: embedding coherence

Content-token entropy cannot detect type-level semantics (e.g., "fires on plural nouns" vs "fires on a random mix"). Replaced with **embedding coherence**: for each feature, collect token IDs of activating tokens, look up input embeddings, compute mean cosine to centroid.

Scale:
- 1.0 = always same token (perfect monosemantic)
- 0.5–0.7 = coherent lexical-semantic category (plural nouns, time units, pronouns)
- 0.3–0.5 = weak lexical pattern or syntactic/positional feature
- < 0.2 = no coherent pattern (distributed residual correction)

**Important caveat:** Embedding coherence measures similarity in **input embedding space**, which encodes lexical semantics but NOT syntax/position. Therefore:
- **Overvalues** token-identity features ("always fires on 'rights'") → high coherence
- **Undervalues** syntactic features ("fires on sentence-initial position" → diverse tokens → low coherence despite being perfectly interpretable)
- **Higher coherence ≠ necessarily better interpretability.** A model with more syntactic/positional features will score lower but may be equally interpretable.
- Differences between architectures may reflect **feature type distribution** (more token-specific vs more syntactic) rather than quality.

Validated on BatchTopK: mid-coherence (0.5) features include time units (week, month, day), quantity words (budget, amount, price), pronouns (her, you, me, us). High-activation features with low coherence often encode legitimate syntactic/structural patterns (sentence boundaries, clause positions, subword positions in multi-token words).

**Script:** `eval/embedding_coherence.py`

### Embedding coherence comparison across architectures

| Model | CE diff | Mean coh | >=0.5 | >=0.7 | Active feats |
|---|---|---|---|---|---|
| BatchTopK k=20 | 0.550 | 0.411 | 23.8% | 9.9% | ~4000 |
| single_leaf_skip8 | 1.942 | 0.423 | 28.5% | 8.1% | 1949 |
| ml_k20_skip8 | 0.852 | 0.291 | 2.1% | 0.1% | 4087 |
| ml_k20_skip64 | 0.553 | 0.340 | 13.1% | 0.8% | 4092 |
| depos BTK (244k) | ~0.55 | 0.434 | 27.8% | 13.5% | ~3600 |

### Multi-leaf top-k effect on features

**Multi-leaf top-k dramatically changes feature characteristics.** Single-leaf (k=1) and multi-leaf k=20 have very different coherence profiles:

- **single_leaf_skip8**: higher coherence (0.423), fewer active features (1949) — features are more token-specific
- **ml_k20_skip8**: lower coherence (0.291), more active features (4087) — features may include more syntactic/positional types that score low on embedding coherence

However, manual inspection confirmed the difference is not just metric bias: ml_k20's high-activation features show genuinely random token mixtures, while single-leaf's top features show interpretable patterns (be-verbs, time words, possessives, degree modifiers).

### Root cause: cascaded probability × top-k interaction

The problem is **multi-leaf top-k selection on cascaded probabilities**:

1. **Single-leaf**: each token takes exactly 1 path via pairwise hard routing → forced 50/50 splits → all 2048 leaves get traffic → each leaf's decoder direction optimized for its specific token cluster
2. **Multi-leaf top-k**: top-k leaves selected by cascaded probability product → selection dynamics depend strongly on k:
   - k=2: extreme winner-take-all (eff_leaves=159, 89% dead). Surviving leaves highly specialized → high coherence but poor dictionary utilization
   - k=20: probability distribution near-uniform (eff_leaves=2028). Top-20 selection nearly random → low coherence
   - k=1 (pairwise routing): eff_leaves=1963. Balanced via forced binary splits, not top-k

### k-leaves × kl_coeff × skip sweep (48k steps)

| Config | CE diff (norm) | CosSim | L0 | Mean coh | >=0.5 | >=0.7 | Active feats |
|---|---|---|---|---|---|---|---|
| k=1 skip8 kl.001 | 0.209 | 0.887 | 11 | 0.418 | 26.2% | 8.1% | 1494 |
| k=2 skip8 kl.001 | 0.218 | 0.883 | 21 | 0.478 | 40.9% | 18.2% | 516 |
| **k=2 skip8 kl.01** | **0.207** | **0.889** | **21** | **0.508** | **41.6%** | **21.7%** | 488 |
| k=2 skip8 kl.1 | 0.191 | 0.898 | 20 | 0.446 | 34.9% | 12.7% | 981 |
| k=3 skip8 kl.001 | 0.200 | 0.893 | 31 | 0.460 | 37.9% | 12.1% | 601 |
| k=3 skip8 kl.01 | 0.200 | 0.894 | 31 | 0.425 | 30.1% | 8.5% | 468 |
| k=3 skip8 kl.1 | 0.183 | 0.903 | 29 | 0.395 | 22.8% | 5.9% | 1090 |
| k=3 skip0 kl.001 | 0.222 | 0.882 | 31 | 0.399 | 22.6% | 6.3% | 735 |
| k=3 skip0 kl.01 | 0.219 | 0.883 | 30 | 0.433 | 31.3% | 9.2% | 683 |
| k=5 skip8 kl.001 | 0.188 | 0.904 | 49 | 0.444 | 34.4% | 8.4% | 521 |
| k=5 skip8 kl.01 | 0.183 | 0.904 | 49 | 0.426 | 26.8% | 10.0% | 559 |
| k=5 skip8 kl.1 | 0.175 | 0.907 | 46 | 0.393 | 22.6% | 8.1% | 1033 |
| k=20 skip8 kl.001 | 0.130 | 0.932 | 87 | 0.294 | 2.3% | 0.1% | 4068 |

SAEBench CE diff (original space): k=2 skip8 kl.01 = **1.93** vs BatchTopK k=20 = **0.55** (3.5x gap at matched L0≈20).

**Key findings:**

1. **k=2 kl=0.01 has highest coherence** (0.508) but only 488/4094 features alive (51%) and CE diff 3.5x worse than BatchTopK. The high coherence is partly because surviving features are highly token-specific (winner-take-all selects only the most distinctive leaves).

2. **KL effect depends on k**: k=2 has a sweet spot at kl=0.01. For k=3,5, higher kl hurts coherence — more uniform routing means less specialized leaves. kl=0.1 increases alive features (981 at k=2) but reduces coherence (0.446).

3. **Skip helps coherence**: k=3 skip8 vs skip0 at kl=0.001: 0.460 vs 0.399 (+15%). Skip absorbs generic reconstruction, but cannot fix dead leaves at small k.

4. **CE diff driven by L0**: at matched normalized MSE, more active nodes = better reconstruction. Tree structure imposes ~11 ancestor nodes as "structural overhead" per leaf path.

### Feature browser: dual-dimension classification (2026-03-30)

Added POS (Part-of-Speech) tagging as second dimension alongside embedding coherence. Uses NLTK POS tagger to classify activating tokens by grammatical role. Combined with coherence, this catches both lexical/semantic features (high coherence) AND syntactic features (low coherence but high POS purity).

**Important:** stop words are NOT filtered from POS analysis — words like will/can/who/they are critical POS signals (modal verbs, pronouns). Stop words only filtered for the "top content tokens" display.

**Script:** `eval/feature_browser.py` — outputs JSON + Markdown per checkpoint.

**Feature type distribution (ml_k20_skip64 vs BatchTopK):**

| Dimension | BatchTopK k=20 (2790) | TaxonSAE ml_k20_skip64 (2047) |
|---|---|---|
| **Semantic** (coh-based) | 768 (27.5%) | 287 (14.0%) |
| **POS-specific** | 311 (11.1%) | 133 (6.5%) |
| **Structural/punct** | 984 (35.3%) | 765 (37.4%) |
| **Noise** (distrib+unknown) | 727 (26.1%) | 862 (42.1%) |
| **Interpretable total** | **2033 (72.9%)** | **1185 (57.9%)** |

**Category breakdown:**

| Category | BatchTopK | TaxonSAE | Detection method |
|---|---|---|---|
| semantic-category | 531 (19.0%) | 264 (12.9%) | coh > 0.5 |
| lexical-specific | 218 (7.8%) | 23 (1.1%) | coh > 0.7 |
| syntactic-semantic | 532 (19.1%) | 268 (13.1%) | coh > 0.3, pos_purity > 0.35 |
| punctuation | 382 (13.7%) | 491 (24.0%) | top token is punct |
| pos:noun | 145 (5.2%) | 32 (1.6%) | pos_purity > 0.5, dominant=noun |
| pos:proper-noun | 127 (4.6%) | 70 (3.4%) | pos_purity > 0.5, dominant=proper-noun |
| pos:verb | 13 (0.5%) | 5 (0.2%) | pos_purity > 0.5, dominant=verb |
| distributed-residual | 29 (1.0%) | 78 (3.8%) | coh < 0.25, pos_purity < 0.35 |
| unknown | 698 (25.0%) | 784 (38.3%) | doesn't match any rule |

**Key observations:**

1. **BatchTopK is better across both dimensions**, but the gap is smaller than pure coherence suggested. POS analysis revealed ~6.5% of TaxonSAE features are syntactic (invisible to coherence alone).

2. **Biggest gap is in lexical-specific** (7.8% vs 1.1%): BatchTopK learns 9x more features that fire on specific tokens/token clusters. This reflects the routing-decoder alignment difference — BatchTopK's feature selection naturally produces token-specific features.

3. **Structural/punct features are comparable** (35% vs 37%). Both architectures encode similar amounts of punctuation and structural patterns.

4. **TaxonSAE has more noise** (42% vs 26%). The "unknown" category (38%) needs further investigation — these features have moderate coherence (0.31) and moderate POS purity (0.34) but don't clearly fit any category. May be mixed syntactic-semantic, or genuinely uninterpretable.

5. **Routing split analysis** (depth 0-5 left-vs-right comparison) shows tree routing does learn some coarse-to-fine grammatical structure: depth 0 splits reporting verbs vs state/relation words; depth 1-2 further separates modals vs past/passive vs entity references vs adjectives. However, this is grammatical hierarchy, not semantic hierarchy (dog → retriever → golden retriever).

### Summary and outlook

**What we know:**
- TaxonSAE's tree routing CAN produce coherent features (single-leaf and small-k multi-leaf)
- But coherent features ≠ good reconstruction — tree's structural overhead (ancestor nodes) makes it ~3.5x less efficient than flat SAE at matched L0
- Multi-leaf top-k (k≥5) trades coherence for reconstruction by making leaf selection near-random
- Dual-dimension analysis (coherence + POS) gives a more complete picture: TaxonSAE ~58% interpretable vs BatchTopK ~73%, with the gap mainly in lexical-specific and POS-noun features
- Tree routing learns grammatical hierarchy (reporting→modal→entity vs state→passive) not semantic hierarchy
- Depos (positional component removal) gives a modest but consistent improvement to any SAE

**Open questions:**
- Can decoder-aligned leaf selection close the reconstruction gap without losing feature quality?
- Is the 38% "unknown" category in TaxonSAE truly uninterpretable, or a classification threshold issue?
- Would the grammatical hierarchy learned by tree routing be useful as a post-hoc analysis tool even if reconstruction is done by a flat SAE?

---

## 11. ForestSAE — Shallow Forest (48k steps)

Purpose: replace single deep binary tree with a forest of shallow trees. Address the gradient bottleneck, combinatorial bottleneck, and routing-decoder misalignment found in Sections 1–10.

### Architecture

- N independent shallow trees, each with a root node + routing levels
- Depth 0 (root): flat top-k tree selection (identical to TopK/BatchTopK)
- Depths 1+: cascaded pairwise softmax within selected trees (only 1–5 levels, not 11+)
- k_trees=20 → L0=20 at every depth level
- Per-depth **independent** reconstruction (each depth reconstructs x alone, no prefix accumulation)
- Loss = mean of per-depth MSEs + KL(tree usage || uniform)

### Pythia-160M L8, leaf=4096 — n_trees sweep

Config: k_trees=20, temp=0.5, kl=0.001, normalized loss, no skip. 48k steps (100M tokens).

| directory | n_trees | depth | layers | leaf norm_MSE | leaf CosSim | orig_MSE |
|-----------|---------|-------|--------|--------------|-------------|----------|
| forest_t128 | 128 | 6 | 128/256/512/1024/2048/4096 | 0.1213 | 0.9369 | 69.48 |
| forest_t256 | 256 | 5 | 256/512/1024/2048/4096 | 0.1163 | 0.9398 | 66.30 |
| forest_48k | 512 | 4 | 512/1024/2048/4096 | 0.1153 | 0.9403 | 66.23 |
| forest_t1024 | 1024 | 3 | 1024/2048/4096 | 0.1095 | 0.9434 | 62.48 |
| forest_t2048 | 2048 | 2 | 2048/4096 | 0.1096 | 0.9434 | 62.61 |

**Baselines (244k steps):**
- BatchTopK 4k k=20: MSE=62.22, CosSim=0.942
- MatryoshkaBTK 4k k=20: MSE=70.03, CosSim=0.936

**Interpretability (Pythia, dual-dimension: coherence + POS):**

| n_trees | mean_coh | >=0.5 | >=0.7 | interp% |
|---------|----------|-------|-------|---------|
| 128 | 0.347 | 11.2% | 3.2% | 52.6% |
| 256 | 0.355 | 13.1% | 4.2% | 55.9% |
| 512 | 0.364 | 14.6% | 5.0% | 57.8% |
| 1024 | 0.377 | 17.7% | 5.7% | 59.5% |
| 2048 | 0.393 | 20.9% | 8.2% | 65.4% |

**Baseline**: BatchTopK mean_coh=0.411, >=0.5: 23.8%, interp%=72.9%.

**Key finding**: more trees + shallower depth = better reconstruction AND better interpretability simultaneously. No tradeoff. t1024/t2048 approach BatchTopK on both metrics despite 1/5 training steps.

### Tree health (Pythia, forest_48k = 512 trees)

- Dead trees: 4/512 (0.8%)
- Usage entropy: 0.93 (near-uniform)
- Top-1 tree frequency: 2.9%, top-20: 21.7%

All n_trees configurations showed healthy tree usage with near-zero dead trees.

### Gemma-2-2B L12, leaf=16384 — architecture comparison

Config: 48k steps (100M tokens). All use temp=0.5, kl=0.001, normalized loss.

| directory | architecture | n_trees | depth | L0 | CE diff | CosSim | EV | Alive |
|-----------|-------------|---------|-------|----|---------|--------|------|-------|
| gemma_forest_t512 | ForestSAE | 512 | 6 | 120 | 0.688 | 0.867 | 0.781 | 83.1% |
| gemma_forest_t1024 | ForestSAE | 1024 | 5 | 100 | 0.578 | 0.867 | 0.785 | 75.9% |
| gemma_taxon_ml14_skip128 | TaxonSAE multi-leaf k=20 skip128 | — | 14 | 143 | 0.438 | 0.895 | 0.820 | 100.0% |
| gemma_taxon_sl14 | TaxonSAE single-leaf | — | 14 | 14 | 3.797 | 0.746 | 0.617 | 28.2% |

**Baselines (244k steps, 16k dict):**
- BatchTopK k=20: L0=20.9, CE diff=0.281, CosSim=0.883, EV=0.629
- MatryoshkaBTK k=20: L0=20.3, CE diff=0.328, CosSim=0.875, EV=0.598
- BatchTopK k=80: L0=83.7, CE diff=0.109, CosSim=0.918, EV=0.734

### SAEBench absorption & splitting (Gemma-2-2B, 16k leaf)

| directory | Absorption | Full absorption | Splits |
|-----------|-----------|----------------|--------|
| gemma_forest_t1024 | 0.002 | 0.000 | 1.08 |
| gemma_taxon_ml14_skip128 | 0.000 | 0.000 | 1.15 |
| gemma_taxon_sl14 | 0.000 | 0.000 | 1.00 |

**Baselines (244k steps):**
- BatchTopK 16k k=20: absorption=0.326, splits=3.04
- MatryoshkaBTK 16k k=20: absorption=0.152, splits=2.00

**Concern**: All our models show near-zero absorption. This could mean (a) the tree/forest architecture genuinely avoids feature absorption, or (b) 48k steps is insufficient for the SAE to learn the fine-grained first-letter concepts that the absorption eval probes for. The absorption eval requires the SAE to have features aligned with first-letter directions — if the SAE hasn't learned those at all, there is nothing to absorb, and the metric reads zero regardless of architecture quality. A 244k-step full training run is needed to disambiguate.

### Summary

1. **ForestSAE works**: reconstruction competitive with BatchTopK at 1/5 training steps (Pythia t1024: CosSim 0.943 vs baseline 0.942).
2. **More trees = better**: monotonic improvement in both reconstruction and interpretability as n_trees increases (t128→t2048). The t2048 config (depth=2, one routing level) is closest to flat top-k and performs best.
3. **Deeper routing still hurts**: consistent with original TaxonSAE findings. ForestSAE t512 (5 routing levels) underperforms t2048 (1 routing level) on all metrics.
4. **TaxonSAE multi-leaf k=20 skip128 surprisingly strong on Gemma**: CE diff 0.438, CosSim 0.895 — better reconstruction than both ForestSAE variants. However, its L0=143 is much higher (structural ancestor overhead), making direct L0-matched comparison unfair.
5. **Single-leaf TaxonSAE n=14 confirmed catastrophic**: CE diff 3.8, 72% dead features. Deep single-path routing does not scale.
6. **Absorption results inconclusive at 48k steps**: all models show near-zero, likely due to insufficient training. Needs 244k-step replication.

### Open questions

- Would ForestSAE at 244k steps close the remaining gap to BatchTopK?
- Would absorption emerge with more training, and if so, would ForestSAE still have an advantage?
- Is the "more trees = better" trend a genuine hierarchy benefit, or does it simply converge to flat top-k with extra overhead?
- Can per-depth independent reconstruction learn meaningful coarse-to-fine hierarchy, or do features at all depths end up similar?

---

## Section 12: Per-depth Interpretability Deep Dive — Gemma-2-2B L12 (2026-04-01)

### Motivation

Section 11 left open the question of whether ForestSAE's per-depth independent reconstruction learns meaningful coarse-to-fine hierarchy. We now run per-depth interpretability analysis with context window sampling on the Gemma-2-2B L12 checkpoints to answer this directly.

### Method

Added `interp-depth` stage to `eval/analyze.py`:
- `collect_feature_data_sampled()`: two-pass approach — warmup 50 sequences to select 200 representative features per depth (half top-frequency, half random-alive), then full 200-sequence pass collecting context windows for selected features only. Reduces inner loop from O(32k × 200) to O(1k × 200), ~30× faster.
- `run_per_depth_interpretability()`: runs embedding coherence, POS distribution, and auto_classify independently per depth, with context window examples grouped by category.
- Also updated `analyze.py` to auto-detect model/layer from checkpoint config.json and handle Gemma bfloat16→float32 precision casting.

### Results: ForestSAE t1024 (1024 trees, depth 5)

**Per-depth summary (200 features sampled per depth):**

| Depth | Features | Mean Coh | Median Coh | Interpretable % | distributed-residual |
|-------|----------|----------|------------|-----------------|---------------------|
| 0 (root) | 1,024 | 0.468 | 0.420 | 75.0% | 1 |
| 1 | 2,048 | 0.461 | 0.419 | 75.0% | 0 |
| 2 | 4,096 | 0.468 | 0.430 | 70.0% | 4 |
| 3 | 8,192 | 0.449 | 0.404 | 70.5% | 11 |
| 4 (leaf) | 16,384 | 0.463 | 0.422 | 64.0% | 20 |

Key observation: interpretable fraction degrades from 75% (root) to 64% (leaf), driven by increasing distributed-residual features at deeper levels.

**Finding 1: No cross-depth semantic differentiation.**
The same feature "semantics" repeat at every depth. For example, a pure "the" feature appears at all 5 depths (feat 610→2244→5513→12051→25127) with near-identical context windows. Similarly for newline features, comma/period features, and boilerplate header detectors ("JERUSALEM / Draft / Summary / Ad / Posted"). Per-depth independent reconstruction causes each depth to independently learn the full feature set rather than specializing.

**Finding 2: Genuine semantic features exist but are rare (~5-8% of sampled features).**
Examples of good semantic features found across depths:
- Concessive connectives: feat 431 (d0) — `Despite/although/despite`, consistent transition contexts
- Secrecy/confidentiality: feat 56 (d0) — `private/secret/confidential`, information access contexts
- Country names: feat 727 (d0) — `China/Egypt/Mexican/Brazil/Somali`, international relations contexts
- Attempt verbs: feat 2742 (d1) — `attempted/try/trying`, consistent "effort" semantics
- Privacy/personal: feat 1461 (d1) — `individual/personal/private`, data privacy contexts
- Enumeration: feat 4195 (d2) — `including × 8`, list contexts
- News media: feat 13846 (d3) — `BBC/NBC/CBC/CNN`
- East Asia: feat 10153 (d3) — `Korean/Korea/Hong/Dragon/Samsung/Taiwan`

These represent ~5-8% of sampled features per depth. The rest are function word detectors, positional features, or noise.

**Finding 3: No coarse-to-fine hierarchy.**
If the architecture worked as intended, shallow depths should capture coarse categories ("politics") and deeper depths should specialize ("US immigration policy"). Instead:
- Depth 0 already has fine-grained features (e.g., `government × 6`)
- Depth 4 has MORE noise (20 distributed-residual vs depth 0's 1), not finer semantics
- Feature granularity is roughly uniform across depths

**Finding 4: Document memorization artifacts.**
Some features bind to specific documents rather than semantic concepts. E.g., feat 4165 (d2) — all 5 context windows come from the same "ethical conundrum" article, with tokens `points | the | ethical | underlying | undrum`. This indicates overfitting to specific training sequences rather than learning generalizable features.

**Finding 5: Auto-classifier noise.**
The `unknown` category (49/200 at d0) captures many features with coherence 0.4-0.5 that don't cleanly match any heuristic rule. The `pos:proper-noun` category contains false positives from capitalized sentence-initial words (`Giving | Allow | Claims`). These classification artifacts should be kept in mind when interpreting category distributions.

### Results: All four Gemma-2-2B L12 checkpoints compared

**Per-depth interpretable% trend (first depth → leaf depth):**

| Model | Depth trend (interp%) | Leaf interp% | Leaf mean coh |
|-------|-----------------------|-------------|---------------|
| forest_t1024 (5 depths) | 75→75→70→70→64 | 64.0% | 0.463 |
| forest_t512 (6 depths) | 74→78→74→68→68→66 | 66.0% | 0.466 |
| taxon_ml14_skip128 (14 depths) | 100→...→59→69→61→66→66→70→72→**77→74**→68 | 68.0% | **0.498** |
| taxon_sl14 (14 depths) | 100→...→62→61→51→63→62→60→58→55→54→52 | 51.5% | 0.458 |

**Distributed-residual (noise features) at leaf depth:**
- forest_t1024: 20/200
- forest_t512: 15/200
- taxon_ml14_skip128: 32/200
- taxon_sl14: **60/200** (catastrophic)

**Finding 6: TaxonSAE multi-leaf skip128 is the only model showing hierarchical structure.**
Coherence rises from 0.365 (d0) → 0.535 (d11) → 0.498 (d13 leaf). Interpretable% peaks at 77% (d11) before declining slightly at the leaf. Lexical-specific features increase from 0 (d0-d4) to 35 (d11). This is the coarse-to-fine progression the architecture was designed for — shallow depths capture syntactic/positional patterns, deeper depths develop token-specific features. The decline at d13 (leaf) suggests the final routing levels add noise without additional specialization.

**Finding 7: ForestSAE shows no hierarchical structure.**
Both forest_t1024 and forest_t512 have flat coherence profiles (0.43-0.47 across all depths) and flat interpretable% (64-78%). The same features (e.g., "the", newline, punctuation) repeat at every depth. Per-depth independent reconstruction makes each depth learn the entire feature set independently rather than specializing.

**Finding 8: Single-leaf TaxonSAE degrades monotonically with depth.**
Interpretable% drops from 100% (d0, 2 features) to 52% (d13 leaf), with distributed-residual growing from 0 to 60/200. The cascaded routing bottleneck (single path) causes progressive information loss through the tree.

**Finding 9: Skip connection enables hierarchy in deep trees.**
Comparing taxon_ml14_skip128 vs taxon_sl14 — same 14-depth binary tree, but skip128 peaks at 77% interpretable (d11) while sl14 never exceeds 63% after d2. The skip connection absorbs positional/norm variance, freeing the tree routing to develop semantic specialization at middle depths (d8-d12). However, multi-leaf k=20 also contributes by activating 20 paths instead of 1.

**Finding 10: Leaf-level feature quality comparison.**

| Model | Lexical | Semantic | Syntactic | Punct | unknown |
|-------|---------|----------|-----------|-------|---------|
| forest_t1024 | 14 | 33 | 28 | 19 | 52 |
| forest_t512 | 15 | 23 | 40 | 22 | 53 |
| taxon_ml14_skip128 | **24** | 15 | **39** | 20 | 32 |
| taxon_sl14 | 8 | 24 | 25 | 26 | 37 |

taxon_ml14 has the most lexical-specific (24) and syntactic-semantic (39) leaf features, fewest unknowns (32). ForestSAE variants have more semantic-category features but more unknowns. However, manual inspection reveals that "semantic-category" in ForestSAE often captures function words (the, to, of) rather than true semantic clusters.

### Revised assessment — after reading full context windows

**Important caveat**: The aggregate metrics (coherence means, category counts, interpretable%) used above are heuristic-based and unreliable. The findings below are based on manually reading all context windows in the JSON output. The category-based summary tables should be treated as rough guides, not ground truth.

**Finding 6 (revised): TaxonSAE ml14_skip128's "hierarchy" is weaker than summary metrics suggest.**
The coherence increase from d0→d11 (0.365→0.535) is real, but context window inspection reveals it is primarily driven by function word features splitting into more specialized variants at deeper depths (e.g., multiple `the/a/that/in/on` features with slightly different usage patterns). This is feature proliferation, not semantic refinement. A few genuinely good syntactic features emerge in middle depths — e.g., `who/which` relative pronouns (d10 feat 2189), `its` possessive (d10 feat 2462), `attempted/try/trying` effort verbs — but these are scattered across depths without a clear coarse-to-fine gradient. The d11 "peak" in interpretable% is substantially inflated by function word detectors being classified as lexical-specific (35 out of 200).

**Finding 7 (confirmed): ForestSAE shows no hierarchical structure.**
Context windows confirm: the same features (function words, punctuation, boilerplate headers like `JER/Draft/Summary/Ad/Posted`, code fragments from tokenizer examples) repeat at every depth with near-identical contexts. Forest_t512 and t1024 are nearly indistinguishable in feature quality. Some features memorize specific training documents (e.g., C++ tokenizer code, "ethical conundrum" article) rather than learning generalizable patterns.

**Finding 8 (confirmed): Single-leaf TaxonSAE degrades catastrophically.**
Context windows at d13 confirm distributed-residual features are pure noise. By d6, many features are URL fragments (`v4-460px`, `wikihow.com`) and non-English script artifacts. The single routing path causes progressive information loss.

**Cross-model finding: genuine semantic features are rare (~5-10%) in all four models.**
Examples that hold up under context window inspection:
- Concessive connectives: forest_t1024 d0 feat 431 — `Despite/although/despite`, consistent transition contexts
- Secrecy: forest_t1024 d0 feat 56 — `private/secret/confidential`
- Countries: forest_t1024 d0 feat 727 — `China/Egypt/Mexican/Brazil/Somali`
- News media: forest_t1024 d3 feat 13846 — `BBC/NBC/CBC/CNN`
- Relative pronouns: taxon_ml14 d10 feat 2189 — `who/which`, consistent relative clause contexts
- Effort verbs: taxon_ml14 d9 (also forest_t1024 d1) — `attempted/try/trying`
- Compound noun modifier: taxon_ml14 d0 feat 1 — `school/car/farm/law/paper` (as pre-nominal modifiers)

The remaining ~90% are: function word detectors (the/a/to/of/in), punctuation, positional features, boilerplate/URL memorization, and noise.

**Cross-model finding: all models heavily memorize specific training documents.**
Multiple features across all models bind to the same documents rather than semantic concepts — the "ethical conundrum" economics paper, the C++ tokenizer tutorial, the ForestSAE Gemma training data's OpenWebText sequences. This suggests 48k steps on 100M tokens is insufficient for features to generalize beyond the training distribution.

### Per-depth Reconstruction Analysis

Ran `--stage reconstruction` on all 4 Gemma-2-2B L12 checkpoints (50 batches × 2048 tokens = 102.4k tokens). Metrics computed cumulatively: starting from pre_bias (+ skip if applicable), each depth's decoder contribution is added in order, and mse_delta measures the MSE change from adding that depth.

**Aggregate metrics:**

| Model | Norm MSE | CosSim | Norm EV | L0 | Dead Leaves |
|-------|---------|--------|---------|-----|-------------|
| forest_t1024 (5d) | 0.2330 | 0.8749 | 0.7670 | 100.0 | 8110/16384 (49.5%) |
| forest_t512 (6d) | 0.2374 | 0.8723 | 0.7626 | 120.0 | 6471/16384 (39.5%) |
| taxon_ml14_skip128 (14d) | 0.1906 | 0.8991 | 0.8094 | 144.4 | 66/16384 (0.4%) |
| taxon_sl14 (14d) | 0.4214 | 0.7567 | 0.5786 | 14.0 | 3346/16384 (20.4%) |

**Per-depth efficiency (mse_delta / incr_norm):**

| depth | forest_t1024 | forest_t512 | taxon_ml14_skip | taxon_sl14 |
|-------|-------------|------------|----------------|-----------|
| 0 | **+0.65** | **+0.62** | +0.08 | +0.16 |
| 1 | -0.60 | -0.56 | +0.10 | +0.16 |
| 2 | -1.86 | -1.74 | +0.07 | +0.19 |
| 3 | -3.14 | -2.94 | +0.13 | +0.16 |
| 4 | -4.47 | -4.17 | +0.10 | +0.13 |
| 5 | — | -5.45 | +0.14 | +0.13 |
| 6 | — | — | +0.13 | +0.13 |
| 7 | — | — | +0.15 | +0.13 |
| 8 | — | — | +0.16 | +0.14 |
| 9 | — | — | +0.17 | +0.16 |
| 10 | — | — | +0.17 | +0.17 |
| 11 | — | — | +0.19 | +0.19 |
| 12 | — | — | **+0.22** | **+0.22** |
| 13 | — | — | +0.21 | +0.21 |

**Per-depth useful_frac (projection of depth contribution onto residual):**

| depth | forest_t1024 | forest_t512 | taxon_ml14_skip | taxon_sl14 |
|-------|-------------|------------|----------------|-----------|
| 0 | 0.773 | 0.742 | 0.121 | 0.203 |
| 1 | 0.057 | 0.062 | 0.152 | 0.191 |
| 2 | 0.000 | 0.000 | 0.103 | 0.240 |
| 3 | 0.000 | 0.000 | 0.199 | 0.196 |
| 4 | 0.000 | 0.000 | 0.144 | 0.162 |
| 5 | — | 0.000 | 0.199 | 0.173 |
| 6–8 | — | — | 0.185–0.235 | 0.170–0.185 |
| 9–11 | — | — | 0.245–0.301 | 0.210–0.264 |
| 12 | — | — | 0.376 | 0.317 |
| 13 | — | — | **0.446** | **0.329** |

**Per-depth L0:**

- ForestSAE: exactly 20.0 at every depth (k_trees=20 → uniform sparsity, no depth specialization)
- TaxonSAE ml14: grows 1.9 → 2.9 → 3.8 → ... → 17.6 → 20.0 (binary tree branching, natural increase)
- TaxonSAE sl14: exactly 1.0 at every depth (single leaf constraint)

**Skip connection (taxon_ml14 only):**
- skip_frac = 0.562 (skip absorbs 56% of reconstruction energy)
- skip_tree_cos = -0.008 (skip and tree contributions nearly orthogonal)
- skip_effective_rank = 35.5 (out of rank-128 capacity)

**Finding 9: ForestSAE per-depth independent reconstruction causes destructive interference.**
ForestSAE trains each depth to independently reconstruct x (not the residual). When depth contributions are summed cumulatively, depths 1+ have useful_frac ≈ 0.00 — their contributions are nearly orthogonal to the residual left by depth 0. The negative mse_delta (worsening by -0.6 to -5.4 per depth) means deeper depths actively harm reconstruction quality. This is an architectural failure of per-depth independent reconstruction: each depth learns the same reconstruction target, producing redundant or interfering outputs.

**Finding 10: TaxonSAE ml14 shows monotonically increasing per-depth efficiency.**
With the skip connection absorbing 56% of variance orthogonally (cos = -0.008), the tree is free to specialize on the residual. Efficiency increases from 0.08 (d0, 2 nodes) to 0.22 (d12, 8192 nodes), and useful_frac increases from 0.12 to 0.45. This is the strongest evidence yet of genuine hierarchical structure in reconstruction space: deeper depths contribute more per unit of output norm, and their contributions are more aligned with the remaining residual. The L0 growth (1.9→20.0) follows the binary tree's increasing capacity naturally.

**Finding 11: TaxonSAE sl14 has identical efficiency curve shape despite 10× less capacity.**
Despite only L0=14 total (vs ml14's 144.4), taxon_sl14 shows the same efficiency curve shape as ml14 at depths 9-13 (both reach 0.22 at d12, 0.21 at d13). The mid-tree dip (efficiency drops to 0.13 at d4-7) suggests a routing bottleneck where the single path narrows through mid-depth nodes. The severe overall MSE (0.421 vs ml14's 0.191) is purely a capacity problem, not a structural one — the single-leaf architecture correctly routes but cannot represent enough features.

**Finding 12: Leaf health strongly correlates with architecture quality.**
- ml14: 66/16384 dead (0.4%), gini=0.697 — excellent utilization
- sl14: 3346/16384 dead (20.4%), gini=0.590 — moderate death from routing bottleneck
- forest_t1024: 8110/16384 dead (49.5%), gini=0.793 — half the leaves never activate
- forest_t512: 6471/16384 dead (39.5%), gini=0.781 — significant leaf death

ForestSAE's high dead-leaf rate (40-50%) combined with independent per-depth reconstruction means a large fraction of parameters are wasted. TaxonSAE ml14's near-zero dead-leaf rate suggests the cascaded routing with multi-leaf top-k effectively distributes load across the full tree.

### Open questions (updated)

- Is the ~5-10% genuine semantic feature rate competitive with flat SAEs (BatchTopK) at matched training? A baseline comparison on the same data is needed before concluding this is an architecture problem vs a training duration problem.
- **ForestSAE fix**: Per-depth independent reconstruction is broken. Would cumulative prefix decoding (each depth reconstructs the residual left by shallower depths, not x) force depth specialization and eliminate the destructive interference?
- TaxonSAE ml14's reconstruction hierarchy is clear, but its interpretability hierarchy is weak (function word splitting, not semantic refinement). Is there a loss function modification (e.g., per-depth contrastive objectives, explicit coarse-to-fine regularization) that could push reconstruction-space hierarchy toward interpretable semantic hierarchy?
- Would 244k steps reduce document memorization and improve the genuine semantic feature ratio?
- The skip connection's 56% energy share and orthogonality to the tree is promising. What is the optimal skip rank? Current effective rank is 35.5 out of 128 — could a smaller skip_rank (e.g., 32 or 64) work equally well?

---

## Deleted Runs

| directory | reason |
|-----------|--------|
| pythia160m_L8_65k | Phase 1 65k training, stopped early (loss function investigation) |
| pythia160m_L8_4k_v1_fair | V1 fair baseline, cancelled (wrong experiment) |
| depth_sweep_n19 | OOM on 48GB GPU |
| skip_taxon_r8_4k_quick | Skip dead — zero init gradient lock |
| skip_taxon_r8_4k_v2 | Quick run, superseded by full run |
| hard_t0.5_4k_ref | Unnecessary duplicate of depth_sweep_n11 |
| skip_taxon_r8_4k_quick | Skip dead — zero init gradient lock |
| skip_taxon_r8_4k_v2 | Quick run, superseded by full run |
