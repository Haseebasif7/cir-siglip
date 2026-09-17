# Phase 19: Running Unified Comparison Table

Per the professor's point 6, this is the single, running comparison table for the whole redirected sequence -- every later step (projection, attention, the dial revisit) adds to this same table rather than starting a new one. All rows evaluated on the identical CIR benchmark used throughout this project (`week4/phase12_controllable_modes/data/cir_benchmark.json`, 29,681 queries, 0 skipped, every row).

**Labeling convention, kept consistent with every prior comparison in this project:**
- **"mine"** = trained/measured in this project, on this exact harness.
- **"published"** = the paper's own reported number, different backbone and candidate-pool construction, directional context only.

## Full table, ordered by Recall@10 ascending

| Configuration | Source | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|---|
| Raw SigLIP (frozen, no training) | mine | 0.0553 | 0.1067 | 0.1437 |
| **Phase 19: single-vector feature weighting (this phase)** | **mine** | **0.0577** | **0.1191** | **0.1616** |
| Phase 14b, run 2: OutfitTransformer, category-restricted random negatives (current main baseline) | mine | 0.0588 | 0.1286 | 0.1809 |
| Phase 13b: CSA-Net mechanism, frozen SigLIP backbone | mine | 0.0725 | 0.1393 | 0.1844 |
| OutfitTransformer published (paper, Polyvore Outfits) | published | 0.0958 | 0.1796 | 0.2198 |

## Phase 19's mechanism against raw SigLIP (the direct ablation target)

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Absolute improvement | +0.0024 | +0.0124 | +0.0179 |
| Relative improvement | +4.3% | +11.6% | +12.5% |

A real, consistent, positive improvement at every K, growing with K rather than shrinking -- the opposite pattern from phase 14's and phase 14b's own OutfitTransformer ratio-vs-published-numbers pattern (which climbed toward a bar from below); here it's a genuine gain that gets larger, not smaller, as K widens.

## Phase 19's mechanism against this sequence's other configurations

| Comparison | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Phase 19 as % of phase 14b run 2 (OutfitTransformer, current baseline) | 98.1% | 92.6% | 89.3% |
| Phase 19 as % of CSA-Net reproduction (frozen SigLIP) | 79.6% | 85.5% | 87.6% |
| Phase 19 as % of OutfitTransformer published | 60.2% | 66.3% | 73.5% |

A single 768-parameter element-wise vector -- no hidden layers, no attention, no context aggregation of its own -- reaches 89-98% of phase 14b's OutfitTransformer reproduction (a full transformer with a learnable outfit token) and 80-88% of CSA-Net's own reproduction (a category-pair-conditioned attention mechanism), on the identical benchmark. See `phase19_notes.md` for the honest interpretation and what this implies for the projection/attention steps this bar is meant to set.
