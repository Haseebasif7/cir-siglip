# Phase 12b: CIR Benchmark Results, Before/After Comparison With Phase 12

Same benchmark as phase 12 (`week4/phase12_controllable_modes/data/cir_benchmark.json`, unchanged): 26494 pool slots across 11 categories, 29681 leave-one-out queries.

## Recall@K -- phase 12 (batch-local pairwise substitute loss) vs phase 12b (fixed: direct global PCA-128 target + verified loss balancing)

| Configuration | Recall@10 | Recall@30 | Recall@50 | N queries |
|---|---|---|---|---|
| Raw SigLIP (alone) [reference] | 0.0553 | 0.1067 | 0.1437 | 29681 |
| Phase 9 Model A (alone) [reference] | 0.1317 | 0.2464 | 0.3216 | 29681 |
| | | | | |
| Phase 12: Substitute mode alone (alpha=1.0) | 0.1329 | 0.2467 | 0.3215 | 29681 |
| **Phase 12b: Substitute mode alone (alpha=1.0)** | 0.1212 | 0.2254 | 0.2957 | 29681 |
| &nbsp;&nbsp;&nbsp;&nbsp;*delta (12b - 12)* | -0.0117 | -0.0213 | -0.0258 | |
| | | | | |
| Phase 12: Complement mode alone (alpha=0.0) | 0.1335 | 0.2507 | 0.3250 | 29681 |
| **Phase 12b: Complement mode alone (alpha=0.0)** | 0.1200 | 0.2262 | 0.2963 | 29681 |
| &nbsp;&nbsp;&nbsp;&nbsp;*delta (12b - 12)* | -0.0135 | -0.0245 | -0.0287 | |
| | | | | |
| Phase 12: Interpolated blend (alpha=0.5) | 0.1366 | 0.2524 | 0.3289 | 29681 |
| **Phase 12b: Interpolated blend (alpha=0.5)** | 0.1207 | 0.2261 | 0.2957 | 29681 |
| &nbsp;&nbsp;&nbsp;&nbsp;*delta (12b - 12)* | -0.0159 | -0.0263 | -0.0332 | |
| | | | | |
| OutfitTransformer (literature anchor, not independently reproduced) | 0.0958 | 0.1796 | 0.2198 | -- |

**Reading this table**: the headline question is not whether Recall@K went up, but whether **substitute mode's** Recall@K moved DOWN toward raw SigLIP's much lower range (0.055 @10) -- that would indicate it is finally behaving like genuine visual similarity rather than a weakly-offset copy of the compatibility-trained signal. See `control_effectiveness.md` for the direct behavioral checks (overlap, per-item cosine, and the two-axis diagnostic) that actually settle whether the fix worked -- Recall@K alone was already shown in phase 12 to look deceptively reasonable even when the two modes were barely distinguishable.

