# Phase 12c: CIR Benchmark Results, Three-Way Comparison

Same benchmark as phase 12/12b (`week4/phase12_controllable_modes/data/cir_benchmark.json`, unchanged): 26494 pool slots, 29681 queries.

| Configuration | Recall@10 | Recall@30 | Recall@50 | N queries |
|---|---|---|---|---|
| Raw SigLIP (alone) [reference] | 0.0553 | 0.1067 | 0.1437 | 29681 |
| Phase 9 Model A (alone) [reference] | 0.1317 | 0.2464 | 0.3216 | 29681 |
| | | | | |
| Phase 12 (batch-local pairwise): Substitute mode | 0.1329 | 0.2467 | 0.3215 | 29681 |
| Phase 12b (PCA-128 target): Substitute mode | 0.1212 | 0.2254 | 0.2957 | 29681 |
| **Phase 12c (ranking distillation): Substitute mode** | 0.0667 | 0.1307 | 0.1734 | 29681 |
| | | | | |
| Phase 12 (batch-local pairwise): Complement mode | 0.1335 | 0.2507 | 0.3250 | 29681 |
| Phase 12b (PCA-128 target): Complement mode | 0.1200 | 0.2262 | 0.2963 | 29681 |
| **Phase 12c (ranking distillation): Complement mode** | 0.0971 | 0.1875 | 0.2471 | 29681 |
| | | | | |
| Phase 12 (batch-local pairwise): Blend (0.5) | 0.1366 | 0.2524 | 0.3289 | 29681 |
| Phase 12b (PCA-128 target): Blend (0.5) | 0.1207 | 0.2261 | 0.2957 | 29681 |
| **Phase 12c (ranking distillation): Blend (0.5)** | 0.0893 | 0.1695 | 0.2255 | 29681 |
| | | | | |
| OutfitTransformer (literature anchor, not independently reproduced) | 0.0958 | 0.1796 | 0.2198 | -- |

**Reading this table**: as established in phases 12 and 12b, Recall@K alone is not sensitive enough to tell whether the two modes are behaviorally distinct -- both prior phases showed substitute and complement scoring almost identically here even when their actual retrieved item sets differed substantially. See `control_effectiveness.md` for the decisive checks (top-10 overlap between modes, and critically, each mode's overlap with RAW SigLIP's own retrieval -- the check that actually settles whether this fix worked).

