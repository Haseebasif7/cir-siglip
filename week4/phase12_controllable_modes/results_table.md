# Phase 12: CIR Benchmark Results

Benchmark: `data/cir_benchmark.json`, 26494 pool slots across 11 categories, 29681 leave-one-out queries kept (see `cir_protocol_notes.md` for exactly how the pools/queries were built).

## Recall@K, all configurations

| Configuration | Recall@10 | Recall@30 | Recall@50 | N queries |
|---|---|---|---|---|
| Raw SigLIP (alone) | 0.0553 | 0.1067 | 0.1437 | 29681 |
| Phase 9 Model A (Polyvore-trained, alone) | 0.1317 | 0.2464 | 0.3216 | 29681 |
| Phase 12: Substitute mode alone (alpha=1.0) | 0.1329 | 0.2467 | 0.3215 | 29681 |
| Phase 12: Complement mode alone (alpha=0.0) | 0.1335 | 0.2507 | 0.3250 | 29681 |
| Phase 12: Interpolated blend (alpha=0.5) | 0.1366 | 0.2524 | 0.3289 | 29681 |
| OutfitTransformer (literature anchor, not independently reproduced -- see cir_protocol_notes.md) | 0.0958 | 0.1796 | 0.2198 | -- |

**Caveat on the literature-anchor row**: this project's benchmark (leave-one-out over every qualifying test item-slot, per-category pools capped at 3,000 with queries whose target missed the cap dropped -- see `cir_protocol_notes.md`) is built independently of OutfitTransformer's own exact candidate-pool construction, since no usable reference implementation of their retrieval eval was found. Comparing *rankings* across configurations measured on this project's own harness (the rows above raw SigLIP/Model A/substitute/complement/blend) is a clean, apples-to-apples comparison; comparing *absolute magnitude* against the literature row is not, for the same reason phase 4's DeepFashion-vs-Amazon comparison flagged absolute-magnitude comparisons across different retrieval setups as unsafe.

