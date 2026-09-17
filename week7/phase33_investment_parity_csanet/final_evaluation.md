# Phase 33, Step 5: Final Evaluation -- Test Benchmark, Once

3-seed ensemble (42, 1, 2), score-averaged (distances, not embeddings), identified entirely through validation-benchmark comparisons (step 4: val R@10=0.1318, +22.7% over the best solo seed). Evaluated here, exactly once, on the actual test CIR benchmark.

## Full progression

| Configuration | Recall@10 | Recall@30 | Recall@50 | Notes |
|---|---|---|---|---|
| Phase 13b original CSA-Net reproduction (un-invested) | 0.0725 | 0.1393 | 0.1844 | superseded baseline |
| **Phase 33, 3-seed investment-parity CSA-Net ensemble** | **0.1247** | **0.2164** | **0.2748** | this phase |
| Phase 32, investment-parity OutfitTransformer (3-seed) | 0.1897 | 0.3246 | 0.4019 | cross-architecture comparison |
| Phase 28 text ensemble (project's own best, 10-model) | 0.1904 | 0.3267 | 0.4079 | |

## Derived comparisons

- vs. phase 13b's original: +72.0% / +55.4% / +49.0% relative.
- vs. phase 28's own 10-model ensemble: 65.5% / 66.2% / 67.4% (trails it at every K).
- vs. phase 32's investment-parity OutfitTransformer (the direct cross-architecture comparison this phase exists to make): -34.3% / -33.3% / -31.6% relative (trails it at every K).

n_total=29681 n_skipped=0.

