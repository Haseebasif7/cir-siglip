# Phase 34, Step 4: Final Evaluation -- Test Benchmark, Once

3-seed ensemble (42, 1, 2), random same-category negatives, score-averaged (distances, not embeddings), identified entirely through validation-benchmark comparisons (step 3: val R@10=0.1805, +11.5% over the best solo seed). Evaluated here, exactly once, on the actual test CIR benchmark. This is the final experimental result for this project's comparison thread.

## Full progression

| Configuration | Recall@10 | Recall@30 | Recall@50 | Notes |
|---|---|---|---|---|
| Phase 13b original CSA-Net reproduction (un-invested) | 0.0725 | 0.1393 | 0.1844 | superseded baseline |
| Phase 33, investment-parity CSA-Net (mined negatives, 3-seed) | 0.1247 | 0.2164 | 0.2748 | superseded by this phase |
| **Phase 34, investment-parity CSA-Net (random negatives, 3-seed)** | **0.1674** | **0.2860** | **0.3586** | this phase, final |
| Phase 32, investment-parity OutfitTransformer (random negatives, 3-seed) | 0.1897 | 0.3246 | 0.4019 | cross-architecture comparison |
| Phase 28 text ensemble (project's own best, 10-model) | 0.1904 | 0.3267 | 0.4079 | |

## Derived comparisons

- vs. phase 13b's original (un-invested CSA-Net): +130.9% / +105.3% / +94.5% relative.
- vs. phase 33's own investment-parity CSA-Net (mined negatives -- the direct single-variable comparison this phase exists to make): +34.2% / +32.2% / +30.5% relative.
- vs. phase 28's own 10-model ensemble: 87.9% / 87.5% / 87.9% (trails it at every K).
- vs. phase 32's investment-parity OutfitTransformer: -11.8% / -11.9% / -10.8% relative (trails it at every K; within 10% relative at every K: False).

n_total=29681 n_skipped=0.

