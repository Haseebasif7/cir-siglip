# Phase 23, Step 7: Final Evaluation -- Test Benchmark, Once

The single best configuration identified entirely through validation-benchmark comparisons (steps 3-6, `tuning_log.md`) evaluated here, exactly once, on the actual test CIR benchmark used throughout this project -- verifying that any improvement is real, not a validation-benchmark artifact.

## Final tuned configuration

| Hyperparameter | Phase 9 original | Phase 23 tuned |
|---|---|---|
| Learning rate | 1e-3 | 0.001 |
| Batch size | 128 | 256 |
| Weight decay | 1e-5 | 0.0 |
| Temperature (tau) | 0.07 | 0.15 |
| Random negatives (R) | 8 | 8 |
| Selection signal | validation loss (best at epoch 0) | validation-benchmark Recall@10 (best at epoch 4 of 20 run) |

## Test-benchmark result (the one and only comparison that matters)

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Phase 9 original (cited) | 0.1317 | 0.2464 | 0.3216 |
| Phase 9 original (re-verified here, same checkpoint) | 0.1317 | 0.2464 | 0.3216 |
| **Phase 23 tuned** | **0.1473** | **0.2684** | **0.3442** |

## Change over phase 9's original

| K | Absolute change | Relative change |
|---|---|---|
| 10 | +0.0156 | +11.8% |
| 30 | +0.0220 | +8.9% |
| 50 | +0.0226 | +7.0% |

**Tuning produced a real improvement over phase 9's original result, on the test benchmark, at every K.** This was verified with a single, final test-benchmark evaluation only after every tuning decision was already locked in on the separate validation benchmark -- not selected by repeatedly checking against this specific benchmark.

