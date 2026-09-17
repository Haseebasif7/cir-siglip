# Phase 26, Step 6: Final Evaluation -- Test Benchmark, Once

Best ensemble composition identified entirely through validation-benchmark comparisons (steps 3-5): same-architecture (width=1024): seeds [42, 1, 2, 3, 4, 5, 6, 7, 8, 9], validation Recall@10=0.1869. Evaluated here, exactly once, on the actual test CIR benchmark.

## Full progression

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Phase 9 original | 0.1317 | 0.2464 | 0.3216 |
| Phase 23/24 tuned (single model) | 0.1473 | 0.2684 | 0.3442 |
| Phase 25 scaled (single model, width=1024) | 0.1505 | 0.2740 | 0.3503 |
| Phase 26 solo (seed=42, single model, re-measured here for a like-for-like reference point) | 0.1505 | 0.2740 | 0.3503 |
| **Phase 26 ensemble (same-architecture (width=1024): seeds [42, 1, 2, 3, 4, 5, 6, 7, 8, 9])** | **0.1767** | **0.3054** | **0.3828** |

**Ensembling produced a real improvement over phase 25's single best model, on the test benchmark, at every K.**

