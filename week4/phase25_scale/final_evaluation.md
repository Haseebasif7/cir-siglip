# Phase 25, Step 6: Final Evaluation -- Test Benchmark, Once

The single best scaled configuration identified entirely through validation-benchmark comparisons (steps 1-3) evaluated here, exactly once, on the actual test CIR benchmark.

## Final scaled configuration

- hidden_dims: [1024]
- out_dim: 128
- lr: 0.0005 (batch_size=256, weight_decay=0.0, tau=0.15, R=8 held fixed from phase 23/24)
- Parameters: 918656
- Selected at epoch 1 of 17 run (validation Recall@10=0.1656)

## Test-benchmark result

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Phase 9 original | 0.1317 | 0.2464 | 0.3216 |
| Phase 23/24 tuned (128-d, original architecture) | 0.1473 | 0.2684 | 0.3442 |
| **Phase 25 scaled** | **0.1505** | **0.2740** | **0.3503** |

**Scale produced a real improvement over phase 23/24's tuned baseline, on the test benchmark, at every K.**

