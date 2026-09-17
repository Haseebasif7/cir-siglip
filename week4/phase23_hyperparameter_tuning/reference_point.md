# Phase 23, Step 2: Reference Point

Phase 9's existing, untuned checkpoint (`model_a_random_negs.pt`, LR=1e-3, BATCH_SIZE=128, WEIGHT_DECAY=1e-5, TAU=0.07, R=8/H=0), evaluated on this phase's own validation benchmark (`data/cir_val_benchmark.json`, never the test benchmark):

| Recall@10 | Recall@30 | Recall@50 | n_queries | n_skipped |
|---|---|---|---|---|
| 0.1410 | 0.2643 | 0.3401 | 22595 | 0 |

Every configuration tried in this phase is compared against this number (validation-benchmark Recall@10 as the primary selection signal), not against phase 9's cited test-benchmark number (0.1317/0.2464/0.3216) directly -- those two numbers are not expected to match exactly since they come from different query sets, but should be in a broadly similar range as a sanity check.
