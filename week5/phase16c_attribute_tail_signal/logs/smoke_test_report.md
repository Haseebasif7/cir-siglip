# Phase 16c: Smoke Test Report

Subset: 2000 relevance train edges / 2000 tail train edges, 3 epochs.

Calibration weight_tail (smoke-scale, informal): 0.9910

| Epoch | collapse_rel | collapse_tail |
|---|---|---|
| 0 | 0.3215 | 0.3012 |
| 1 | 0.3425 | 0.3272 |
| 2 | 0.2474 | 0.2439 |

**Collapse detected: False** (mean pairwise cosine > 0.9 threshold)

**Recommendation: leave OFF the uniformity regularizer for the full training run (05_train.py).**