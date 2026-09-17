# Phase 16d: Smoke Test Report

Subset: 2000 relevance train edges / 2000 tail train edges (reused from phase 16c), 3 epochs.

Calibration weight_tail (smoke-scale, informal): 0.9916

| Epoch | collapse_rel (relevance head) | collapse_tail (tail head) |
|---|---|---|
| 0 | 0.4469 | 0.4544 |
| 1 | 0.3576 | 0.3442 |
| 2 | 0.2171 | 0.2290 |

**Collapse detected: False** (mean pairwise cosine > 0.9 threshold, checked independently for both dedicated heads)

**Recommendation: leave OFF the uniformity regularizer for the full training run (02_train.py).**