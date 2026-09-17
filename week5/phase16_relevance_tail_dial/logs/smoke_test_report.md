# Phase 16: Smoke Test Report

Subset: 2000 train edges / 400 val edges, 3 epochs.

Calibration weight_tail (smoke-scale, informal): 1.0302

| Epoch | train_rel | train_tail | val_rel | val_tail | collapse_rel | collapse_tail |
|---|---|---|---|---|---|---|
| 0 | - | - | - | - | 0.4117 | 0.4001 |
| 1 | - | - | - | - | 0.2818 | 0.2302 |
| 2 | - | - | - | - | 0.1748 | 0.0838 |

**Collapse detected: False** (mean pairwise cosine > 0.9 threshold)

**Recommendation: leave OFF the uniformity regularizer for the full training run (03_train.py).**