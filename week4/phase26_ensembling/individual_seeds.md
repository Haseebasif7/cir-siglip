# Phase 26, Step 3: Individual Seed Variance

Ten independent copies of the winning phase 25 architecture (hidden_dims=[1024], out_dim=128, lr=0.0005, batch_size=256, weight_decay=0.0, tau=0.15, R=8 -- all unchanged, only the random seed varies), each selected by its own validation-benchmark Recall@10 peak. Seed 42 is phase 25's own confirmatory-retrain checkpoint, reused rather than retrained (see `phase26_notes.md`); seeds 1-9 are new. Recall@10 below is recomputed locally from each checkpoint's projected embeddings (single-member case of the ensemble evaluator) as a bit-consistency cross-check against the Modal-reported training-time number.

| Seed | val Recall@10 (local) | val Recall@10 (Modal training-time) | val Recall@30 | val Recall@50 | best_epoch |
|---|---|---|---|---|---|
| 42 | 0.1656 | 0.1656 | 0.2924 | 0.3691 | 1 |
| 1 | 0.1629 | 0.1629 | 0.2902 | 0.3647 | 1 |
| 2 | 0.1622 | 0.1622 | 0.2899 | 0.3663 | 0 |
| 3 | 0.1624 | 0.1624 | 0.2888 | 0.3660 | 1 |
| 4 | 0.1660 | 0.1660 | 0.2937 | 0.3711 | 1 |
| 5 | 0.1645 | 0.1645 | 0.2882 | 0.3643 | 1 |
| 6 | 0.1640 | 0.1640 | 0.2880 | 0.3634 | 1 |
| 7 | 0.1617 | 0.1617 | 0.2901 | 0.3701 | 1 |
| 8 | 0.1644 | 0.1644 | 0.2917 | 0.3679 | 1 |
| 9 | 0.1635 | 0.1635 | 0.2884 | 0.3625 | 1 |

**Range across the 10 seeds: 0.1617 - 0.1660 (spread 0.0043, mean 0.1637, std 0.0014).**

