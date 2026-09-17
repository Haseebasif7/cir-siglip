# Phase 25, Step 1: Width Sweep

Holding depth (1 hidden layer) and final embedding size (128) fixed, testing wider hidden layers than phase 23/24's tuned 256 -- doubling (512) and quadrupling (1024) -- with a small learning-rate recheck at each width (batch_size=256, weight_decay=0.0, tau=0.15, R=8 held fixed throughout, per the brief). Selected by validation-benchmark Recall@10, max_epochs=12/patience=4.

| Width | lr | val Recall@10 | best_epoch / epochs run | Parameters | Wall time |
|---|---|---|---|---|---|
| 256 (phase 23/24 baseline) | 0.001 | 0.1600 | 4 / 20 (60-epoch confirmatory run) | 229,760 | -- |
| 512 | 0.0005 | 0.1609 | 1 / 6 | 459,392 | 634s |
| 512 | 0.001 | 0.1605 | 1 / 6 | 459,392 | 622s |
| 512 | 0.002 | 0.1587 | 2 / 7 | 459,392 | 699s |
| **1024** | **0.0005** | **0.1656** | 1 / 6 | 918,656 | 668s |
| 1024 | 0.001 | 0.1603 | 1 / 6 | 918,656 | 688s |
| 1024 | 0.002 | 0.1605 | 1 / 6 | 918,656 | 665s |

**Winner: width=1024, lr=0.0005 (val Recall@10 = 0.1656).**

## Interpretation

Width does help, but modestly and not in a simple "bigger is better, any lr" way:

- Every width/lr combination beat the width=256 baseline (0.1600) except width=512/lr=0.002 (0.1587), so wider layers are net-positive here, but the effect size is small (+0.0056 absolute / +3.5% relative at best).
- **The gain is lr-dependent, not free capacity.** At width=1024, only lr=0.0005 reaches 0.1656 -- lr=0.001 and lr=0.002 at the same width both land around 0.160, barely above the 256-width baseline and *below* width=512's own best result (0.1609). If the learning-rate recheck had been skipped and phase 23/24's tuned lr=0.001 had simply been assumed to carry over unchanged, width=1024 would have looked like a wash (0.1603, statistically indistinguishable from the 256-width baseline) rather than the real winner it actually is at the correct, smaller learning rate. This is exactly the risk the brief flagged in advance.
- The winning run's own best epoch is epoch 1 out of only 6 epochs total -- early stopping fires almost immediately once width increases, consistent with a larger model reaching (and then overfitting past) its validation peak faster than the smaller baseline. See `overfitting_check.md` for the full train-vs-validation trend.
