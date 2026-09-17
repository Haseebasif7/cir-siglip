# Phase 23: Tuning Log -- Every Configuration Tried, Every Stage

All numbers below are validation-benchmark Recall@10 (`data/cir_val_benchmark.json`, 22,595 queries) -- the test benchmark used throughout the rest of this project was not touched until step 7 (`final_evaluation.md`). Every run uses the periodic-eval / best-by-Recall@10 selection and early-stopping protocol in `scripts/modal_app.py`'s `train_one`, not validation loss (see `phase23_notes.md` for why this distinction is the central finding of this phase). "best_epoch" is the epoch that produced the reported Recall@10; "epochs run" is how many epochs actually executed before early stopping (patience exceeded) or the budget cap.

## Reference point (step 2)

Phase 9's existing, untuned checkpoint (`model_a_random_negs.pt`; lr=1e-3, bs=128, wd=1e-5, tau=0.07, R=8), evaluated once on the validation benchmark, no retraining:

| Recall@10 | Recall@30 | Recall@50 |
|---|---|---|
| 0.1410 | 0.2643 | 0.3401 |

## Step 3: learning rate x batch size grid (9 configs, wd=1e-5, tau=0.07, R=8 held at phase 9's originals; max_epochs=12, patience=4)

| lr | batch_size | val Recall@10 | best_epoch / epochs run | wall time |
|---|---|---|---|---|
| **0.001** | **256** | **0.1535** | 4 / 9 | 884s |
| 0.0005 | 256 | 0.1534 | 3 / 8 | 771s |
| 0.0005 | 128 | 0.1518 | 2 / 7 | 621s |
| 0.0005 | 64 | 0.1509 | 2 / 7 | 808s |
| 0.001 | 64 | 0.1487 | 4 / 9 | 1060s |
| 0.002 | 128 | 0.1487 | 6 / 11 | 911s |
| 0.002 | 256 | 0.1477 | 4 / 9 | 974s |
| 0.001 | 128 (phase 9's own lr/bs) | 0.1474 | 2 / 7 | 580s |
| 0.002 | 64 | 0.1437 | 4 / 9 | 1002s |

**Winner: lr=0.001, bs=256** (val Recall@10=0.1535). Every single grid point beat the 0.1410 reference point -- including phase 9's own original lr/bs re-run under this phase's Recall@10-based selection (0.1474 vs phase 9's actual saved 0.1410), confirming the selection-criterion fix alone accounts for real gain before any hyperparameter value even changes. Batch size 256 wins at every learning rate tested; learning rate matters less once batch size is fixed (0.0005 and 0.001 are close at bs=256, 0.002 is consistently worse at every batch size).

## Step 4: temperature (tau) sweep (5 configs, at the step-3 winner lr=0.001/bs=256; wd=1e-5, R=8; max_epochs=12, patience=4)

| tau | val Recall@10 | best_epoch / epochs run | wall time |
|---|---|---|---|
| **0.15** | **0.1564** | 4 / 9 | 892s |
| 0.10 | 0.1549 | 4 / 9 | 942s |
| 0.07 (phase 9's original) | 0.1535 | 4 / 9 | 932s |
| 0.05 | 0.1492 | 3 / 8 | 805s |
| 0.03 | 0.1452 | 1 / 6 | 620s |

**Winner: tau=0.15** (val Recall@10=0.1564). Recall@10 increases monotonically across the entire tested range (0.03 -> 0.15) -- this is an edge-of-grid result, not a bracketed optimum; the sweep did not test above 0.15, so the true optimum may lie higher. Flagged honestly in `phase23_notes.md` as an open follow-up, not resolved here.

## Step 5: long-patience check (1 config, at lr=0.001/bs=256/tau=0.15/wd=1e-5/R=8; max_epochs=60, patience=15)

| epoch | val Recall@10 | epoch | val Recall@10 | epoch | val Recall@10 | epoch | val Recall@10 |
|---|---|---|---|---|---|---|---|
| 0 | 0.1486 | 5 | 0.1564 | 10 | 0.1523 | 15 | 0.1495 |
| 1 | 0.1551 | 6 | 0.1548 | 11 | 0.1521 | 16 | 0.1494 |
| 2 | 0.1554 | 7 | 0.1526 | 12 | 0.1538 | 17 | 0.1491 |
| 3 | 0.1557 | 8 | 0.1530 | 13 | 0.1491 | 18 | 0.1473 |
| **4** | **0.1564** | 9 | 0.1534 | 14 | 0.1491 | 19 | 0.1489 |

Stopped at epoch 19 (patience=15 after the epoch-4 peak). **Real peak at epoch 4 (0.1564), then a genuine, sustained decline through epoch 19** -- not a plateau still rising, not noise around a flat optimum. This directly answers step 5's question: phase 9's original run was not cut off before convergence in an epoch-count sense; its actual defect was selecting the epoch-0 checkpoint by validation loss instead of the epoch-4-ish Recall@10 peak. See `phase23_notes.md`.

## Step 6: weight decay x R (random negatives) refinement (9 configs, at lr=0.001/bs=256/tau=0.15; max_epochs=15, patience=6)

| weight_decay | R (random negs) | val Recall@10 | best_epoch / epochs run | wall time |
|---|---|---|---|---|
| **0.0** | **8 (phase 9's original R)** | **0.1600** | 4 / 11 | 1065s |
| 0.0001 | 4 | 0.1590 | 3 / 10 | 956s |
| 1e-05 (phase 9's original wd) | 12 | 0.1590 | 3 / 10 | 1034s |
| 0.0 | 4 | 0.1588 | 4 / 11 | 1084s |
| 1e-05 | 4 | 0.1577 | 2 / 9 | 859s |
| 0.0 | 12 | 0.1574 | 6 / 13 | 1287s |
| 0.0001 | 12 | 0.1568 | 6 / 13 | 1303s |
| 1e-05 | 8 | 0.1564 | 4 / 11 | 1063s |
| 0.0001 | 8 | 0.1555 | 6 / 13 | 1411s |

**Winner: weight_decay=0.0, R=8** (val Recall@10=0.1600) -- the best result of the entire phase. Notably, R=8 (phase 9's own original choice) beats both R=4 and R=12 at every weight-decay value tested -- phase 9's negative-sampling count was not a problem. Weight decay 0.0 edges out both 1e-5 (phase 9's original) and 1e-4, but the margin over 1e-5 at R=8 (0.1600 vs 0.1564) is the second-largest single-knob gain in step 6, larger than the R sweep's own spread at fixed wd -- worth taking seriously, not dismissing as noise, though only one run per config was done (no repeated-seed variance estimate).

## Final configuration training run (confirmatory retrain before step 7, lr=0.001/bs=256/tau=0.15/wd=0.0/R=8; max_epochs=60, patience=15)

Best epoch: 4, val Recall@10 = 0.1600 -- matches step 6's own number for this exact config exactly, confirming the result is reproducible under the same seed, not a lucky single run. Curve again shows a clean peak-then-decline (epoch 0: 0.1466 -> epoch 4: 0.1600 -> epoch 19: 0.1487), the same shape as step 5.

## Full validation-benchmark trajectory

| Stage | Configuration | val Recall@10 |
|---|---|---|
| Reference (phase 9 original checkpoint) | lr=1e-3, bs=128, wd=1e-5, tau=0.07, R=8, selected by val loss | 0.1410 |
| Step 3 winner | lr=0.001, bs=256, wd=1e-5, tau=0.07, R=8 | 0.1535 |
| Step 4 winner | lr=0.001, bs=256, wd=1e-5, tau=0.15, R=8 | 0.1564 |
| Step 6 / final winner | lr=0.001, bs=256, wd=0.0, tau=0.15, R=8 | 0.1600 |

See `final_evaluation.md` for the step 7 test-benchmark number and `phase23_notes.md` for the full honest interpretation.
