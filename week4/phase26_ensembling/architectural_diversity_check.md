# Phase 26, Step 5: Architectural Diversity Check (Secondary)

Phase 25's own width sweep never saved checkpoints (it was a validation-only comparison), so two fresh width=512 models were trained here at phase 25's own winning width=512 setting (lr=0.0005, val Recall@10=0.1609 in the original sweep), seeds 201/202, specifically to test whether mixing architecture sizes into the ensemble adds anything beyond seed diversity alone. This is a smaller, secondary check -- the primary result is the same-architecture sweep in `ensemble_size_sweep.md`.

| Width=512 solo seed | val Recall@10 |
|---|---|
| 201 | 0.1624 |
| 202 | 0.1610 |
| Both width=512 seeds, ensembled together | 0.1733 |

## Same-size comparison at ensemble size 10

| Composition | Seeds | val Recall@10 | val Recall@30 | val Recall@50 |
|---|---|---|---|---|
| All width=1024 (same-architecture baseline) | [42, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 0.1869 | 0.3221 | 0.4018 |
| 8 x width=1024 + 2 x width=512 (mixed) | [42, 1, 2, 3, 4, 5, 6, 7, 201, 202] | 0.1865 | 0.3210 | 0.4006 |

**Architectural diversity did NOT improve on the same-architecture ensemble at this size** (0.1865 vs 0.1869) -- seed diversity alone accounts for the ensemble gain here, mixing in a weaker solo architecture (width=512 solo scores below width=1024 solo) did not help further.

