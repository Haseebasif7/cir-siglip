# Phase 13c: Training Log -- Widening the Negative Pool

## Step 1: the one change

`NUM_NEGATIVES` 10 -> 20 in `train_core.py` (all of each item's already-mined
top-20 same-category candidates, instead of a random subset of 10).
Everything else -- architecture, loss, margin (0.3), aggregation (min),
optimizer (Adam, lr 5e-5, linear decay), 40-epoch/patience-5 schedule --
copied unchanged from phase 13b. No new negative mining was done; this
reuses phase 13's already-mined `negative_candidates.json` directly.

## Step 2: the training run

Ran locally (MPS), no GPU needed -- same as phase 13b. **One real
interruption worth logging honestly**: the first attempt (started
2026-08-10 22:18) was killed after ~2.7 epochs when the laptop was closed
and went to sleep (confirmed by the user), silently terminating the
background process with no error in the log. Restarted from scratch
(2026-08-11, under `caffeinate -is` to prevent the same interruption) --
training from epoch 0 is a clean, uninterrupted run; the partial first
attempt's checkpoint was discarded, not resumed from, since `run_training`
has no checkpoint-resume path and restarting was simpler and still well
within the "free, no GPU" budget of this phase.

**Full 40-epoch schedule completed, no early stopping** -- LR decayed to
exactly 0. Total wall time: 29,633.6s (~493.9 min, ~8.2 hours), roughly
1.6x phase 13b's ~6.9 hours, consistent with doubling the negative count
(more unique vectors and more per-context-item distance computations per
training sample; see the brief's own note that this was worth flagging).
Best checkpoint (by the same `>1e-4`-improvement rule phase 13b used) is
from **epoch 37** (`val_loss = -3.4448`) -- epochs 38 and 39 both landed
within the noise-floor threshold of that value, not clearing it.

## D_pos/D_neg gap: this phase vs. phase 13b, epoch by epoch

| Epoch | Phase 13b gap (10 negatives) | Phase 13c gap (20 negatives) |
|---|---|---|
| 0 | 0.109 | 0.148 |
| 5 | 0.096 | 0.133 |
| 10 | 0.090 | 0.128 |
| 15 | 0.086 | 0.125 |
| 20 | 0.085 | 0.122 |
| 25 | 0.083 | 0.121 |
| 30 | 0.082 | 0.120 |
| 35 | 0.082 | 0.120 |
| 37 (13c best) / 38 (13b best) | 0.082 | 0.120 |
| 39 (final) | 0.082 | 0.120 |

**The gap did not close relative to phase 13b -- it is larger at every
single epoch, start to finish, and plateaus at a HIGHER value (~0.120 vs
~0.082).** Widening the negative pool did not just fail to help; under this
architecture and loss, it left the model with a worse (larger,
wrong-signed) ranking gap at convergence than the narrower 10-negative
version had.
