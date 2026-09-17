# Phase 30: Training Log

No learning rate check was run this phase -- per the brief, all hyperparameters are held at week 6's
already-tuned values (lr=0.001, batch size 256, temperature 0.15, 8 random negatives, weight decay 0.0),
identical to phase 27/28's own DEFAULT_CONFIG. Only the projection architecture varies.

## Smoke test (2 epochs, before committing to the full single-seed run)

| Epoch | Train loss | Val Recall@10 | Val Recall@30 | Val Recall@50 | Val agg entropy (norm) |
|---|---|---|---|---|---|
| 0 | 4.728 | 0.1781 | 0.3183 | 0.3972 | 0.919 |
| 1 | 4.334 | 0.1738 | 0.3056 | 0.3861 | 0.894 |

No errors, gradients flowed into every component, aggregation weights summed to 1 as expected. Cleared
to proceed to the full single-seed gate run.

## Single-seed gate run (seed=42), full training curve

| Epoch | Train loss | Val Recall@10 | Val Recall@30 | Val Recall@50 | Train agg entropy (norm) | Val agg entropy (norm) |
|---|---|---|---|---|---|---|
| 0 (best) | 4.728 | 0.1781 | 0.3183 | 0.3972 | 0.916 | 0.919 |
| 1 | 4.334 | 0.1738 | 0.3056 | 0.3861 | 0.902 | 0.894 |
| 2 | 4.040 | 0.1637 | 0.2920 | 0.3686 | 0.835 | 0.801 |
| 3 | 3.837 | 0.1584 | 0.2816 | 0.3572 | 0.739 | 0.715 |
| 4 | 3.689 | 0.1527 | 0.2740 | 0.3479 | 0.663 | 0.634 |

Early stopping fired at epoch 4 (patience=4 past the epoch-0 best) -- five total epochs run.

**The pattern worth naming directly**: the best validation Recall@10 comes at epoch 0, when the
aggregation layer is still close to its near-uniform random-init state (normalized entropy 0.92, close to
the theoretical ceiling of 1.0 -- averaging all four aspects almost equally, close to what plain,
non-decomposed projection would do). Every subsequent epoch, as training loss keeps falling smoothly and
the aggregation weights specialize further away from uniform (normalized entropy drops monotonically,
0.92 -> 0.63 by epoch 4), validation Recall@10 gets monotonically WORSE, not better. This is the direct
opposite of what the brief's stated hypothesis would predict (that letting the model learn which aspects
matter per query should help) -- here, the more the model actually uses its query-conditioned aggregation
mechanism, the worse it performs. See `phase30_notes.md` for the full interpretation and the geometric
explanation `aggregation_qualitative.md`'s per-aspect similarity numbers point to.
