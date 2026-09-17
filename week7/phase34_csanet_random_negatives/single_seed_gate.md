# Phase 34, Step 2: Single-Seed Gate

Seed 42, phase 33's exact winning configuration (image+text, recall10 selection, lr=1e-4, batch_size=48, patience=5, max_epochs=40), random same-category negatives instead of mined candidates -- the only variable under test.

## Result

- Phase 33 (mined negatives), seed 42, validation R@10: **0.1075**
- Phase 34 (random negatives), seed 42, validation R@10: **0.1610**
- Relative gain: **+49.7%**
- Gate (>= 5% relative to proceed to ensembling): **CLEARED**
- best_epoch=27, n_epochs_run=33, wall_time=2231s, n_params=100485

## Decision

Gate cleared. Proceeding to step 3: train 2 additional seeds (1, 2) and build the 3-seed ensemble, matching phase 33's own ensemble size exactly.
