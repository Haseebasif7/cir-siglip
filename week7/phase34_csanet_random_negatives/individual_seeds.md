# Phase 34, Step 3: Individual Seed Results

Same 3 seeds as phase 33's own ensemble (42, 1, 2), same winning hyperparameters (lr=1e-4, batch_size=48,
patience=5, max_epochs=40, image+text, recall10 selection), the only variable changed being negative
sampling (random same-category instead of mined). All trained locally on the M4, no Modal budget used.

## Per-seed validation R@10

| Seed | Val R@10 | best_epoch | n_epochs_run | wall_time (s) | Note |
|---|---|---|---|---|---|
| 42 | 0.1610 | 27 | 33 | 2231 | step 2's gate run |
| 1 | 0.1597 | 39 | 40 | 3560 | ran the full 40 epochs, still climbing at the end |
| 2 | 0.1618 | 31 | 37 | 2517 | |

Mean: **0.1608**, std (population): **0.00087**, range: **0.0021** (0.1597-0.1618).

## Comparison against phase 33's own seed spread

Phase 33's mined-negative CSA-Net seeds had std **0.0027** (`phase33/individual_seeds.md`) -- roughly
3x wider than this phase's random-negative spread. Phase 32's OutfitTransformer seeds (also random
negatives) had std **0.00074** -- much closer to this phase's 0.00087 than to phase 33's own CSA-Net number.

This is a second, unprompted piece of corroborating evidence for the negative-sampling-asymmetry hypothesis,
beyond the mean R@10 gain itself: mined hard negatives didn't just produce a lower mean, they produced a
noisier, less consistent training signal across random seeds -- consistent with hard-negative mining being
a harder, more idiosyncratic optimization target than uniform random sampling, for this architecture and
loss combination. Not something this phase set out to measure, but worth reporting since it appeared.

## Seed 1's full-budget note

Seed 1 ran the complete 40-epoch budget without early stopping (patience=5 was never triggered), and its
`curve` shows val R@10 still rising slightly at epoch 39 (see `data/seed_results.json`). This did not change
the decision to stop at `max_epochs=40` -- reusing phase 33's own budget exactly, per this phase's own scope
(re-tuning `max_epochs` under the new negative-sampling scheme is out of scope, same reasoning as the
hyperparameter reuse noted in `negative_sampling_change.md`). Flagged here for completeness, not acted on.
