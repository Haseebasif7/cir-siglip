# Phase 33, Step 3: Hyperparameter Tuning

Base configuration going into this step: A3 (`input_mode=image_text`, `selection_metric=recall10`,
`patience=5`, phase 13b's original `lr=5e-5, batch_size=96`) -- val Recall@10 = 0.0986 (see
`text_input_integration.md`). Ran on Modal (T4) -- this phase's "heavy step," genuinely parallelizable via
`.map()`, unlike steps 1-2 (a handful of isolated local runs). Unlike phase 31's OutfitTransformer grid,
which needed a deliberately cheaper reduced-epoch sweep budget to stay affordable, CSA-Net's tiny model
size made the FULL `max_epochs=40, patience=5` budget affordable for every grid point directly (measured
~9.2s/epoch on T4 including eval; the whole 15-config grid cost $0.7844 total).

## Joint LR x batch-size grid (15 configs + 1 bracket extension)

`LR_GRID = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]` x `BS_GRID = [48, 96, 192]`, centered on phase 13b's own
`lr=5e-5` (not deliberately skewed the way phase 31's OutfitTransformer grid was -- step 1 found no
evidence CSA-Net's original LR was mis-set the way OutfitTransformer's was).

| LR | Batch size | Val R@10 | Best epoch | Epochs run |
|---|---|---|---|---|
| **1e-4** | **48** | **0.1075** | 20 | 26 |
| 1e-4 | 96 | 0.1066 | 25 | 31 |
| 2e-4 | 48 | 0.1044 | 9 | 15 |
| 2e-4 | 192 | 0.1044 | 16 | 22 |
| 1e-4 | 192 | 0.1017 | 25 | 31 |
| 2e-4 | 96 | 0.1031 | 9 | 15 |
| 5e-5 | 48 | 0.1032 | 29 | 35 |
| 5e-5 | 96 | 0.0983 | 29 | 35 |
| 5e-5 | 192 | 0.0913 | 25 | 31 |
| 2e-5 | 48 | 0.0883 | 29 | 35 |
| 2e-5 | 96 | 0.0811 | 25 | 31 |
| 2e-5 | 192 | 0.0758 | 38 | 40 |
| 1e-5 | 48 | 0.0753 | 34 | 40 |
| 1e-5 | 96 | 0.0684 | 36 | 40 |
| 1e-5 | 192 | 0.0631 | 37 | 40 |

**Winner: lr=1e-4, bs=48 (val R@10=0.1075)**, LR bracketed cleanly within the grid (peaks at 1e-4, not an
edge value), but bs=48 sits at BS_GRID's bottom edge -- per the bracketing rule (phase 23/31's own
precedent), extended downward:

| LR | Batch size | Val R@10 |
|---|---|---|
| 1e-4 | 24 | 0.1049 (worse -- bracket resolved, 48 confirmed) |

## Step 3 summary

`input_mode=image_text, selection_metric=recall10, lr=1e-4, batch_size=48, patience=5, max_epochs=40` --
**val Recall@10 = 0.1075**, best epoch 20 of 26 run (clean early stop, well within the 40-epoch budget --
no evidence of truncation the way phase 31's OutfitTransformer sweep needed a separate budget-check step
to rule out).

| Stage | Val R@10 | Relative gain vs. prior stage |
|---|---|---|
| A0 (phase 13b as-shipped) | 0.0786 | -- |
| A2 (selection fix, patience unchanged) | 0.0779 | -0.9% (within noise, see checkpoint_selection_check.md) |
| A3 (+ text) | 0.0986 | +26.6% |
| Step 3 (+ LR/batch-size tuning) | **0.1075** | **+9.0%** |
