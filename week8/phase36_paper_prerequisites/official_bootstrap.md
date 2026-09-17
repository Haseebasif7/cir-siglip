# Phase 36: Paired Bootstrap on the Official-Split Metrics (primary protocol)

Seed 20260915. FITB: paired percentile bootstrap over the 10,000 questions, B=10000, plus exact McNemar on discordant pairs. AUC: the 20,000 compatibility lines are resampled with replacement (B=5000) and both systems' rank-sum AUC recomputed on the identical resample. **The equivalence bound was pre-declared for Recall@K only; the same 2%-relative rule is applied here as an explicitly post-hoc extension** (delta_AUC = +/-0.0193, delta_FITB = +/-0.0151, relative to `ours_ens`). The CIs and significance tests do not depend on that label.

## Per-system, with 95% bootstrap CI

| System | AUC [95% CI] | FITB [95% CI] |
|---|---|---|
| `ours_ens` | 0.9655 [0.9631, 0.9679] | 75.48% [74.65, 76.29] |
| `ours_solo` | 0.9567 [0.9540, 0.9594] | 72.95% [72.08, 73.82] |
| `ot_ens` | 0.9349 [0.9317, 0.9380] | 74.64% [73.79, 75.46] |
| `ot_solo` | 0.9313 [0.9279, 0.9345] | 73.16% [72.29, 74.01] |
| `csa_ens` | 0.9349 [0.9315, 0.9383] | 70.51% [69.63, 71.37] |

## Paired differences (A minus B)

| Pair | AUC diff | AUC 95% CI | AUC sig. (CI excl. 0) | AUC post-hoc equiv. | FITB diff (pts) | FITB 95% CI (pts) | McNemar b / c | McNemar p | FITB post-hoc equiv. |
|---|---|---|---|---|---|---|---|---|---|
| `ours_ens` - `ot_ens` | +0.0307 | [+0.0281, +0.0332] | yes | no | +0.84 | [+0.13, +1.58] | 736 / 652 | 0.0259 | yes |
| `ours_ens` - `csa_ens` | +0.0307 | [+0.0278, +0.0335] | yes | no | +4.97 | [+4.12, +5.84] | 1243 / 746 | 5.11e-29 | no |
| `ot_ens` - `csa_ens` | +0.0000 | [-0.0024, +0.0023] | no | yes | +4.13 | [+3.39, +4.87] | 964 / 551 | 1.72e-26 | no |
| `ours_solo` - `ot_solo` | +0.0255 | [+0.0227, +0.0282] | yes | no | -0.21 | [-1.03, +0.61] | 865 / 886 | 0.633 | yes |

Interpretation in `phase36_notes.md`.
