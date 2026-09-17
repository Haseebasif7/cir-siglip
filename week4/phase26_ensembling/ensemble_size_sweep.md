# Phase 26, Step 4: Ensemble Size Sweep

Ensembles built by score-averaging (see `cir_eval_ensemble.py`) the first N of the 10 trained members, in a fixed order (seed 42 first -- phase 25's own winner -- then seeds 1-9 in ascending order). Evaluated on the validation benchmark.

| Ensemble size | Seeds used | val Recall@10 | val Recall@30 | val Recall@50 |
|---|---|---|---|---|
| 1 (solo, best single seed for reference) | 42 | 0.1656 | 0.2924 | 0.3691 |
| 2 | [42, 1] | 0.1767 | 0.3093 | 0.3861 |
| 3 | [42, 1, 2] | 0.1805 | 0.3129 | 0.3922 |
| 5 | [42, 1, 2, 3, 4] | 0.1854 | 0.3184 | 0.3991 |
| 7 | [42, 1, 2, 3, 4, 5, 6] | 0.1852 | 0.3200 | 0.3994 |
| 10 | [42, 1, 2, 3, 4, 5, 6, 7, 8, 9] | 0.1869 | 0.3221 | 0.4018 |

**Best: size=10 (val Recall@10=0.1869).**

