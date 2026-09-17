# Phase 33, Step 4: Validation Ensemble Result

## Individual seeds, locally recomputed vs. Modal-reported

| Seed | Val R@10 (Modal-reported) | Val R@10 (locally recomputed) | Val R@30 | Val R@50 | Match? |
|---|---|---|---|---|---|
| 42 | 0.1075 | 0.1075 | 0.1907 | 0.2446 | yes |
| 1 | 0.1008 | 0.1008 | 0.1819 | 0.2377 | yes |
| 2 | 0.1028 | 0.1028 | 0.1839 | 0.2392 | yes |

## 3-seed ensemble (score averaging, over distance not similarity -- see train_core.py's own note)

| Configuration | Val R@10 | Val R@30 | Val R@50 |
|---|---|---|---|
| Solo seed 42 | 0.1075 | 0.1907 | 0.2446 |
| Solo seed 1 | 0.1008 | 0.1819 | 0.2377 |
| Solo seed 2 | 0.1028 | 0.1839 | 0.2392 |
| **3-seed ensemble (42, 1, 2)** | **0.1318** | **0.2262** | **0.2888** |

Ensemble vs. best individual seed: +22.7% relative at R@10 (best solo=0.1075, ensemble=0.1318).
n_total=22595 n_skipped=0.

