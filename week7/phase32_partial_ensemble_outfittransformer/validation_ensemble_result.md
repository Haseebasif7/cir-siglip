# Phase 32, Step 4: Validation Ensemble Result

## Individual seeds, locally recomputed vs. Modal-reported

| Seed | Val R@10 (Modal-reported) | Val R@10 (locally recomputed) | Val R@30 | Val R@50 | Match? |
|---|---|---|---|---|---|
| 42 | 0.1924 | 0.1924 | 0.3264 | 0.4065 | yes |
| 1 | 0.1942 | 0.1942 | 0.3301 | 0.4074 | yes |
| 2 | 0.1929 | 0.1929 | 0.3285 | 0.4057 | yes |

## 3-seed ensemble (score averaging)

| Configuration | Val R@10 | Val R@30 | Val R@50 |
|---|---|---|---|
| Solo seed 42 | 0.1924 | 0.3264 | 0.4065 |
| Solo seed 1 | 0.1942 | 0.3301 | 0.4074 |
| Solo seed 2 | 0.1929 | 0.3285 | 0.4057 |
| **3-seed ensemble (42, 1, 2)** | **0.2035** | **0.3402** | **0.4185** |

Ensemble vs. best individual seed: +4.8% relative at R@10 (best solo=0.1942, ensemble=0.2035).
n_total=22595 n_skipped=0.

