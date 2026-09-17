# Phase 34, Step 3: Validation Ensemble Result

3-seed ensemble (42, 1, 2), score-averaged over distances (never embeddings -- same principle as
phases 26/28/30/32/33), evaluated on `week4/phase23_hyperparameter_tuning/data/cir_val_benchmark.json`
(22,595 queries, n_skipped=0).

## Result

| Configuration | Val R@10 | Val R@30 | Val R@50 |
|---|---|---|---|
| Best solo seed (2) | 0.1618 | -- | -- |
| **3-seed ensemble** | **0.1805** | **0.3032** | **0.3759** |

Ensemble gain over the best solo seed: **+11.5%** relative at R@10.

## Comparison against phase 33's own validation ensemble

| Configuration | Val R@10 | Ensemble gain over best solo |
|---|---|---|
| Phase 33 (mined negatives, 3-seed) | 0.1318 | +22.7% |
| Phase 34 (random negatives, 3-seed) | 0.1805 | +11.5% |

Two things worth naming precisely, not conflating:
1. **The absolute validation number is far higher** (0.1805 vs. 0.1318, +37.0% relative) -- this is the
   phase's central result, carried through to the test-benchmark evaluation in `final_evaluation.md`.
2. **The relative ensembling GAIN is smaller** (+11.5% vs. +22.7%). This is consistent with, not
   contradicting, `individual_seeds.md`'s spread finding: phase 33's mined-negative seeds disagreed with
   each other more (std 0.0027) than this phase's random-negative seeds do (std 0.00087), and per this
   project's own established pattern (phase 26/28), ensembling gains are larger when individual members
   disagree more. Random negatives appear to have made the individual models both stronger AND more
   consistent with each other, leaving proportionally less headroom for ensembling to close -- itself a
   secondary, unprompted piece of evidence that the earlier mined-negative training was a noisier
   optimization regime, not just a lower-scoring one.
