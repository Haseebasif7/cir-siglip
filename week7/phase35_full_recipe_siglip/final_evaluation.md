# Phase 35: Final Evaluation -- Test Benchmark, Once

Single model (stage 2's checkpoint, epoch 34, no ensembling per the brief's explicit "do not" list),
evaluated exactly once on `week4/phase12_controllable_modes/data/cir_benchmark.json` (29,681 queries,
n_skipped=0).

## Result

**Phase 35, full recipe (CP pre-training + target-category token + curriculum negatives), single model:
Recall@10/30/50 = 0.1311 / 0.2380 / 0.3075**

## Full comparison table

| Configuration | R@10 | R@30 | R@50 | Notes |
|---|---|---|---|---|
| Phase 14b original (un-invested) | 0.0588 | 0.1286 | 0.1809 | superseded baseline |
| **Phase 35, full recipe (this phase, single model)** | **0.1311** | **0.2380** | **0.3075** | |
| Phase 34, investment-parity CSA-Net (3-seed ensemble) | 0.1674 | 0.2860 | 0.3586 | different architecture |
| Phase 32/31 solo seed 42, no recipe (same arch/backbone/hparams) | 0.1799 | 0.3111 | 0.3844 | **the matched comparison -- see below** |
| Phase 32, 3-seed partial ensemble, no recipe | 0.1897 | 0.3246 | 0.4019 | cited for context; phase 35 doesn't ensemble |
| Phase 28 text ensemble (project's own best, 10-model) | 0.1904 | 0.3267 | 0.4079 | |

## The comparison that matters most (per the brief): phase 35 vs. phase 32's solo model

Both runs share the identical architecture (`d_model=128, d_embed=64, n_heads=8, n_layers=4, d_ffn=512,
dropout=0.1`), the identical frozen SigLIP+text backbone, and the identical training hyperparameters
(`lr=1.5e-4, batch_size=384, margin=0.2, uniformity_weight=0.1, max_epochs=100, patience=25, seed=42`).
The ONLY differences are the three recipe pieces this phase adds (CP pre-training warm start, target-category
token, curriculum negative sampling in place of pure random negatives).

**Phase 35 trails phase 32's solo model by -27.1% / -23.5% / -20.0% relative at R@10/30/50 -- it does not
beat it at any K.** `beats_phase32_solo = false`.

## Derived comparisons

- vs. phase 32's solo model (the matched, recipe-isolating comparison): **-27.1% / -23.5% / -20.0%**
- vs. phase 32's 3-seed ensemble: -30.9% / -26.7% / -23.5%
- vs. phase 28's 10-model ensemble: -31.1% / -27.2% / -24.6%
- vs. phase 34's investment-parity CSA-Net ensemble (different architecture): -21.7% / -16.8% / -14.2%
- vs. phase 14b's original (un-invested): **+123.0% / +85.1% / +70.0%** -- still a large improvement over
  the un-invested starting point, just a smaller one than phase 32's simpler recipe achieves at the same
  investment level.

See `phase35_notes.md` for the full interpretation, including the within-run evidence (stage 2's own
epoch-by-epoch curve, and phase 31/32's own reference curve at the identical hyperparameters) for why this
gap appears, and its honest limits.
