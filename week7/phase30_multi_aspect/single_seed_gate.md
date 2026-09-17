# Phase 30, Step 3: Single-Seed Gate

Baseline: phase 27/28's own text-only single-seed result, seed=42, mean-pooled single-vector
projection -- val Recall@10 = 0.18194 (`week7/phase27_text_and_category/data/train_variants_results.json`,
`results_table.md`). Same baseline phase 29 used for its own gate.

Gate rule, per the brief: proceed to the full ten-seed ensemble only if the multi-aspect single seed
beats this baseline by at least 2% relative on validation Recall@10. Otherwise stop, report, and treat
this as the answer for now.

| Configuration | Val Recall@10 |
|---|---|
| Phase 27/28 text-only, single seed (42), mean-pool | 0.1819 |
| Phase 30 multi-aspect, single seed (42) | 0.1781 |

**Relative gain: -2.09%. Decision: NO-GO.**

This is meaningfully different in character from phase 29's own single-seed gate result (-32.5%, a clear
collapse). Here the multi-aspect architecture lands almost exactly at the baseline -- 2.1% below it, just
past the "same or worse" line the brief draws, not a dramatic regression. Given the added architectural
complexity (four separate aspect heads, a learned aggregation layer, per-query-conditioned scoring) buys
no measurable improvement and in fact costs a small amount of quality, the brief's own instruction applies
directly: don't scale up something that isn't showing signal. No ten-seed ensemble is trained, no
ensemble size sweep is run, and the test benchmark is never touched in this phase.
`individual_seeds.md`, `ensemble_size_sweep.md`, and `final_evaluation.md` are intentionally not produced --
conditional on reaching ensemble scale per the brief's own required-outputs list, and this phase does not
reach it.

See `training_log.md` for the full epoch-by-epoch curve and the specific pattern the gate result sits on
top of (best epoch is the one closest to uniform aggregation, and quality degrades monotonically as the
aggregation mechanism specializes further), and `phase30_notes.md` for the full honest interpretation.
