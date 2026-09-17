# Phase 28: Text-Only Ensemble -- Honest Interpretation

## A correction to this phase's own brief, made before anything else

Phase 28's brief states phase 27's text-only result as "Recall@10 from 0.1473 to 0.1590, a real 7.9 percent relative gain," and describes the category conditioning failure as caused by embedding collapse ("0.86 mean pairwise cosine similarity across 11 categories... the model learned no meaningful differentiation"). Neither matches what phase 27 actually produced and verified:

- Phase 27's text-only single-seed Recall@10 was **0.1656** (`week7/phase27_text_and_category/results_table.md`), not 0.1590.
- The category embedding table's mean pairwise cosine was **0.0815**, not 0.86, and the check explicitly found it was NOT collapsed -- real, structured differentiation (`week7/phase27_text_and_category/category_embedding_check.md`). The actual diagnosed cause of category conditioning's failure was a training/evaluation negative-sampling mismatch (a shortcut the model could exploit during training that doesn't exist at evaluation time), the opposite of "no differentiation."

This phase proceeds using phase 27's own verified numbers throughout, not the brief's mismatched figures. The rest of this document, and every number in `individual_seeds.md`, `ensemble_size_sweep.md`, and `final_evaluation.md`, traces back to files this phase itself generated or to phase 27's own verified `results_table.md` -- never to the brief's context paragraph.

## Headline: yes, the text-only ensemble beats the image-only ensemble, at every K

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Week 6 image-only, single tuned model | 0.1473 | 0.2684 | 0.3442 |
| Week 6 image-only, 10-model ensemble | 0.1767 | 0.3054 | 0.3828 |
| Phase 27 / 28 text-only, single seed | 0.1656 | 0.2947 | 0.3733 |
| **Phase 28 text-only, 10-model ensemble** | **0.1904** | **0.3267** | **0.4079** |

The text-only ensemble beats the image-only ensemble by **+7.7%/+7.0%/+6.6%** relative at K=10/30/50 -- a clean win at every K, not a mixed or marginal result. This is now the strongest test-benchmark result in the project, superseding week 6's image-only ensemble.

## Does the single-seed gain survive ensembling proportionally? Not quite -- and that itself is the interesting finding

The brief predicted (using its own, since-corrected numbers) that a ~7.9% single-seed gain would carry through to the ensemble level. Using phase 27's actual, verified single-seed comparison (text-only 0.1656 vs. the week 6 tuned single model 0.1473), the single-seed relative gain was actually **+12.4%** at K=10 -- larger than the brief assumed. But the ensemble-level gain measured here is **+7.7%** at K=10 (against week 6's own image-only ensemble). Neither the brief's original prediction nor the corrected single-seed prediction survived unchanged into the ensemble comparison -- the gain shrank.

The reason is visible directly by comparing how much each approach gained from ensembling on its own terms:

- Image-only: single model 0.1505 (phase 25's scaled checkpoint) -> ensemble 0.1767 = **+17.4%** relative gain from ensembling (phase 26's own reported figure).
- Text-only: single model 0.1656 (seed=42, phase 27) -> ensemble 0.1904 = **+15.0%** relative gain from ensembling.

**Ensembling helped image-only proportionally more than it helped text-only.** This is exactly the "closes the gap" scenario the brief itself flagged as one of two possible outcomes, and it's the one that happened, though not enough to erase the underlying advantage -- text-only's ensemble still wins outright, just by a smaller margin than the single-seed comparison alone would have predicted.

## Individual seed variance: not meaningfully different from image-only's

Phase 26's image-only seeds landed in a range of 0.1617-0.1660 (spread 0.0043, std 0.0014). This phase's text-only seeds landed in 0.1763-0.1819 (spread 0.0056, std 0.0016) -- a very slightly wider absolute spread, but on a higher base (mean 0.1794 vs. 0.1638), so the coefficient of variation is nearly identical (~0.0085 vs. ~0.0089). **Text does not produce a meaningfully tighter or wider seed-to-seed variance than image-only** -- the brief's hypothesis that "text is a more consistent signal" isn't supported by the raw spread numbers, even though ensembling still helped image-only proportionally more. Whatever is causing image-only to benefit more from ensembling, it isn't showing up as a difference in how tightly the individual seeds cluster.

## Ensemble size sweep: same diminishing-returns shape as phase 26

| Size | Val Recall@10 | Gain over solo (0.1819) |
|---|---|---|
| 2 | 0.1925 | +0.0106 |
| 3 | 0.1965 | +0.0146 |
| 5 | 0.2016 | +0.0197 |
| 7 | 0.2020 | +0.0201 |
| 10 | 0.2047 | +0.0228 |

Most of the gain arrives by size 5 (matching phase 26's own observation), with real but shrinking returns after that. Size=10 is still the best point tested, consistent with phase 26.

## Inference cost: a wash between the two options, unlike week 6's single-vs-ensemble tradeoff

Both the winning configuration here and week 6's own winning ensemble are 10-seed ensembles, so choosing text-only over image-only doesn't add any additional inference cost beyond what week 6 already accepted when it adopted ensembling -- this is a clean upgrade, not a cost/quality tradeoff the way single-model-vs-ensemble was in week 6.

## Final, honest verdict

**Text-only, ensembled, is the new best result in this project.** Test-benchmark Recall@10/30/50 = **0.1904/0.3267/0.4079**, beating week 6's image-only ensemble (0.1767/0.3054/0.3828) at every K, at the same inference cost (10 models either way). The gain is real but smaller in relative terms than the single-seed comparison alone predicted, because ensembling recovers somewhat more headroom for image-only than for text-only -- not because text-only's individual seeds are unusually tight (they aren't, relative to their own mean). Category conditioning remains correctly excluded from this phase per its own explicit scope; the fix diagnosed in phase 27 (category-restricted negative sampling) remains untested and open for a future phase.

## Reference configuration going forward

- **Ensemble**: 10 independently-seeded copies of phase 27's text-only architecture (image+text concat, 1536-d input, `hidden_dims=[1024]`, `out_dim=128`; `lr=0.001, batch_size=256, weight_decay=0.0, tau=0.15, r_neg=8` -- all unchanged from week 6/phase 27). Seeds: 42 (phase 27's own text_only checkpoint, reused directly) plus 1-9 (trained fresh this phase).
- **Combination method**: average per-query-per-candidate cosine similarity scores across all 10 members (never raw embeddings), identical principle to phase 26.
- **Test-benchmark result**: Recall@10/30/50 = 0.1904/0.3267/0.4079.
- **Checkpoints**: `week7/phase28_text_ensemble/models/text_ensemble_seed{42,1..9}.pt` (10 files).
