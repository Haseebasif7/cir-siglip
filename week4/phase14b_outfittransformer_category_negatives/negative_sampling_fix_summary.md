# Phase 14b, Step 1: Negative Sampling Category-Match Rate, Before and After

Measured the same way as phase 14's own diagnostic (`week4/phase14_outfittransformer_siglip/phase14_notes.md`): 200 random 96-outfit batches from the training split.

| | Category-match rate | Source |
|---|---|---|
| Before (phase 14, in-batch negatives, no category restriction) | 13.1% | `week4/phase14_outfittransformer_siglip/phase14_notes.md`, line 29 (cited, not recomputed -- the unrestricted in-batch mechanism it measured no longer exists in this phase's training loop) |
| **After (phase 14b, same-category candidate pool)** | **1.0000 (100.0%)** | computed fresh here, 192000 total sampled negatives across 19200 samples |

Mean negatives actually returned per sample: 10.00 / 10 target. Samples that fell short of 10 negatives (target category's own pool, plus mined candidates, together held fewer than 10 eligible items): 0 / 19200 (0.00%).

Every negative returned by `_sample_negatives` is drawn either from the mined same-category candidate list (`week4/phase13_csa_net_baseline/data/negative_candidates.json`) or, as a top-up, from the target's own category pool (`train_items_by_category[target_cat]`) directly -- both sources are restricted to `target_cat` by construction, so any match rate below 100% would indicate a bug in `item_cat` lookup consistency, not a sampling choice.
