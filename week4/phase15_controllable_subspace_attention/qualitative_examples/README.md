# Phase 15, Step 8: Qualitative Examples

6 example queries (one per category, randomly sampled per category, seed=7),
using the **discrete-alpha checkpoint** (the better-performing of the two
per `ablation_results.md`). Top-5 retrieved items and the true target's rank
shown at alpha in {0.0, 0.33, 0.67, 1.0}. Full data in
`qualitative_examples.json`.

| Category | Alpha=0.0 top-5 changes vs. alpha=1.0? | Target rank range across alpha |
|---|---|---|
| all-body | No -- identical top-5 at all 4 alphas | 10 (unchanged) |
| bags | **Yes** -- 4th/5th positions swap, ordering shifts | 523 -> 740 (worse as alpha rises) |
| outerwear | Partial -- last two positions swap | 236 -> 228 (slightly better) |
| shoes | No -- identical top-5 at all 4 alphas (single-item query) | 1167 -> 1154 (marginal) |
| sunglasses | **Yes** -- most visible reordering of all 6 examples | 312 -> 259 (meaningfully better) |
| bottoms | No -- top-5 nearly identical, target enters top-5 at alpha=1.0 | 6 -> 5 |

**This matches the quantified finding in `architecture_notes.md` and
`ablation_results.md` directly**: alpha's effect on retrieval is real but
concentrated in specific category pairs (bags, sunglasses show visible
reordering) rather than uniform across all categories (all-body, shoes show
no change at all in their top-5, consistent with those category pairs
sitting at or near the near-zero end of the attention-weight-shift probe's
0-1.909 per-pair range). No category in this small sample shows a dramatic,
qualitatively obvious "substitute-style" vs. "complement-style" behavior
swap the way phase 12d's simpler mechanism did (see that phase's own
`qualitative_examples`) -- consistent with this phase's much smaller
aggregate Recall@K and diagnostic-axis swings.
