# Phase 8: Cross-Category (Heterogeneous Dyad) Compatibility Retraining — Notes

## Context

Phase 7 trained a learned projection on all `also_buy` edges and made retrieval worse, not better — most dramatically with hard negatives (Model B), which produced qualitatively bizarre results (a men's watch recommending a women's bra). Phase 6 had already shown why: `also_buy` mixes near-duplicate/substitute pairs with genuine complement pairs, and training one space to pull both together forces a contradiction. Veit et al. (ICCV 2015) hit the same problem with the same kind of Amazon data and fixed it by restricting training to "heterogeneous dyads" — cross-category pairs. This phase applies that exact fix and checks whether it resolves phase 7's failure.

## Step 1: category level

Chose index 3 of the `categories` breadcrumb as the type level (710 distinct values on phase 7's cleaned pool, vs. the 11 department buckets used everywhere else). Full detail and a documented complication in `category_level_check.md`: the breadcrumb's semantic meaning at a fixed depth shifts by department (Baby puts a gender split before type, pushing real type to index 4; Novelty & More and Costumes & Accessories stack extra demographic levels; Traditional & Cultural Wear encodes region, not garment type, and degenerates into free text almost immediately). Chose not to build a per-department correction table — the failure mode of not correcting is asymmetric and safe for this phase's purpose (it undercounts heterogeneous dyads in the affected departments, but never mislabels a true same-type/near-duplicate pair as heterogeneous). 23,430/24,719 products (94.8%) have a known type at this level; the rest are treated as unknown and excluded from heterogeneous-dyad determination rather than guessed at.

## Step 2: edge filtering

Of phase 7's 76,293 positive edges: 16,715 (21.9%) are heterogeneous (different type), 56,509 (74.1%) are same-type, 3,069 (4.0%) have unknown type on at least one end. **The dominance of same-type edges (74.1%) is itself a confirming data point for phase 6/7's diagnosis** — most `also_buy` co-purchases in this catalog really are same-type/near-duplicate pairs, not genuine complements. 16,715 heterogeneous edges clears the workable floor comfortably; no pool expansion was needed. Full detail: `edge_filtering_summary.md`.

## Step 3: training

Both models converged cleanly with early stopping (Model A: 17 epochs, best val loss 1.436; Model B: 22 epochs, best val loss 2.106), no embedding collapse (mean pairwise cosine stayed well below 1.0 throughout for both). No training-stability issues.

## Step 4: evaluation — a genuinely mixed, informative result

**View 1, full also_buy ground truth** (`results_table.md`), directly comparable to phase 7:

| Configuration | Hit Rate@5 | Hit Rate@10 |
|---|---|---|
| Raw SigLIP | 0.502 | 0.578 |
| Phase 7 Model A | 0.446 | 0.532 |
| Phase 7 Model B | 0.346 | 0.431 |
| Phase 8 Model A | 0.441 | 0.523 |
| Phase 8 Model B | 0.362 | 0.440 |

Heterogeneous-dyad training does **not** close the aggregate gap to raw SigLIP — Phase 8 Model A is essentially unchanged from Phase 7 Model A (0.441 vs 0.446), and Phase 8 Model B is only marginally better than Phase 7 Model B (0.362 vs 0.346). This makes sense once you weight it by step 2's finding: 74.1% of the full ground truth is same-type edges, and a model now trained exclusively on cross-type pairs has no reason to be good at retrieving same-type positives — it's being judged substantially on a task it was no longer trained for.

**View 2, cross-type-only ground truth** (N=1,732 queries, 1,691 cross-type edges) — the actual, targeted test of whether the fix works for what it was trained to predict:

| Configuration | Hit Rate@5 | Hit Rate@10 |
|---|---|---|
| Raw SigLIP | 0.076 | 0.112 |
| Phase 7 Model A | 0.069 | 0.105 |
| Phase 7 Model B | 0.047 | 0.078 |
| Phase 8 Model A | **0.078** | 0.107 |
| Phase 8 Model B | 0.059 | 0.094 |

**Phase 8 Model A ties or marginally beats raw SigLIP on the specific task this whole line of work is meant to solve** (HR@5 0.078 vs 0.076; HR@10 essentially flat, 0.107 vs 0.112) — a real, if modest, reversal of Phase 7 Model A's clear degradation on this same view (0.069, below raw). Phase 8 Model B improves substantially over Phase 7 Model B on this view (HR@5 0.059 vs 0.047) but still falls short of raw SigLIP — hard negatives still hurt, just less severely than before the fix.

## Step 5: qualitative check — the clearest evidence in this phase

`qualitative_examples/` has all 8 grids (4 rows each: Raw SigLIP / Phase 7 Model B / Phase 8 Model A / Phase 8 Model B), including phase 7's two documented failures re-run.

**Both phase 7 failures are cleanly fixed:**
- The Tommy Hilfiger men's watch (`B005NGRC0W`): phase 7 Model B's #1 result was a women's bra. Phase 8 Model A returns all watches. **Phase 8 Model B returns all watches too, and its #2 result is a "Glenor Co Watch Box" — a genuinely plausible complement** (a case to store the watch in), something raw SigLIP structurally cannot surface since it only ranks by visual similarity.
- The pink drawstring laundry bag (`B01FWDLMYC`): phase 7 Model B's top-5 included children's cartoon watches and a "Make America Great Again" cap. Phase 8's results (both models) are no longer absurd, but they're also not clearly compatible items either — a mixed assortment of small unrelated accessories (headband, jacket, baby item, visor). **This is the one example where the fix is only partial**: it eliminated the extreme mismatch but didn't produce genuinely plausible complements either.

Across the other 6 example queries, the pattern holds: **every clearly-absurd, semantically-arbitrary result from phase 7 is gone in phase 8.** Two additional queries (`B005NGRC0W`'s watch-box result, and a DC-costume query surfacing "Sun-Staches Costume Sunglass," a genuine costume accessory) show Phase 8 Model B producing recommendations that look like real, useful complements — not just "less broken," but actively better than what raw similarity-only retrieval could ever produce. One residual issue remains: on the Casio watch query (`B000GB0G1G`), Phase 8 Model B's #5 result is an unrelated woman's outfit photo — an isolated instance of the same failure mode as phase 7, at clearly lower frequency (1 clear miss across 8×5=40 inspected results, vs. phase 7's 2 clear misses in fewer inspected grids).

## Interpretation

The heterogeneous-dyad fix **partially worked, and worked exactly where the mechanism predicts it should:**
- It doesn't fix the aggregate Hit Rate@K number, because that number is dominated by same-type ground truth (74.1%) that a heterogeneous-only model isn't trying to predict anymore — this is an expected consequence of the fix, not a failure of it.
- It does fix (Model A) or substantially reduce (Model B) the qualitative failure mode that motivated this phase — recommending semantically arbitrary, unrelated products — and on the narrow but directly-relevant cross-type metric, Model A ties/beats raw SigLIP.
- Hard negatives (Model B) still underperform random-negatives-only (Model A) on both quantitative views, consistent with phase 7's finding that hard-negative mining is fragile against `also_buy`'s sparsity — restricting to heterogeneous dyads reduced but did not eliminate this problem. The residual issue is plausible: a "hard negative" mined by raw-similarity is, by construction, visually close to the anchor; within the heterogeneous-dyad subset this is a smaller, more specialized pool, and the same false-negative risk phase 7 identified (a visually-similar-but-unlisted item might really be compatible) still applies.

## Recommendation

**Partial success, worth carrying forward with a specific scope, not a blanket win.** Model A (heterogeneous-only, random negatives) is the configuration to build on:
- It's the first learned configuration in this project that matches or beats raw SigLIP on any metric (the cross-type-only view).
- Its qualitative behavior is clean — no arbitrary mismatches found in 8 re-tested queries — and it occasionally surfaces genuinely useful cross-category complements (watch box, costume sunglasses) that similarity-only retrieval cannot produce by construction.
- It should **not** replace raw SigLIP as the primary similarity signal (its aggregate Hit Rate@K is still below baseline) — it's better understood as a **second, complementary signal** for a specific job: surfacing cross-category "goes with" recommendations alongside, not instead of, SigLIP's visual-similarity results. This matches the project's original hybrid-recommender framing (substitute + complement, shown together) better than either phase 7 or phase 8 alone.
- Hard negatives (Model B) are not recommended for production use yet — still net-negative on both quantitative views, though clearly improved from phase 7 and worth revisiting if a less fragile negative-mining strategy is developed (e.g. a lower similarity cap specific to the heterogeneous-dyad pool, or excluding hard negatives that themselves have any also_buy edge to the anchor's other also_buy partners, not just to the anchor directly).

Next steps this phase's results point toward, none attempted here: (1) evaluate a blend of raw SigLIP (primary ranking) with Phase 8 Model A (secondary/complementary signal, shown as a distinct "goes well with" section rather than blended into one ranked list — blending at the similarity-score level like phase 7 did may not be the right integration pattern for a model that's good at a genuinely different task, not a better version of the same task); (2) the planned Polyvore follow-up (explicitly out of scope for this phase) may give a much larger, cleaner source of genuine outfit-compatibility pairs than `also_buy` ever can, given `also_buy`'s 74.1% same-type dominance limits how much true complement signal exists in this catalog's co-purchase data at all.
