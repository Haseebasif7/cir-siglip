# Phase 7: Learned Visual-Compatibility Projection — Notes

## Context

Phases 5 and 6 closed off the substitute/complement labeling approach (`also_viewed` is empty; `also_buy` doesn't cleanly decompose by visual similarity). This phase reframed the goal around VBPR's original idea: train a small MLP projection head on top of frozen SigLIP, using real `also_buy` edges as supervision, evaluated against the untouched 1,872-product eval sample used by every prior phase. This is the first phase in the project that trains anything.

## What was built

- **Training pool**: 27,970 products downloaded (of a ~30,300 target across 11 categories), explicitly excluding all 1,872 eval asins (confirmed zero overlap). Traditional & Cultural Wear (822 final) and Uniforms, Work & Safety (478 final) came in well under the 3,000/category quota, as anticipated — their entire catalog-wide qualifying pool is smaller than that.
- **Hygiene filters**: 187 corrupted-title products removed (0.67%, word-boundary-anchored regex after an initial version false-positived on the brand name "Manyavar"), 1,081 duplicate-image groups collapsed to one representative each (3,064 products demoted, edges remapped rather than dropped). Final cleaned pool: 24,719 products.
- **Training pairs**: 76,293 positive `also_buy` edges with both endpoints in the cleaned pool (68,664 train / 7,629 val). Hard-negative mining via chunked matmul found candidates for all 13,677 anchors with outgoing edges (0 anchors fell back to random-only).
- **Training**: both Model A (random negatives only) and Model B (random + hard negatives) converged smoothly — steady loss decrease, no embedding collapse (mean pairwise cosine similarity stayed low throughout, 0.02-0.18 range for A, 0.02-0.41 for B), early stopping triggered cleanly (epoch 49 for A, epoch 62 for B). Training itself has no evidence of a bug; both models minimized their own objective as designed.

## Result: the learned projection makes retrieval worse, not better

| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| Raw SigLIP (baseline) | 0.502 | 0.578 | 0.191 | 0.144 |
| Model A (random negatives only) | 0.446 | 0.532 | 0.160 | 0.126 |
| Model B (random + hard negatives) | 0.346 | 0.431 | 0.118 | 0.094 |
| Blended alpha=0.7 (raw-heavy) | 0.497 | 0.572 | 0.188 | 0.144 |
| Blended alpha=0.5 (even) | 0.466 | 0.534 | 0.173 | 0.133 |
| Blended alpha=0.3 (Model-B-heavy) | 0.424 | 0.496 | 0.150 | 0.117 |

The sanity check (recomputed raw SigLIP metrics against the existing reported baseline) passed exactly, so this isn't a measurement bug. Every learned configuration underperforms raw SigLIP alone. **Hard negatives make it worse, not better** — Model B is clearly worse than Model A, the opposite of what the ablation was designed to test for. Even the most raw-heavy blend (0.7/0.3) doesn't recover the raw baseline.

## Qualitative check: why it fails

`qualitative_examples/` has 8 side-by-side comparisons (Raw SigLIP / Model B / Blended). The pattern is not uniform breakage — it's **inconsistent, and that inconsistency is itself the finding**:

- Some queries stay coherent under Model B and even show a plausible broadening: a women's Batgirl costume query (`B01B5BIU0E`) retrieves other DC-universe costumes (Wonder Woman, Harley Quinn) under Model B, a step beyond raw SigLIP's tighter "same exact costume, different size/color" matches — arguably a step toward genre-level "compatible" grouping rather than strict visual identity.
- Other queries break outright. A Tommy Hilfiger men's watch (`B005NGRC0W`) retrieves a **women's bra** as Model B's #1 result (similarity 0.639) — completely unrelated to the query in every visual and functional sense. A pink drawstring laundry bag (`B01FWDLMYC`) retrieves **children's Marvel/Disney watches** and a **red "Make America Great Again" cap** in Model B's top-5. Raw SigLIP's results for both queries are sensible (other watches; other bags/pouches).

This is consistent with what the aggregate numbers show: the learned space isn't randomly noisy everywhere, but it introduces enough high-similarity, semantically arbitrary outliers to net hurt retrieval quality overall, and hard-negative training makes this worse rather than better.

## Interpretation: why hard negatives hurt instead of helping

This project's own prior phases already established the mechanism that explains this result:

1. **`also_buy` is a sparse, noisy, popularity-skewed signal** (phases 3, 5, 6) — absence of an edge between two products doesn't mean they're incompatible, it just means Amazon's specific also_buy list for that one product didn't happen to include the other. Mining "hard negatives" as *visually similar items with no also_buy edge to this specific anchor* systematically manufactures false negatives out of genuinely similar, plausibly compatible products (the 0.97 similarity cap excludes exact/near-duplicates, but doesn't exclude the much larger population of "similar but not this anchor's listed partner" items).
2. **Raw SigLIP similarity already correlates well with `also_buy`** — this is the most-validated finding in the project (phases 1, 1b, 4), holding across three Amazon sample constructions and a second, completely different dataset. Explicitly training a projection to *push apart* pairs that are visually similar but not also_buy-linked directly fights the mechanism that made SigLIP's raw similarity useful for this task in the first place.
3. **The training objective and the eval objective are misaligned.** Both models minimized their own train/val contrastive loss cleanly (no bug), but that loss is computed against the training pool's own `also_buy` edges — a different, noisier, sparser supervision signal than "does top-K retrieval on a *different* product set recover held-out `also_buy` edges." Good in-domain loss convergence didn't transfer to better downstream retrieval quality, and the failure mode (occasional catastrophic, semantically arbitrary top-ranked results) suggests the 128-d learned space has real structural damage in places, not just a generically "less sharp" version of SigLIP's space.

## Recommendation

**Do not carry this learned-compatibility approach forward as currently framed.** Frozen SigLIP alone remains the best-performing configuration found anywhere in this project. Specifically:
- Do not deploy Model A, Model B, or any of the tested blends — all underperform the existing frozen baseline on the metric this phase was judged against.
- The ablation's own premise (hard negatives should teach the model something beyond raw similarity) is not supported by this result — hard negatives measurably hurt, not helped, given `also_buy`'s known sparsity/noise properties.

If a learned compatibility signal is still wanted later, this phase's results point toward specific, different next attempts rather than a blanket dead end:
1. **A pairwise ranking loss less sensitive to negative-sampling noise** (e.g. BPR-style, which only requires positive-scores-exceed-a-single-sampled-negative-score rather than a full softmax over hard negatives) may be more robust to `also_buy`'s known false-negative problem than the multiple-negatives-ranking loss used here.
2. **A residual/gated architecture** (the head learns a small correction added to the raw embedding, rather than an independent 128-d space built from scratch) might preserve raw SigLIP's already-good structure by construction, rather than risking wholesale distortion.
3. **Directly optimizing for the eval metric's objective** (e.g. a ranking loss over the actual retrieval task, or reweighting hard negatives by confidence rather than a hard 0/1 exclusion) rather than a generic contrastive loss, might close the train/eval misalignment gap identified above.

None of these has been attempted. Given how clearly negative and internally consistent this result is (sanity-checked baseline, no training bugs, a coherent mechanistic explanation grounded in this project's own prior findings, and concrete qualitative failure examples), the more urgent open question for the project isn't "how do we tune this approach" but whether a learned-embedding approach is the right next step at all, versus other avenues (e.g. the still-open popularity-bias/BFS-sampling-skew threads from phase 3, or revisiting whether a non-embedding-space approach to compatibility, like a lightweight re-ranker over raw SigLIP's own top-K, would sidestep this phase's failure mode).
