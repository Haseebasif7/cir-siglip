# Phase 29: Architecture Notes

## A correction to this phase's own brief, made before anything else

The brief's Context and Step 5 sections cite phase 28's text-only ensemble as Recall@10/30/50 = 0.1888/0.3229/0.4028. Phase 28's actual verified numbers (`week7/phase28_text_ensemble/final_evaluation.md`, `phase28_notes.md`) are **0.1904/0.3267/0.4079**. This mirrors phase 28's own correction of phase 27's brief. Every number in this phase's outputs uses phase 28's real, verified figures, not the brief's.

Also per phase 16/27/28 precedent, work lives in `week7/phase29_cross_attention/`, not the brief's literal `week7_fair_comparison/phase29_cross_attention/` (that top-level folder doesn't exist in this project; `week7/` is the established convention for this week's phases).

## What stays exactly as phase 28 left it

- Per-item base representation: `base_repr(i) = normalize(concat(image(i), text(i)))`, 1536-d, same SigLIP image tower + cascaded-fallback text tower, same `siglip_base.npz` / `text_embeddings.npz`.
- The projection head: `ProjectionHeadGeneral`, 1536 -> 1024 -> 128, same dropout, same L2-normalized output. Unchanged shape, unchanged weights-from-scratch initialization.
- The loss: `mnrl_loss`, unchanged byte-for-byte (in-batch negatives with false-negative masking, plus `R_NEG=8` random extra negatives, softmax cross-entropy at `tau=0.15`).
- Hyperparameters: `lr=0.001` (subject to the Step 2 LR check), `batch_size=256`, `weight_decay=0.0`, `tau=0.15`, `r_neg=8`.
- Category conditioning: not touched, per the brief's explicit instruction (answered in phase 27).

## What changes: candidate-conditioned cross-attention over context

Four new 128->128 linear layers on top of the (unchanged) projection head's output space, exactly as specified: `candidate_key`, `context_query`, `context_value`, `candidate_value`. No depth, no multi-head, no residuals, no LayerNorm -- the brief's "minimum viable inductive bias" instruction, taken literally. Parameter count: 4 x (128*128 + 128) = **66,048** new parameters (the brief estimated "roughly 65k," matches).

Formula, exactly as the brief specifies:

```
attention_weights = softmax(candidate_key(C) . context_query(X_i) / sqrt(128))
query = sum_i attention_weights[i] * context_value(X_i)
score = query . candidate_value(C)
```

## A real problem found during design, and the fix it required

**The problem.** Phase 27/28's training data is single-item anchor -> positive pairs (`positive_edges.json`, one pairwise co-outfit edge per row). Under that data, every training example's "context" has exactly one item. Softmax over a single logit is the constant function 1 -- its derivative with respect to that logit is exactly zero, for any input. That means `candidate_key` and `context_query`'s gradients are **exactly zero** on every training step, provably, not just small. Training this architecture on phase 28's literal pairwise data would leave two of the four new layers at their random initialization forever. At evaluation time, when real queries do have multi-item context, those two layers would then be driving attention with weights that were never trained -- an untrained random projection, not a learned inductive bias. A negative result under that setup wouldn't be evidence that cross-attention doesn't help; it would be evidence that half the mechanism was never given a chance to learn. That's an implementation bug wearing the costume of a negative finding, and it would be dishonest to report it as the latter without flagging it first.

**The fix.** Train on real multi-item context instead of flattened pairs, built from the same underlying source Polyvore data as before. `00_build_context_training_pairs.py` rebuilds training/validation examples directly from the official `nondisjoint/{train,valid}.json` outfit files (the same files phase 9's `positive_edges.json` was itself flattened from) -- for every outfit, every item takes a turn as the held-out target, with every other item in that outfit as its context. This is a strictly more faithful match to the real eval-time query structure (context items + one held-out target) than the old pairwise framing ever was. Same official split, same 251,008-item embedding universe, no new leakage risk. Result: 284,767 train examples / 26,781 val examples, context length mean 4.83 (min 1, max 18) -- closely matching the real CIR benchmark's own query context length (mean 4.86). See `training_pool_summary.md` for the full build report.

This is a data-construction change, not an architecture or loss change, and it was required for the phase's actual hypothesis to be testable at all under its own stated design. It's flagged here explicitly rather than silently substituted, the same way phase 27 flagged its pooling-order asymmetry.

**A remaining simplification, made explicitly and for a stated reason.** True candidate-conditioned attention means the query vector for a context set differs depending on which candidate is being scored against it -- that's the entire point of the mechanism. At evaluation time this phase implements exactly that: for each (query, candidate) pair in a pool, a fresh candidate-conditioned query vector is computed. During *training*, doing that in full (a different query vector for every in-batch negative and every extra negative, not just the true positive) would mean an O(batch_size^2) attention computation per step instead of O(batch_size) -- a real, avoidable cost increase the brief's own "keep this a single-variable test," "don't over-engineer this" instructions argue against taking on without a stated reason. This phase instead conditions the training-time query on the true positive target only (`query_for_candidate(context, target)`), and reuses that same query vector when scoring the batch's in-batch negatives and extra random negatives -- the standard MNRL structure, just with the query now a function of (context, true positive) instead of a mean-pool of context alone. This still routes real, non-degenerate gradient through all four new layers (candidate_key and context_query now condition on the true positive item, which genuinely varies example to example and pairs with a context of size >1 for the large majority of examples), while keeping per-step cost close to phase 28's. The gap between this and full per-candidate conditioning is a genuine approximation, named here rather than left implicit, and is exactly the kind of "practical training-time simplification vs. exact eval-time mechanism" asymmetry phase 27 already established precedent for with its pooling-order split.

## Masking and negative sampling, generalized to multi-item context

Phase 27/28's `positive_sets` (item global index -> set of true co-outfit partners, built from `positive_edges.json`) is reused unchanged as the underlying data. What changes is how it's queried: since an "anchor" is now a context *set* plus a held-out target rather than a single item, both false-negative masking and negative sampling check membership against the **union** of `positive_sets` over every item in the context set plus the target itself, rather than a single item's set. This is a direct generalization of the existing per-anchor logic (a strict superset of what it would check for context size 1), not a new masking philosophy.

## Evaluation: genuinely candidate-conditioned, unlike training

At evaluation time, the full per-(query, candidate) attention mechanism is used, exactly as specified in the brief: for a category's candidate pool, `candidate_key`/`candidate_value` are computed once for every pool item, `context_query`/`context_value` once for every query's context items, and for every (query, candidate) pair a fresh attention-weighted query vector is computed and dotted against that candidate's value vector. This is where the real test of the inductive bias happens -- if `candidate_key`/`context_query` learned anything meaningful from the true-positive-conditioned training signal, it should show up here as real per-candidate reweighting, not just at the trivially-conditioned training step.

## Attention entropy monitoring (collapse diagnostic)

Every epoch, mean attention entropy is logged over the training batch's own context weights, both raw (`-sum(p*log(p))`) and normalized by `log(context_length)` per example (0 = fully collapsed onto one item, 1 = uniform/mean-pool-equivalent). See `training_log.md` for the actual numbers.
