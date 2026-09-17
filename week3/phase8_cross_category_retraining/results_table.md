# Phase 8, Step 4: Evaluation on the Untouched 1,872-Product Eval Sample

Sanity check: recomputed raw SigLIP metrics matched the existing reported baseline within tolerance -- confirms the untouched eval sample and eval methodology are unchanged before trusting the learned-model rows below.

Cross-type-only ground truth: 1691 edges across 1732 queries (140 queries excluded from this view, unknown own type at index 3).

## View 1: Full also_buy ground truth (directly comparable to phase 7 and raw SigLIP)

| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| Raw SigLIP (baseline) | 0.502 | 0.578 | 0.191 | 0.144 |
| Phase 7 Model A (random negs, all also_buy) | 0.446 | 0.532 | 0.160 | 0.126 |
| Phase 7 Model B (hard negs, all also_buy) | 0.346 | 0.431 | 0.118 | 0.094 |
| Phase 8 Model A (random negs, heterogeneous-only) | 0.441 | 0.523 | 0.160 | 0.120 |
| Phase 8 Model B (hard negs, heterogeneous-only) | 0.362 | 0.440 | 0.117 | 0.089 |

## View 2: Cross-type-only ground truth (N=1732 queries, 1691 cross-type edges -- isolates whether the fix works for what it was trained to predict)

| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| Raw SigLIP (baseline) | 0.076 | 0.112 | 0.019 | 0.016 |
| Phase 7 Model A (random negs, all also_buy) | 0.069 | 0.105 | 0.017 | 0.014 |
| Phase 7 Model B (hard negs, all also_buy) | 0.047 | 0.078 | 0.012 | 0.011 |
| Phase 8 Model A (random negs, heterogeneous-only) | 0.078 | 0.107 | 0.020 | 0.016 |
| Phase 8 Model B (hard negs, heterogeneous-only) | 0.059 | 0.094 | 0.014 | 0.012 |

