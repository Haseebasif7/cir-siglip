# Phase 10, Step 4.1: Official Polyvore Test-Split Evaluation

Same protocol as phase 9 (compatibility AUC via rank-sum formula, FITB accuracy). Phase 9 numbers carried forward verbatim for direct comparison, not recomputed.

| Configuration | Compatibility AUC | FITB Accuracy |
|---|---|---|
| Raw SigLIP (reference) | 0.7170 | 0.4840 |
| Phase 9 Model A (Polyvore-trained, random negs) | 0.9470 | 0.7030 |
| Phase 9 Model B (Polyvore-trained, hard negs) | 0.9370 | 0.6840 |
| **Phase 10 color-invariant model (random negs + invariance loss)** | **0.9420** | **0.6907** |

(Compatibility test: n=20000 lines scored; FITB test: n=10000 questions scored.)


# Phase 10, Step 4.2: Transfer Test on the Untouched Amazon Eval Sample

Sanity check: recomputed raw SigLIP metrics matched the existing reported baseline within tolerance -- confirms the untouched Amazon eval sample and eval methodology are unchanged before trusting the transfer-test row below.

Cross-type-only ground truth: 1691 edges across 1732 queries (140 excluded, unknown own type) -- identical to phase 8/9's view.

## View 1: Full also_buy ground truth

| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| Raw SigLIP (baseline) | 0.502 | 0.578 | 0.191 | 0.144 |
| Phase 8 Model A (Amazon-trained, random negs) | 0.441 | 0.523 | 0.160 | 0.120 |
| Phase 9 Model A (Polyvore-trained, naive, random negs) | 0.315 | 0.384 | 0.098 | 0.073 |
| **Phase 10 color-invariant model** | **0.301** | **0.380** | **0.093** | **0.071** |

## View 2: Cross-type-only ground truth (the direct head-to-head)

| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| Raw SigLIP (baseline) | 0.076 | 0.112 | 0.019 | 0.016 |
| Phase 8 Model A (Amazon-trained, random negs) | 0.078 | 0.107 | 0.020 | 0.016 |
| Phase 9 Model A (Polyvore-trained, naive, random negs) | 0.046 | 0.074 | 0.012 | 0.011 |
| **Phase 10 color-invariant model** | **0.047** | **0.067** | **0.012** | **0.010** |


# Phase 10, Step 4.3: Direct Color-Reliance Diagnostic

Pearson correlation between each model's predicted compatibility score (cosine similarity, projected for trained models / raw for SigLIP) and raw color similarity (HSV hue/saturation 2D histogram intersection) across sampled item pairs. Lower = less reliant on color matching as a proxy for compatibility.

- Polyvore domain: 3000 pairs (real test-split outfit co-occurrences, randomly sampled from all such pairs, capped at 3000).
- Amazon domain: 6605 pairs (all also_buy edges within the untouched eval sample, both endpoints present, deduplicated).

| Configuration | Polyvore-domain r | Amazon-domain r | Combined r |
|---|---|---|---|
| Raw SigLIP | 0.053 | 0.391 | 0.317 |
| Phase 8 Model A (Amazon-trained) | -0.039 | 0.219 | 0.127 |
| Phase 9 Model A (Polyvore-trained, naive) | 0.181 | 0.198 | 0.153 |
| Phase 10 color-invariant model | 0.180 | 0.192 | 0.151 |

