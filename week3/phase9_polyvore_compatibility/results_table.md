# Phase 9, Step 5.1: Official Polyvore Test-Split Evaluation

Literature anchor (Vasileva et al. ECCV'18, full type-aware trained network, NOT directly comparable to this phase's frozen-SigLIP+small-MLP setup): AUC=0.88, FITB accuracy=0.576.

| Configuration | Compatibility AUC | FITB Accuracy |
|---|---|---|
| Raw SigLIP (reference) | 0.7172 | 0.4843 |
| Model A (random negs) | 0.9469 | 0.7031 |
| Model B (hard negs) | 0.9373 | 0.6840 |

(Compatibility test: n=20000 lines scored; FITB test: n=10000 questions scored.)


# Phase 9, Step 5.2: Transfer Test on the Untouched Amazon Eval Sample

Sanity check: recomputed raw SigLIP metrics matched the existing reported baseline within tolerance -- confirms the untouched Amazon eval sample and eval methodology are unchanged before trusting the transfer-test rows below.

Cross-type-only ground truth: 1691 edges across 1732 queries (140 excluded, unknown own type) -- identical to phase 8's view.

## View 1: Full also_buy ground truth

| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| Raw SigLIP (baseline) | 0.502 | 0.578 | 0.191 | 0.144 |
| Phase 8 Model A (Amazon-trained, random negs) | 0.441 | 0.523 | 0.160 | 0.120 |
| Phase 8 Model B (Amazon-trained, hard negs) | 0.362 | 0.440 | 0.117 | 0.089 |
| Phase 9 Model A (Polyvore-trained, random negs) | 0.315 | 0.384 | 0.098 | 0.073 |
| Phase 9 Model B (Polyvore-trained, hard negs) | 0.261 | 0.323 | 0.078 | 0.056 |

## View 2: Cross-type-only ground truth (the direct head-to-head)

| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| Raw SigLIP (baseline) | 0.076 | 0.112 | 0.019 | 0.016 |
| Phase 8 Model A (Amazon-trained, random negs) | 0.078 | 0.107 | 0.020 | 0.016 |
| Phase 8 Model B (Amazon-trained, hard negs) | 0.059 | 0.094 | 0.014 | 0.012 |
| Phase 9 Model A (Polyvore-trained, random negs) | 0.046 | 0.074 | 0.012 | 0.011 |
| Phase 9 Model B (Polyvore-trained, hard negs) | 0.038 | 0.054 | 0.010 | 0.008 |

