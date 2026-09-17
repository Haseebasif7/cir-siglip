# Phase 13b: Architecture Notes -- What Changed From Phase 13, What Didn't

## What stayed exactly the same (copied unchanged from `phase13_csa_net_baseline/scripts/model.py`)

- 5 subspaces (`NUM_SUBSPACES = 5`), embedding size 64, margin 0.3.
- The category-pair attention sub-network: `Linear(22, 64) -> ReLU -> Linear(64, 5) -> softmax`,
  taking the concatenation of two one-hot category vectors (same 11-category
  vocabulary phases 9/12/13 already established).
- The masking + weighted-sum mechanism: `f = sum_i (x ⊙ m_i) * w_i` (eq. 1).
- L2-normalization of the final embedding before distance computation (phase
  13's fix for magnitude collapse).
- The outfit ranking loss (margin hinge, `min` aggregation over negatives)
  and the uniformity regularizer (phase 13's fix for direction collapse).
- Near-identity mask initialization (phase 13's fix for dead gradients).
- The two eval-time embedding-caching helpers (`all_category_embeddings_from_feature`
  / `all_as_candidate_embeddings_from_feature`) that let a candidate/context
  item's category-conditioned embeddings be computed once and reused across
  every query it appears in -- unchanged, since this logic operates entirely
  on the post-backbone feature `x` and has nothing to do with how `x` itself
  was produced.
- The training data (outfit lists, leave-one-out sample construction,
  category vocabulary) and the mined negative-candidate lists -- reused
  DIRECTLY from `phase13_csa_net_baseline/data/{training_data.json,
  negative_candidates.json}`, not regenerated. Both are entirely
  backbone-independent (outfit membership, item categories, and even the
  negative mining itself was already done using phase 9's SigLIP embeddings
  -- see phase 13's `02_mine_negative_candidates.py` -- so it's actually a
  more natural fit for this phase than it was for phase 13's own
  ResNet18-backbone run).
- The CIR evaluation harness and scoring adaptation (phase 12's benchmark,
  phase 13's per-context-item average-pairwise-distance scoring) -- reused
  directly, only pointed at this phase's own features/checkpoint.

## What changed

**Only the base feature extractor** (`encode_image` in phase 13 ->
`encode_feature` here):

| | Phase 13 | Phase 13b |
|---|---|---|
| Input | Raw JPEG image, `(3, 224, 224)` | Phase 9's precomputed SigLIP embedding, `(768,)` |
| Backbone | ResNet18, ImageNet-pretrained, **fine-tuned end-to-end** | **None** -- SigLIP embedding is looked up, never recomputed or updated |
| Projection | `Linear(512, 64)` | `Linear(768, 64)` |
| Trainable params in the "backbone" role | ~11M (full ResNet18) + projection | 0 (frozen lookup) + projection only |
| Compute per training step | Full CNN forward+backward over ~1,500 unique images | A `(1500, 768) -> (1500, 64)` matrix multiply -- negligible |

Everything downstream of that one swapped function is byte-for-byte the
same code path. No image loading, no gradient-accumulation/micro-batching
(phase 13 needed that specifically to avoid OOM on ~1,500 images through
ResNet18 with gradients -- irrelevant here, a full logical batch of 96
outfits' worth of SigLIP lookups fits trivially in memory), no
freeze/unfreeze schedule (there's no backbone to freeze -- the SigLIP
embedding was never part of this model's trainable parameters in the first
place, unlike phase 13 where freezing was an explicit, temporary
intervention on an otherwise-trainable ResNet18).

## Why this is the more meaningful comparison for this project's own argument

This project's controllable-mode mechanism (phases 12-12d) is also built on
frozen SigLIP. Comparing it against CSA-Net's conditioning mechanism with
the backbone held constant on both sides isolates the actual variable this
project's research argument is about (conditioning/steering mechanism), free
of the confound "well, CSA-Net also gets to use a different, possibly
better-suited backbone." See `phase13b_notes.md` for that comparison.
