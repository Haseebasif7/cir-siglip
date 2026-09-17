# Phase 35: Implementation Notes

Per the brief's own instruction, the paper (arXiv 2204.04812, "OutfitTransformer: Learning Outfit
Representations for Fashion Recommendation") was read directly (via its HTML rendering), not recalled from
memory or earlier phases' summaries. This document quotes what the paper actually says for each of the
three pieces, states what's clear vs. genuinely ambiguous, and records the adaptation made in each case
where this project's own existing architecture/data conventions (SigLIP frozen backbone, 11-category
system, cascaded-fallback text pipeline) required a documented deviation from a literal reading.

## What stays held constant, and why

This phase's central comparison is phase 35 (full recipe) vs. phase 32 (same architecture, same backbone,
no recipe) -- isolating the training recipe's contribution alone. Everything phase 31/32 already tuned
(`d_model=128, d_embed=64, n_heads=8, n_layers=4, d_ffn=512, dropout=0.1, margin=0.2, uniformity_weight=0.1,
input_mode=image_text, lr=1.5e-4, batch_size=384, max_epochs=100, patience=25, num_negatives=10`) is held
byte-identical in stage 2. Only three things are new: CP pre-training (stage 1, entirely new), the
target-category token (piece 2), and curriculum negative sampling replacing stage 2's negatives (piece 3).
The loss formula itself (triplet hinge on the hardest negative + uniformity regularizer) is deliberately
**not** changed to the paper's own `L_All + L_Hard` combination (margin=2) -- the brief names three specific
pieces to add, and the loss shape isn't one of them; changing it too would confound exactly the isolation
this phase's central comparison depends on. This mirrors phase 34's own single-variable discipline.

## Piece 1: compatibility prediction (CP) pre-training

**What the paper says** (section 3.1): an "outfit token" is prepended to the item feature set; "the state of
the outfit token at the output of the transformer encoder serves as the global outfit representation," fed
to an MLP producing a compatibility score in [0,1], trained with **focal loss**. The paper does not specify
how negative (incompatible) outfits are constructed for this task at all.

**What was implemented**: exactly the brief's own explicit instruction, since the paper is silent here --
"negative outfits (items shuffled across different real outfits to create fake combinations)." Concretely
(`train_lib.py`'s `CPState`): a fake outfit of size k draws k items, each from an independently, uniformly
chosen *different* real training outfit -- drawn fresh every epoch, not fixed once. Real outfits (label 1)
and their fake counterparts (label 0) are batched 1:1 (a batch of 192 outfits yields 384 sequences/step,
matching stage 2's per-step sequence count).

**MLP head**: the paper says "MLP" without specifying depth; a single `Linear(d_model, 1)` (`cp_head`) was
used -- the simplest reading, and consistent with this project's own established preference (elsewhere in
this project, e.g. phase 23) for the simplest architecture that satisfies a stated requirement, reserving
complexity for when a simpler version is shown insufficient.

**Focal loss parameters**: the paper names "focal loss" (Lin et al. 2017) without stating gamma/alpha.
Implemented with that paper's own original defaults, `gamma=2.0, alpha=0.25` -- the standard values, not
something this project introduced, and documented as an assumption per the brief's own instruction.

**Outfit-token role split**: `self.outfit_token` (unchanged from phase 14b/31/32) is reused as CP's own lead
token, since the paper explicitly describes CP's "global outfit representation" reader as *the* outfit
token -- the same one phase 14b's retrieval query used. In stage 2, the outfit token is retired from the
query role (replaced by the target-category token, piece 2) but its CP-pretrained weights are not otherwise
touched or reused; only `proj` and `set_enc` (the shared item-encoding backbone) are warm-started into stage
2, per the paper's own statement: "we pre-train the framework ... and use the learned weights to initialize
the transformer, image and text encoder ... for complementary item retrieval." `embed_ffn` (stage 2's new
output head) and `cp_head` (stage-1-only) are never shared between stages.

## Piece 2: target-category token conditioning

**What the paper says** (section 3.2): the target token is *not* a learned per-category embedding. It is
constructed as `s = x_Img || E_text(T)`, where `x_Img` is "an empty image" placeholder and `E_text(T)`
encodes the target's text description via the same text encoder used for items. The transformer then takes
"the set of feature vectors F of the partial outfit, and the target item specification s," and the query
embedding `t = MLP(E_trans(s, F))` is read from `s`'s own output position, not a separate outfit token.

**What was implemented**: this is unambiguous in the paper, so the brief's alternative ("learned embedding")
was not used. Adapted to this project's existing SigLIP image+text concat convention (`input_mode=image_text`,
768+768=1536-d per item): for each of the 11 categories (the exact vocabulary and text embeddings already
built in phase 27 -- `week7/phase27_text_and_category/data/category_text_embeddings.npz`, raw SigLIP text
features for the 11 category name strings, no phrase template), the target-category "item" vector is
`concat(zeros(768), category_text_embedding)` -- the same "empty image, real text" construction the paper
describes, expressed in this project's own 1536-d item-vector shape. This vector is then run through the
*exact same* `encode_item_tokens` (normalize -> `proj` -> normalize) as every real item, so it requires **no
new learnable parameters** -- consistent with the paper's own derivation-based (not learned-table) design.
`embed_query_targeted` (model.py) replaces the outfit token with this category token as the lead position,
mirroring the paper's `E_trans(s, F)` exactly; `embed_ffn` then reads out position 0, same as before.

One real limitation, stated honestly: because `proj` is *trainable* during stage 2 (not frozen after stage
1), the category token's own encoded form shifts over the course of training along with every item's --
this is expected and correct (it's not supposed to be a fixed target, it's supposed to co-adapt with the
retrieval objective), but it means the 11 category tokens are recomputed fresh every forward pass
(`category_tokens_all`, called on 11 rows only -- negligible cost) rather than cached.

## Piece 3: curriculum negative sampling

**What the paper says** (section 3.2.2): "In the first stage, we sample the negatives from the same
high-level category as the positive item. Subsequently in the second stage, we sample harder negatives from
more fine-grained categories." Hard negatives are the minimum-distance candidate among a randomly sampled
pool (`min_j d(t, f_j^N)`). Ten negatives per outfit are used. No exact epoch schedule or transition formula
is given.

**Genuine ambiguity #1 (documented, not resolved by guessing)**: this project's dataset uses a single
11-category vocabulary (`accessories, all-body, bags, bottoms, hats, jewellery, outerwear, scarves, shoes,
sunglasses, tops`) with no finer sub-category level to distinguish "high-level" from "fine-grained" --
Polyvore's own metadata doesn't carry a second granularity. The paper's exact two-level distinction cannot
be reproduced literally. **Adaptation**: mapped the paper's easy-to-hard *intent* onto this project's own,
already-established, empirically distinct negative-sampling schemes instead: "easy" = uniform-random
same-category negatives (this project's own repeatedly-validated best-performing scheme, phases 7-9/13c/14b
run2/34); "hard" = the pre-mined, same-category, visually-similar-by-raw-SigLIP-cosine candidate lists
already built in phase 13
(`week4/phase13_csa_net_baseline/data/negative_candidates.json`, read-only reuse, not rebuilt) -- each
item's own top-20 same-category nearest neighbors by frozen SigLIP cosine similarity, which is exactly the
kind of "harder, more visually similar" negative the paper's second stage describes, just without a second
category granularity to draw it from.

**Genuine ambiguity #2 (schedule)**: no epoch numbers are given. Per the brief's own suggested default,
implemented as a **linear ramp**: `hard_fraction(epoch) = min(1, epoch / 99)` over stage 2's 100-epoch
budget -- 0% hard (pure random) at epoch 0, 100% hard (pure mined) at the final epoch. Each of the 10
negative slots per sample is filled by drawing `round(10 * hard_fraction)` from the mined pool and the
remainder from the random pool (`RetrievalTrainState.sample_negatives`, `train_lib.py`); if the mined pool
for a given target item is missing or exhausted, the shortfall is topped up with random negatives rather
than failing the sample -- so `hard_fraction` is a target proportion, not a hard guarantee, documented here
rather than silently absorbed.

**On this project's own prior finding**: this project has independently found, five separate times, that
mined hard negatives *underperform* random same-category negatives as the *sole* training signal (phases
7-9, 13c, 14b run1-vs-run2, phase 34). The brief is explicit that curriculum learning is a structurally
different use of hard negatives (introduced gradually, after the model has already learned from easier
examples, never used exclusively from the start) and instructs this not be assumed to inherit that earlier
finding automatically. `stage2_training_log.md` reports what actually happened as `hard_fraction` rises
across training, honestly, rather than assuming either outcome in advance.

## Everything else: unchanged from phase 32

`encode_item_tokens`, `embed_item_alone`, the transformer set-encoder shape, the triplet+uniformity loss
formula, the AdamW+OneCycleLR schedule shape, and the category-restricted-random negative sampler's
vectorized implementation are all copied verbatim from
`week7/phase32_partial_ensemble_outfittransformer/scripts/model.py` and its `modal_app.py`'s training loop,
with only the additions described above layered on top. `week4/` is read-only reused (training_data.json,
negative_candidates.json, siglip_base.npz) and never written to.

## Execution: local, not Modal

Per the brief's "run locally first" instruction, timing was measured directly before committing to a full
run (M4, MPS with CPU fallback for `torch.cdist`'s backward, which has no MPS kernel as of torch 2.6):
stage 1 (CP, batch=192 outfits/384 sequences) measured at ~246ms/step, ~68s/epoch; stage 2 (retrieval,
batch=384) measured at ~276ms/step, ~38s/epoch; validation-benchmark eval ~7.5s. At these rates the full
budget (stage 1: up to 40 epochs; stage 2: up to 100 epochs) comes to well under 2.5 hours total wall time
-- comfortably practical locally, no Modal budget spent on this phase. See `stage1_training_log.md` and
`stage2_training_log.md` for the actual measured wall-clock once both runs completed.
