# Phase 15: Architecture Notes

## What this phase extends

Phase 13b's `CSANetSigLIP` (`week4/phase13b_csa_net_siglip_backbone/scripts/model.py`)
implements CSA-Net's category-pair-conditioned subspace attention on a frozen
SigLIP backbone: `attn_net` (`Linear(22,64)->ReLU->Linear(64,5)`) takes a
concatenated `(cat_s_onehot, cat_t_onehot)` pair and outputs softmax weights
over 5 subspaces; a masked, weighted sum of those subspaces (each subspace =
elementwise mask applied to the 64-d projected feature) produces the final
L2-normalized embedding.

This phase (`CSANetSigLIPControllable` in `scripts/model.py`) extends
`attn_net`'s input to `2*num_categories + 1 = 23` dims by concatenating the
substitute/complement control scalar `alpha` (0 = complement-leaning, 1 =
substitute-leaning) alongside the two category one-hots. Everything else --
masks (near-identity init), L2-normalization, `pairwise_distance`,
`outfit_ranking_loss` (margin=0.3, `min` aggregation), `uniformity_loss`
(Wang & Isola 2020, weight 1.0) -- is copied unchanged from phase 13b.

## Design decisions

**1. Alpha representation: raw scalar concatenation.** `attn_net`'s input
width becomes 23 (`2*11 category one-hots + 1 raw alpha`), matching CSA-Net's
existing convention of feeding raw features into a tiny 2-layer subnet with
no embedding step. Not revisited with a learned embedding this phase -- see
"Named risk" below for why a richer encoding would not have addressed the
problem actually found.

**2. Same-category restriction for substitute training pairs.** CSA-Net's
`embed_from_feature` requires a `(cat_s, cat_t)` pair -- there is no
category-free embedding mode. Phase 12c's `nn_lookup.npz` (top-50 raw-SigLIP
neighbors per item) is not category-restricted. Since the CIR eval task
always retrieves within a single category pool, each anchor's neighbor list
is filtered to same-category-only (`00_build_same_category_neighbors.py`)
before use, and `cat_s = cat_t = anchor's own category` for both anchor and
neighbor embeddings in the substitute loss. This deliberately avoids phase
14's documented train/eval negative-distribution mismatch. Preprocessing
found this restriction is cheap: same-category neighbors dominate the raw
top-50 for nearly every item (median = 50/50 same-category in 8 of 11
categories; only scarves, the smallest category, sees meaningful attrition --
10.84% of items fall below the MIN_NEIGHBORS=5 threshold and are excluded
from substitute sampling; overall only 0.90% of the catalog is excluded). See
`data/neighbor_preprocessing_report.md` for the full per-category table.

**3. Batch/alpha sampling granularity.** One alpha per training step
(`alpha ~ Uniform(0,1)` for the continuous run, `alpha ~ {0,1}` for the
discrete ablation), shared by both the complement sub-batch (CSA-Net
leave-one-out outfit samples) and the substitute sub-batch (anchors reused
from the same complement batch's positive items, looked up against their
same-category-filtered neighbors) -- mirrors phase 12c's own convention of
reusing one batch's anchors for both losses.

**4. Loss combination**: `total_loss = alpha * weight_sub * substitute_loss
+ (1-alpha) * complement_loss + UNIFORMITY_WEIGHT * uniformity_loss`.

## Named risk: the alpha double-duty confound

Alpha plays two roles: the conditioning input fed into `attn_net`, and the
outer weight on each loss term. At alpha->0 the substitute term's gradient
is scaled toward zero by the envelope regardless of what `attn_net` does
with alpha as an input (symmetrically for complement at alpha->1) -- the
network can satisfy the loss at each endpoint without `attn_net` ever
learning to respond to alpha at all. A richer alpha encoding (decision 1's
learned-embedding fallback) would not fix this, since the cause is a
gradient-signal problem, not an encoding-capacity problem.

**Safeguards run (`02_smoke_test.py`, see `training_log_smoke_test.md` for
full numbers):**
- *Conditioning-consistency check*: `decoupled_alpha_frac=0.2` runs 20% of
  smoke-test steps with the forward-pass alpha sampled independently from
  the loss-weighting alpha, keeping the conditioning pathway exercised near
  the envelope's own endpoints. Training remained stable under this
  (`train_core.py`'s `compute_batch_loss` takes `alpha_forward` and
  `alpha_weight` as separate arguments throughout, specifically to make this
  possible).
- *Attention-weight-shift probe* (`model.attention_weight_shift`): for a
  fixed sample of all 121 category pairs, sweep the forward-pass alpha alone
  from 0 to 1 (no loss) and measure the mean L1 distance between the
  resulting attention-weight vectors. Untrained baseline: 0.0405. After 5
  smoke-test epochs (2,000 outfits): 0.0242 (no uniformity), 0.0269 (with
  uniformity), 0.0282 (with 20% decoupled alpha) -- **all at or below the
  untrained baseline**, i.e. no clear positive signal yet that `attn_net` is
  learning to use alpha at this small scale. This is not yet a verdict (the
  smoke test is deliberately tiny and undertrained), but it is exactly the
  pattern the confound predicts, and the same probe is re-run on both real
  trained checkpoints before any claim is made in `phase15_notes.md` about
  whether the richer conditioning mechanism is actually realized.

## Alpha double-duty confound: full-scale finding (after real training)

Re-running the attention-weight-shift probe on the actual trained continuous
checkpoint (`04_train_full.py`'s output, 60 epochs to convergence) gives a
different picture than the smoke test: mean L1 shift = **0.1881** (vs. 0.0405
untrained, ~4.6x), with substantial variance across the 121 category pairs
(max 0.871, min ~0). **`attn_net` did learn a real, non-trivial response to
alpha for at least some category pairs** -- the confound did not fully
suppress conditioning, contrary to what the smoke test alone suggested.

However, the CIR alpha sweep (`03_cir_eval_sweep.py`, `data/cir_sweep_continuous.json`)
shows this learned response barely moves aggregate retrieval behavior:
Recall@10 moves only 0.0611 -> 0.0596 (alpha 0 -> 1), axis1 (visual
similarity) 0.7030 -> 0.7011, axis2 (co-occurrence hit rate) 0.0580 -> 0.0600
-- all far smaller swings than phase 12c/12d's simple additive mechanism
achieved with the same backbone (axis1 0.72 -> 0.75, axis2 0.10 -> 0.076,
see `week4/phase12d_alpha_sweep/results_table.md`). The adjacent-vs-distant
top-K overlap smoothness check (`07_smoothness_check.py`, phase 12d's own
methodology) confirms this quantitatively: overlap decays smoothly and
monotonically as |delta alpha| grows (0.993 at delta=0.1 down to 0.942 at
delta=1.0), but the **gap is only 0.051** -- barely above phase 12d's
"partial confirmation" threshold (0.05) and far below its "real, smooth
relationship confirmed" threshold (0.15) that phase 12c/12d's own simpler
mechanism cleared by more than 3x (gap 0.475).

**Reconciling these two findings**: `attn_net` learned SOME real,
category-pair-specific sensitivity to alpha (the shift probe), but that
sensitivity is concentrated unevenly (a few category pairs shift a lot, most
shift little to none -- the wide 0-0.871 per-pair range), so its effect on
AGGREGATE retrieval behavior averaged across the whole benchmark is modest.
This is a genuine, quantified finding about this architectural extension,
not a training bug: the alpha double-duty confound partially, not fully,
suppressed the intended conditioning mechanism. See `phase15_notes.md` for
the full interpretation against phase 12c/13b.

## Collapse check (`02_smoke_test.py`)

Mean pairwise candidate-embedding cosine similarity (fixed category pair,
alpha=0.5, 2,000-item sample), 5 epochs / 2,000 outfits:
- WITHOUT uniformity regularizer: 0.7565
- WITH uniformity regularizer (weight=1.0): 0.4039

A real, if less extreme than phase 13's original 0.974, collapse signal
without the regularizer -- consistent with phase 13/13b/14's generic
direction-collapse failure mode resurfacing here. Uniformity regularizer
enabled from the start of both real training runs (matches phases 13b/14's
proactive convention).

## Loss balancing (`loss_balancing_check.md`, step 3 / design decision 6)

`weight_sub` calibrated at alpha=0.5 only (`initial_comp=0.3600,
initial_sub=0.0376 -> weight_sub=9.5801`). Gradient-norm verification at
alpha in {0.2, 0.5, 0.8}, isolating each RAW loss term directly (bypassing
the uniformity regularizer, which couples both branches together via a
shared representative-embedding set and would otherwise contaminate the
isolation): target ratio ~1.0 within a declared 3x tolerance. Passed at
alpha=0.5 (2.43x) and alpha=0.8 (2.66x); **failed at alpha=0.2 (3.75x)** --
the raw gradient-scale ratio is itself somewhat alpha-dependent, and a
single fixed scalar does not fully correct for it there. Per the pre-declared
fallback, training proceeds with this weight regardless (no alpha-dependent
weight introduced, since that was declared a fallback only if the check
demonstrated it was needed) -- see `phase15_notes.md` for how this is
accounted for in the final interpretation.
