# Phase 12: Controllable Mode Embeddings, a Cheap First Test

## Context

Phases 5-11 (see `PROJECT_MEMORY.md` for the full arc) investigated whether a learned
"complementary item" signal could improve on frozen SigLIP for Amazon recommendations,
culminating in phase 11's blended raw-SigLIP + Amazon-trained-compatibility retrieval
configuration. After discussing results with the professor, the project pivoted at this
phase toward a specific, well-defined research target: **Complementary Item Retrieval
(CIR)**, the standard task from CSA-Net (CVPR 2020) / OutfitTransformer (WACV 2023),
evaluated on Polyvore Outfits. The identified angle for a genuine contribution (not just
a stronger backbone): a single model whose retrieval behavior is **steerable at inference
time** between visually-similar-substitute and genuinely-different-but-compatible-
complement behavior, rather than needing two separately trained models blended together
the way phase 11 did.

This phase is the cheap, fast proof-of-concept the brief asked for, explicitly before
committing to the bigger investment (extending CSA-Net's own category-pair conditioning
mechanism directly): can a simple additive "mode vector" actually steer retrieval
behavior in the expected direction?

## Step 1: building the CIR evaluation harness

Full detail in `cir_protocol_notes.md`. Summary: investigated
`github.com/bigohofone/outfit-transformer` as instructed and found it real, but its "CIR"
evaluation script is 4-way FITB accuracy, not the candidate-pool Recall@K retrieval task
described in the brief -- no usable reference implementation of that protocol was found in
the repo's accessible source, and its checkpoint uses a different embedding backbone
(CLIP) and dataset pipeline than this project's. Per the brief's own fallback instruction,
did not sink time into running that checkpoint; built the harness independently instead,
leakage-safe by direct construction (queries and pools built exclusively from Polyvore's
test split, category-restricted candidate pools capped at 3,000, disjoint from
train/valid). One real bug caught and fixed during this step: an initial version
force-added every query's target into its category's pool, which silently defeated the
3,000 cap for popular categories (shoes ballooned 3,000 -> 8,551) -- fixed by sampling the
pool first and dropping queries whose target missed the sample, rather than growing the
pool to compensate. Final benchmark: **29,681 leave-one-out queries** across 11
categories, capped candidate pools (`data/cir_benchmark.json`,
`data/cir_benchmark_coverage.md`).

## Step 2: baseline numbers, before building anything new

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Raw SigLIP (alone) | 0.0553 | 0.1067 | 0.1437 |
| Phase 9 Model A (Polyvore-trained, alone) | 0.1317 | 0.2464 | 0.3216 |
| OutfitTransformer (literature anchor, not independently reproduced) | 0.0958 | 0.1796 | 0.2198 |

Notable, unplanned finding: **phase 9's existing Polyvore-trained compatibility model,
reused with zero new training, beats the OutfitTransformer literature anchor on this
project's own harness** (0.1317 vs 0.0958 @10; 0.2464 vs 0.1796 @30; 0.3216 vs 0.2198
@50). This is *not* a claim that phase 9's model beats OutfitTransformer in general -- the
two numbers come from different candidate-pool constructions (see the caveat in
`results_table.md`), so only relative rankings measured on this project's own harness are
a clean comparison. But it's a genuinely reassuring internal sanity check: phase 9's
model, already known to dramatically beat raw SigLIP on Polyvore's official AUC/FITB
benchmark (phase 9: 0.717->0.947 AUC), also dramatically beats raw SigLIP here (0.055 ->
0.132 @10, roughly 2.4x) on a completely different, independently-built retrieval
protocol -- the in-domain compatibility signal phase 9 found is real and transfers across
evaluation methodologies within Polyvore's own domain, even if (per phase 9's own
finding) it does not transfer to Amazon's domain.

## Step 3: training the controllable mode-embedding mechanism

Architecture and training procedure: `scripts/model.py` (`ControllableProjectionHead`)
and `scripts/03_train_controllable_modes.py` -- full design rationale in that script's
own docstring. In short: one shared 768->256->128 MLP (same shape as phase 7/8/9's
`ProjectionHead`) plus two learnable 128-d mode vectors, added to the shared projection's
output before final L2-normalization, interpolatable via a scalar alpha (1.0=substitute,
0.0=complement). Trained jointly: every step applies both a complement loss (MNRL /
InfoNCE against real Polyvore co-outfit pairs, reusing phase 9's exact positive edges,
random-negatives-only hyperparameters) and a substitute loss (MSE distillation of the
batch's pairwise raw-SigLIP cosine similarity) to the same shared batch, backpropagating
both into the shared net.

Training converged cleanly (best checkpoint at epoch 0, early-stopped at epoch 5,
`models/training_curves.json`, `logs/step03_train_run.log`) -- same "best epoch is epoch
0" pattern phase 9 found at this edge-count scale, not a new phenomenon. No collapse
(mean pairwise cosine stayed in the 0.55-0.65 range across both modes throughout
training, well short of the near-1.0 collapse signature). Substitute-mode's distillation
loss dropped from 0.0148 to ~0.0107-0.0111 and stayed there; complement-mode's MNRL loss
dropped from 4.39 to ~4.05-4.28 (train) with a rising validation loss after epoch 0
(5.34 -> 6.00), the same early-overfitting pattern phase 9's own models showed at this
edge-count scale.

## Step 4: does the control knob actually work?

### 4.1 -- CIR benchmark Recall@K (full table: `results_table.md`)

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Raw SigLIP (alone) | 0.0553 | 0.1067 | 0.1437 |
| Phase 9 Model A (alone) | 0.1317 | 0.2464 | 0.3216 |
| **Substitute mode (alpha=1.0)** | **0.1329** | **0.2467** | **0.3215** |
| **Complement mode (alpha=0.0)** | **0.1335** | **0.2507** | **0.3250** |
| **Blend (alpha=0.5)** | **0.1366** | **0.2524** | **0.3289** |

The first red flag, visible before running any diagnostic: **substitute mode's Recall@K is
nearly identical to complement mode's, and both are close to phase 9 Model A's** -- not
close to raw SigLIP's, which is dramatically lower (0.055 vs 0.133 @10). If substitute
mode genuinely reproduced "plain visual similarity" retrieval behavior, its Recall@K
should track raw SigLIP's much lower numbers, not the compatibility-trained numbers. It
doesn't. This alone is strong evidence the mechanism did not achieve real separation
before even running the dedicated diagnostic.

### 4.2 -- direct control-effectiveness diagnostic (full detail: `control_effectiveness.md`)

On a 1,000-query sample, comparing substitute-mode's and complement-mode's own top-10
lists directly (same checkpoint, two different alpha values):

- **Axis 1 (should-be-more-visually-similar): confirmed, but weakly.** Substitute mode's
  top-10 averages 0.7193 raw-SigLIP similarity to the query vs. complement mode's 0.7120
  -- correct direction, ~1% relative gap.
- **Axis 2 (should-better-match-real-co-occurrence): not confirmed at all.** Substitute
  and complement mode hit the true held-out target at the *exact same rate*: 128/1000
  each. Not "not significantly different" -- identical.

Followed this up directly (not asked for by the brief, but necessary to explain the
exact tie rather than just report it): **mean top-10 overlap between substitute mode's
and complement mode's retrieved lists is 76.4%** (median 80%, n=500 queries, every query
checked) and **the average per-item cosine similarity between an item's substitute-mode
and complement-mode projection is 0.829**. The two "modes" are largely retrieving the
same items. Qualitative grids (`qualitative_examples/`, 6 categories) confirm this
visually -- in every example inspected, the substitute/complement/blend rows show the
same handful of items, just lightly reordered, not qualitatively different retrieval
character the way phase 11's raw-SigLIP-vs-phase-9-alone comparison showed clear,
visible character differences.

**Root cause, checked directly from the trained checkpoint**: the shared base
projection's own output norm averages 2.356; the learned `mode_substitute` vector has
norm 1.186 and `mode_complement` has norm 0.256 -- both smaller than the base projection
they're added to, complement's especially so (~9x smaller). The two loss terms are
summed unweighted every step, but they differ in raw magnitude by roughly 400x
(complement's MNRL loss sits around 4.0-4.3; substitute's MSE distillation loss sits
around 0.011). Since both losses backpropagate into the *same shared* `net`, the shared
representation ends up almost entirely shaped by the complement objective's much larger
gradient signal -- the substitute loss gets absorbed mostly by the small
`mode_substitute` offset rather than by reshaping the underlying geometry, and that
offset is not large enough relative to the base projection to meaningfully redirect
retrieval rankings. A secondary, compounding factor: the substitute loss only
constrains pairwise similarity *within each training batch* (~128 anchor items), a much
narrower target than reproducing raw SigLIP's actual global nearest-neighbor structure
across the full 251,008-item catalog needed for retrieval-quality behavior.

## Step 5: qualitative check

6 example queries across 6 categories (bags, jewellery, tops, bottoms, shoes,
outerwear), `qualitative_examples/`. Every grid shows the same pattern documented above:
substitute, complement, and blend rows retrieve near-identical top-5 sets, differing
mostly in ranking order and occasionally in the 4th/5th slot. This is not a case of
subtle numeric differences hiding a real qualitative distinction -- the images
themselves look interchangeable across modes in every example checked. None of the 6
examples' true target happened to land in the top-5 under any mode (consistent with the
~13% Recall@10, most individual queries miss).

## Interpretation

The control mechanism, as built here, **does not work** -- not because the underlying
idea (a single model whose retrieval behavior can be steered at inference time) is
wrong, but because this specific cheap implementation let one loss term's much larger
raw magnitude dominate the shared representation, leaving the mode vectors too weak to
meaningfully separate the two modes' actual retrieval rankings. This is a genuine,
well-evidenced negative result on three independent lines of evidence (aggregate
Recall@K tracking complement/Model A rather than raw SigLIP, the control-effectiveness
diagnostic's exact-tie axis and the follow-up 76% overlap/0.829 per-item-cosine numbers
that explain it, and the qualitative grids showing visually interchangeable results) --
not a single ambiguous number.

Critically, though, the root cause identified is **specific and cheaply fixable**, not a
fundamental objection to conditioning-based control. The failure traces to an
identifiable engineering choice (summing two losses of wildly different scale
unweighted, plus a purely additive conditioning mechanism) rather than to "additive mode
vectors on a shared projection can never separate retrieval behavior." That distinction
matters directly for the brief's own question.

## Recommendation

**Do not commit yet to the larger investment (extending CSA-Net's own category-pair
conditioning mechanism) based on this result alone -- but also do not conclude
controllable retrieval is a dead end.** This cheap version failed for a reason this phase
pinned down precisely (loss-scale imbalance overwhelming a too-weak additive
perturbation, compounded by the substitute loss's narrow in-batch scope), not for a
reason that indicts the broader idea. Two candidate next steps, in order of cost:

1. **Cheapest, try first**: re-run this exact same additive-mode-vector architecture
   with the loss imbalance directly fixed -- e.g. weight the substitute loss by roughly
   the ratio of the two losses' starting magnitudes (~400x), or normalize both losses
   before summing, or alternate mode-specific training steps instead of joint summing.
   This is a small change to `03_train_controllable_modes.py`, reuses everything already
   built in this phase (the CIR harness, the evaluation/diagnostic scripts), and would
   directly test whether the *idea* (shared projection + additive mode vectors) can work
   once the identified confound is removed -- before spending the much larger effort of
   building CSA-Net-style conditioning.
2. **If step 1 also fails to separate the two modes**, that would be much stronger
   evidence that purely additive conditioning on a shared projection is the wrong
   mechanism (not just under-weighted), and would make a real case for the bigger
   investment: CSA-Net's own conditioning approach (learned per-category-pair projection
   subspaces / attention-based gating, not a single additive offset) is a structurally
   stronger mechanism specifically at the place this phase's version was shown to be
   weak.

Either way, this phase's deliverable -- the CIR evaluation harness
(`cir_protocol_notes.md`, `scripts/01_build_cir_benchmark.py`, `scripts/cir_eval.py`) and
its baseline numbers (phase 9's model beating the literature anchor on this project's own
harness) -- is reusable as-is for whichever path comes next, and did not require the
control mechanism to work in order to be valuable on its own.
