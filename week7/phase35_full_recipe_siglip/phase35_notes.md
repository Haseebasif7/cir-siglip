# Phase 35: OutfitTransformer's Full Training Recipe -- Honest Result

## The central finding, stated plainly

**No, at the hyperparameters this phase was required to hold constant, OutfitTransformer's full published
training recipe (CP pre-training + target-category conditioning + curriculum negative sampling) does not
produce something stronger than this project's own simpler, already-strengthened recipe (phase 32) on the
same frozen SigLIP backbone. It produces something clearly weaker.**

Phase 35's single model: test Recall@10/30/50 = **0.1311 / 0.2380 / 0.3075**. The matched, recipe-isolating
comparison -- phase 32's solo seed-42 model, same architecture, same backbone, same hyperparameters, no
recipe -- scores **0.1799 / 0.3111 / 0.3844**. Phase 35 trails it by **-27.1% / -23.5% / -20.0%** relative at
every K, and does not beat it once. This is not a boundary case; it's a clear, unambiguous result in the
opposite direction from what the recipe's own published motivation would predict.

## Why this is a real, useful negative result, not a failed experiment

The brief asked the cleanest possible question: does the full recipe help, holding everything else fixed?
It does not overstate to say the answer is a clean, unambiguous no under this test's own conditions. That
answer is itself informative for this project's report -- it says the earlier investment-parity work (phases
31/32) had already captured most or all of what a strengthened OutfitTransformer needs on this backbone, and
that OutfitTransformer's own published recipe pieces, built and validated on a different, weaker backbone
(ResNet-18, per the paper's own comparison point), do not automatically transfer as a further improvement on
top of a strong frozen SigLIP backbone with the hyperparameters already tuned for a simpler mechanism.

## Within-run evidence for what's happening, and its honest limits

Two pieces of directly measured evidence, both already surfaced in `stage2_training_log.md`:

1. **Phase 35's own curriculum curve peaks early and then plateaus/declines**: val Recall@10 climbs cleanly
   from 0.0853 (epoch 0, `hard_fraction`=0) to a peak of 0.1403 (epoch 34, `hard_fraction`=0.34), then does
   NOT continue improving as the curriculum pushes toward harder negatives -- it plateaus and drifts down
   through epoch 59 (0.12-0.13 range), where early stopping (patience=25 from the epoch-34 peak) ends the
   run. This happens with no divergence, no loss instability, no D_pos/D_neg collapse -- a genuine ceiling,
   not a training failure.

2. **Phase 31/32's own reference curve, at the IDENTICAL hyperparameters but no recipe at all (no warm
   start, no category token, pure random negatives from epoch 0), climbs far past where phase 35 plateaus**:
   by epoch 34 -- the exact epoch phase 35 peaks at 0.1403 -- phase 31/32's plain run had already reached
   0.1675 (`week7/phase31_fair_baseline_outfittransformer/data/budget_check_results.json`), and it kept
   climbing smoothly all the way to 0.1924 by epoch 83-95 before plateauing. **The plain recipe was already
   ahead of the full recipe's own eventual peak by epoch 34, using a colder start (no CP warm start at all)
   and no target-category signal.**

Taken together, these two curves are the honest evidence available from this phase's own (deliberately
un-ablated, single-run) design: something about the combination of pieces this phase adds -- CP warm start,
category-token conditioning, curriculum negatives, or some interaction among them -- trains slower and caps
lower than the simpler mechanism, at hyperparameters that were tuned FOR the simpler mechanism. This phase
cannot and does not attribute the gap to any one of the three pieces individually -- the brief asked for one
un-ablated "full recipe" run, not a per-piece ablation, and no such breakdown is claimed here.

## What this does and doesn't say about the recipe itself

**What it doesn't say**: that CP pre-training, category conditioning, or curriculum negatives are
inherently bad ideas, or that the published paper's own reported results (on ResNet-18, with its own
separately-tuned hyperparameters -- lr=1e-5, batch=50, margin=2, a completely different optimization
regime, per `implementation_notes.md`) are wrong. This phase deliberately held phase 32's hyperparameters
constant, per its own brief, specifically to isolate the recipe as the only variable -- and that same
discipline means any hyperparameter mismatch between what phase 31/32 tuned FOR (a single outfit token,
pure random negatives, no pre-training) and what phase 35's structurally different mechanism actually needs
is a real, stated confound this phase cannot rule out. A structurally different query mechanism (a
target-category token whose role and gradient signal differ from a fixed outfit token) and a structurally
different negative distribution (curriculum-shifting, not stationary) trained at hyperparameters chosen for
neither is not a guaranteed apples-to-apples test of the recipe's ceiling -- only of the recipe's effect
*at phase 32's specific operating point*, which is exactly, and only, what the brief asked this phase to
measure.

**What it does say, with confidence**: at phase 32's specific, already-strengthened operating point, adding
this project's best-effort implementation of OutfitTransformer's full recipe does not help, and costs
meaningfully at every K. Given that phase 32's simpler mechanism already reaches within ~2% of this
project's own best result (phase 28, cited in `week7/phase32_partial_ensemble_outfittransformer/phase32_notes.md`),
there wasn't much headroom left for a recipe change to close in the first place -- and this phase now shows,
directly, that the recipe alone (unretuned) doesn't close what little headroom remained.

## For the paper's central argument

This result adds a genuinely useful data point without undermining the project's own central finding
(investment asymmetry, not mechanism, explains most of the apparent architecture gap -- phases 31-34): once
investment parity is reached, a mechanism's OWN published training recipe is not automatically a further
source of improvement on a stronger, already-adapted backbone. The comparison that matters for a discussion
section is precise and should be stated exactly this way: **phase 32's simpler recipe (fixed
negative-sampling, checkpoint selection by Recall@10, proper val/test discipline) is what closed the
architecture gap; phase 35's more elaborate recipe, layered on top of that same operating point without
re-tuning for it, did not add further ground.** Whether recipe-specific re-tuning would change this is an
open question this phase deliberately did not answer (out of scope, no tuning was performed here per this
phase's own design) -- reported as an honest limitation, not something acted on further in this phase.

## Citation guidance

**Phase 35's single-model result (0.1311/0.2380/0.3075) is a new, separate data point -- it does NOT
supersede phase 32's baseline citation.** Phase 32's 3-seed ensemble (0.1897/0.3246/0.4019) remains the
honest "OutfitTransformer, investment parity" citation for this project, unchanged. Phase 28's 10-model
ensemble (0.1904/0.3267/0.4079) remains this project's own best overall result. Phase 35 should be cited
specifically as: "OutfitTransformer's full published recipe, tested at the same operating point as phase
32's investment-parity result, without recipe-specific hyperparameter re-tuning."
