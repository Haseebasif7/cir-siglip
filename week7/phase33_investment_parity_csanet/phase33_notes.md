# Phase 33: Honest Interpretation -- The Cross-Architecture Comparison

## The central finding, stated plainly: the pattern does NOT replicate

Phase 32 found that OutfitTransformer, given fair investment (checkpoint-selection fix, text, tuning,
partial ensembling), closed its gap to this project's own model to under 2% at every K. **CSA-Net, given
the same class of investment, does not.** Its investment-parity result reaches only **65.5% / 66.2% / 67.4%**
of phase 28's own 10-model ensemble, and trails phase 32's investment-parity OutfitTransformer result by
**-34.3% / -33.3% / -31.6%** relative at every K. This is a genuine divergence between two independently
published architectures given the same investment treatment, not a second confirmation of phase 31/32's
pattern -- reported precisely, not smoothed over, per the brief's own explicit instruction.

## Full progression

| Configuration | R@10 | R@30 | R@50 |
|---|---|---|---|
| Phase 13b original CSA-Net reproduction (un-invested) | 0.0725 | 0.1393 | 0.1844 |
| **Phase 33, 3-seed investment-parity CSA-Net ensemble** | **0.1247** | **0.2164** | **0.2748** |
| Phase 32, investment-parity OutfitTransformer (3-seed) | 0.1897 | 0.3246 | 0.4019 |
| Phase 28 text ensemble (project's own best, 10-model) | 0.1904 | 0.3267 | 0.4079 |

CSA-Net DID improve substantially with investment: +72.0%/+55.4%/+49.0% relative over its own un-invested
baseline -- a real, large gain, comparable in kind (if not in final destination) to OutfitTransformer's own
+206%/+142%/+112% over ITS un-invested baseline. Investment clearly matters for both architectures. What
differs is where each ends up relative to this project's own best result.

## Per-step contribution, and where this phase's investment differs from phase 31/32's

| Step | Val R@10 | Relative gain |
|---|---|---|
| A0 (phase 13b as-shipped, evaluated on val) | 0.0786 | -- |
| A2 (selection fix, patience UNCHANGED at 5) | 0.0779 | -0.9% (within noise) |
| A3 (+ text) | 0.0986 | +26.6% |
| Step 3 (+ LR/batch-size tuning) | 0.1075 | +9.0% |
| Step 4 (+ 3-seed ensemble) | 0.1318 | +22.7% |

**Step 1 found a genuine, notable architecture-specific difference from OutfitTransformer, reported
honestly in `checkpoint_selection_check.md`**: the naive checkpoint-selection fix (swap val_loss for
Recall@10, same patience) worked fine here, with NO need for the patience recalibration OutfitTransformer
required. CSA-Net's Recall@10 climbs almost monotonically through training (no ~17-epoch noisy plateau),
so val_loss's imperfect proxy happened to land near-optimally by coincidence of the trajectory's shape, not
because it's a principled criterion. This is exactly the kind of "if it doesn't show the same pattern,
describe precisely how and where it diverges" outcome the brief invited for step 1 specifically -- and it
turned out to extend to the phase's central question too.

## What this phase's investment did NOT include, and why that matters for interpreting the gap

This is the most important honest caveat in this document. **CSA-Net's negative-sampling scheme was never
touched in this phase** -- it still trains on the same mined (SigLIP-nearest-neighbor, same-category) hard
negatives phase 13/13b originally used (`week4/phase13_csa_net_baseline/data/negative_candidates.json`),
per the brief's own step list (checkpoint selection, text, tuning, ensembling -- no negative-sampling
change specified, and none attempted here, consistent with staying in scope).

But this project's OWN prior work makes a specific, well-evidenced case that this matters a great deal:
OutfitTransformer's negative-sampling scheme WAS already fixed, in phase 14b, well before phase 31 even
started -- mined negatives (run 1) scored 0.0051, catastrophically worse than random same-category
negatives (run 2, the adopted fix) at 0.0588, an over 10x difference (`week4/phase14b_.../training_log.md`,
and this project's own established phase 7-9 finding that hard-negative mining underperforms random
negatives for these embedding losses, independently reconfirmed a second time there). **OutfitTransformer's
"investment history" going into phase 31 therefore already included a negative-sampling fix that
CSA-Net's investment history, going into this phase, still lacks.** This is a genuine, structural asymmetry
between the two architectures' cumulative investment -- not something phase 31/32 vs. phase 33 treated
identically, because it was never phase 33's brief to test it, and this document flags that gap explicitly
rather than let it pass as coincidental.

This is offered as the single most plausible, testable, NOT YET TESTED explanation for a meaningful share
of CSA-Net's remaining gap -- stated as a hypothesis with real supporting precedent from this project's own
history, not as a proven cause. It was not tested here, per this phase's own scope.

## Two other genuine, un-conflated factors worth naming

1. **Ensemble-size mismatch, and CSA-Net looks further from its ceiling.** Both architectures got only 3 of
   a possible 10 seeds. But CSA-Net's 3-seed ensembling gain over its own best solo seed was **+22.7%**
   relative -- almost 5x larger than OutfitTransformer's own +4.8% (phase 32). A larger per-seed gain at
   the SAME ensemble size is a real signal that CSA-Net's individual seeds disagree with each other more
   (individual-seed spread: CSA-Net std 0.0027 vs. OutfitTransformer std 0.00074, `individual_seeds.md`),
   which this project's own established pattern (phase 26/28's own diminishing-but-real returns past size
   3-5) suggests means CSA-Net is likely further from its own 10-seed ceiling than OutfitTransformer was at
   the same 3-seed checkpoint. Untested here: whether a full 10-seed CSA-Net ensemble would close more of
   the remaining gap than OutfitTransformer's own (still-open) 10-seed question would.
2. **Raw model capacity.** CSA-Net-on-SigLIP has 51,333-100,485 parameters (image-only/image+text) --
   roughly 5-10x fewer than OutfitTransformer's 899,840-998,144. Architectural scale testing is explicitly
   out of scope for this phase (per the brief's own "do not," mirroring phase 31's own scope limit) --
   whether CSA-Net's k=5 subspaces / embed_dim=64 design is simply under-provisioned for this task,
   independent of training investment, is a real, completely untested question.

None of these three factors (negative sampling, ensemble size, capacity) is isolated from the others here
-- this phase cannot and does not claim which one explains how much of the remaining gap. What it can say
honestly is that all three are real, plausible, unconflated candidates, and none of them was part of this
phase's own investment treatment.

## Does the investment-asymmetry claim generalize across architectures?

**Partially, and with an important qualification, not a clean yes.** The core mechanistic finding from
phase 31 -- that under-investment (broken checkpoint selection, no text, no tuning) can produce an
artificially large apparent gap to this project's own model -- DOES generalize: CSA-Net closed a real,
substantial fraction of its own gap (from ~38% of phase 28's ensemble at its un-invested number, to ~66%
with the same class of investment). But the STRONGER claim from phase 32 -- that fair investment closes the
gap to near-parity -- does NOT generalize to CSA-Net at the investment level actually applied here. The
honest, precise statement for this project's central methodological claim: **investment asymmetry explains
a meaningful share of the apparent architecture gap in this research area, but it is not the WHOLE
explanation for every architecture** -- OutfitTransformer's specific gap turned out to be almost entirely
investment-driven; CSA-Net's gap is partially investment-driven, with real, identified, but untested
remaining candidates (negative-sampling scheme, ensemble scale, raw capacity) that this phase's own scope
did not reach.

## The honest, final comparison numbers for this project going forward

**Phase 33's 3-seed CSA-Net ensemble -- test-benchmark Recall@10/30/50 = 0.1247/0.2164/0.2748, checkpoints
`week7/phase33_investment_parity_csanet/models/ot33_seed{42,1,2}.pt` (score-averaged over distances, never
embeddings) -- is now the honest CSA-Net baseline citation for this project, superseding phase 13b's
original number (0.0725/0.1393/0.1844), which remains valid and citable as the un-invested starting point.**

**Phase 28's mean-pooled text-only ensemble (0.1904/0.3267/0.4079) remains this project's best overall
result, ahead of BOTH investment-parity baselines at every K** -- narrowly ahead of OutfitTransformer's
(phase 32, under 2% relative) and substantially ahead of CSA-Net's (this phase, 33-35% relative). Any
report language should state the two comparisons separately, not average them into one "how does our model
compare to prior architectures" number -- they tell genuinely different stories, and collapsing them would
misrepresent both.

## For the paper's central argument

This result is more valuable to the paper than a second clean confirmation would have been: it turns
"investment asymmetry explains the apparent gap" from an unqualified claim (risky, resting on one data
point) into a nuanced, evidence-backed one with a genuine boundary condition identified and named, not
hidden. The identified boundary (negative-sampling investment history, ensemble-scale distance from
ceiling, raw capacity) gives the paper's discussion section real, specific, testable follow-up questions
instead of a vague "more work is needed."

## Untested, flagged explicitly for a future phase

- CSA-Net with random (not mined) negatives, holding everything else in this phase's own configuration
  fixed -- the single most promising, well-motivated, not-yet-attempted next step, directly informed by
  this project's own established phase 7-9/14b precedent.
- A full 10-seed CSA-Net ensemble (`scripts/04_train_seed.py` is built and proven safe, ready for seeds
  3-9), given the larger per-seed ensembling gain observed here suggests more headroom than OutfitTransformer
  had at the same scale.
- Architectural scale testing for CSA-Net (wider embed_dim, more subspaces k), completely untested,
  explicitly out of scope for this phase per the brief's own instruction.
