# Phase 18b: Contrastive Sub-Tail Signal -- Testing a Specific Diagnosed Cause

## The verdict, stated plainly

**The targeted fix worked, cleanly, on exactly the thing it was built to fix.** Replacing sub_tail's
ranking-distillation objective (which never presents a popular item as something to push away from)
with an explicit MNRL contrastive loss using head-tier-biased negatives resolved BOTH the narrow
pure-corner reversal AND the broader axis-2 cross-axis interaction phase 18 found. This confirms phase
18's specific diagnosis was correct, not just plausible: the axis-2 reversal was a training-method gap
in the substitute-tail corner, not (or at least not primarily) a more fundamental capacity-competition
or bilinear-blending artifact. One thing this phase did NOT touch and does NOT resolve, reported
separately and honestly: axis 1's hit-rate anomaly (substitute corners out-scoring comp_rel on real
co-purchase hit-rate) persists unchanged -- it was never this phase's target, and the evidence continues
to point at comp_rel's own retrieval quality / dataset-specific visual-similarity-correlates-with-
co-purchase effect, not at anything sub_tail related.

## Step 1: the new contrastive sub-tail signal

Positive side: each anchor's top-3 nearest tail-tier visual neighbors, reused directly from phase 18's
own `tail_nn_lookup.npz` (unchanged raw-SigLIP-derived structure) -- **74,157 clean pairs** (66,742
train / 7,415 val), 0 self-pairs, 100% tail-tier targets confirmed. Negative side (the actual fix): for
each anchor, half of the 8 sampled negatives are drawn specifically from HEAD-tier items (9,961
available in the pool, comfortably enough), the other half uniformly random -- mirroring comp_tail's
proven general-purpose negative sampling while deliberately targeting the exact failure mode diagnosed
(retrieving popular items). Coverage confirmed adequate on both sides before training. Full detail:
`signal_construction_summary.md`.

## Step 2: loss rebalancing -- the asymmetry was formulation-specific, not corner-specific

Measured fresh, not assumed. **The ~22-24x substitute/complement asymmetry does NOT persist for
sub_tail**: its weight dropped from phase 18's 23.71x to **1.01x** now that it's the same MNRL loss
family as comp_rel/comp_tail (initial magnitude 4.73, essentially identical scale to comp_rel's 4.77 and
comp_tail's 4.85). sub_rel, whose formulation is unchanged (still ranking-distillation), keeps
essentially the same weight as phase 18 (22.17x vs 22.17x) -- direct confirmation that the asymmetry was
always a property of the ranking-distillation loss's naturally-small-at-init KL-divergence scale, not
something inherent to the tail-exposure axis or either corner specifically. Gradient-norm verification
passed for all 3 non-reference corners (0.49x-1.61x, comfortably inside [0.1, 10]). Full detail:
`loss_balancing_check.md`.

## Step 3: training

Converged cleanly, 88 epochs, early-stopped, no collapse in any head throughout. **best_val_comp_rel =
1.1119** -- better than phase 18's 1.1746 (a real, if secondary, improvement: with sub_tail's weight
dropping from 23.71x to ~1x, total pressure on the shared trunk from the substitute side dropped
substantially, leaving more room for comp_rel). This is a useful side-confirmation of phase 18's own
capacity-competition framing: reducing an unnecessarily large loss weight measurably helped the
reference objective, independent of the sub_tail-specific fix itself.

## Step 4: the pure-corner check -- FIXED

| | sub_rel (alpha1=1.0, alpha2=1.0) | sub_tail (alpha1=1.0, alpha2=0.0) |
|---|---|---|
| Mean ref_count@10 -- phase 18 | 16.86 | 17.23 (WRONG direction) |
| Mean ref_count@10 -- **phase 18b** | 16.94 | **16.53 (correct)** |
| Tail fraction@10 -- phase 18 | 0.3415 | 0.3303 (WRONG direction) |
| Tail fraction@10 -- **phase 18b** | 0.3510 | **0.3596 (correct)** |

**FIXED.** sub_tail's pure corner now retrieves both a lower mean ref_count and a higher tail fraction
than sub_rel's pure corner -- completely reversed from phase 18's broken state, at exactly the
unblended, no-interpolation-involved level this phase's diagnosis pointed at.

## Step 5: full re-run of phase 18's evaluation

**Axis 2 (relevance/tail-exposure): FULLY RESOLVED, not just at the pure corners.** The cross-axis
reversal phase 18 found (ref_count/tail_fraction direction flipping at high alpha1) is gone entirely --
both metrics now move in the correct direction at EVERY value of alpha1:

| alpha1 | ref_count delta (phase 18) | ref_count delta (phase 18b) | tail_frac delta (phase 18) | tail_frac delta (phase 18b) |
|---|---|---|---|---|
| 0.0 | +3.46 | +2.13 | +0.0257 | +0.0214 |
| 0.25 | +3.04 | +1.85 | +0.0211 | +0.0134 |
| 0.5 | +1.41 | +1.21 | +0.0059 | +0.0019 |
| 0.75 | -0.46 (WRONG) | **+0.70 (fixed)** | -0.0052 (WRONG) | **+0.0041 (fixed)** |
| 1.0 | -0.38 (WRONG) | **+0.41 (fixed)** | -0.0112 (WRONG) | **+0.0087 (fixed)** |

Every delta is now the correct sign, at every alpha1 -- axis 2 is now genuinely independent of where
axis 1 is set. Full tables: `axis_independence_check.md`.

**Axis 1's hit-rate anomaly: UNCHANGED, and expected to be unchanged.** Hit-rate still increases (not
decreases) with alpha1 at every alpha2 (deltas -0.014 to -0.105, same pattern and same rough magnitude
as phase 18's -0.005 to -0.117). This was never this phase's target -- it's about comp_rel's own
retrieval quality relative to sub_rel/sub_tail on real co-purchase ground truth, unrelated to sub_tail's
training-method fix. Overlap-with-raw-SigLIP remains the reliable, fully axis-2-independent diagnostic
for axis 1, as in phase 18 (deltas +0.163 to +0.283, correct sign at every alpha2).

**Because axis 1's hit-rate check still fails, the overall automated "fully independent" verdict is
still False** -- but this is now cleanly attributable to a single, already-identified, orthogonal issue,
not to the axis-2 problem this phase set out to fix (which is gone).

**Smoothness: modestly improved across the board**, consistent with a better-calibrated, less
lopsided four-way loss:

| Gap | Phase 18 | Phase 18b |
|---|---|---|
| Axis 1 | 0.3476 | 0.3578 |
| Axis 2 | 0.2825 | 0.3082 |
| Diagonal | 0.2895 | 0.3112 |

Full detail: `smoothness_2d.md`.

## Step 6: what this means for phase 18's diagnosis

**Confirmed, not just plausible.** Phase 18's notes proposed two possible explanations for the axis-2
reversal: (1) a training-method gap specific to sub_tail's ranking-distillation objective lacking real
negatives, or (2) something more structural (genuine 4-way capacity competition, or bilinear-blending
dynamics not preserving rank-space interpolation even with orthogonal heads). This phase isolated
explanation (1) by changing nothing else -- same architecture, same three other corners' signals, same
overall procedure, only sub_tail's objective formulation and the resulting rebalanced weights changed --
and the axis-2 reversal disappeared completely, at both the narrow pure-corner level and the broader
interpolated-sweep level. **This is real evidence that (1) was the correct, sufficient explanation for
the axis-2 problem specifically** -- explanation (2)'s bilinear-blending concern may still be a real,
separate phenomenon (it's the most plausible remaining explanation for axis 1's UNRELATED hit-rate
anomaly, which this phase did not touch and did not fix), but it was not the cause of the axis-2 issue.

**Practical implication for this project's unified two-axis model**: with this fix, the tail-exposure
axis is now a genuinely reliable, axis-1-independent dial in the unified 4-head model -- a real
improvement over phase 18's state, achieved via a training-method correction (real negatives,
appropriately calibrated weight) rather than an architecture change, exactly as this phase's own
"do not do yet" scope required. The substitute/complement axis's separate, unresolved hit-rate anomaly
remains open for any future thread that wants to pursue it (the most likely direction: understanding why
visual similarity correlates so strongly with real co-purchase on this specific Amazon catalog, since
that's a property of the data more than of the model).

## Required-output checklist

- `signal_construction_summary.md` -- done, contrastive sub-tail pairs, coverage confirmed adequate on both positive and negative sides.
- `loss_balancing_check.md` -- done, rebalanced 4-way calibration, direct comparison to phase 18's weights, confirms the asymmetry was formulation-specific.
- `corner_sanity_checks.md` -- done, step 4's pure-corner comparison highlighted first (FIXED), full corner table, before/after comparison to phase 18.
- `axis_independence_check.md` -- done, axis 2 fully resolved, axis 1's unrelated hit-rate issue persists unchanged.
- `smoothness_2d.md` -- done, modest improvement across all three gaps.
- This file -- verdict: **the targeted fix worked cleanly on the specific issue diagnosed; a separate, orthogonal issue (axis 1 hit-rate) remains, as expected, since this phase never targeted it.**

## What comes next (not decided here, per this phase's own "do not do yet" instruction)

No architecture change (wider shared layer or otherwise) was attempted -- this phase deliberately tested
the training-method explanation in isolation first, per its own brief. No paper writeup was started. The
axis-1 hit-rate anomaly remains the one open thread in the unified two-axis model, for a future phase
that wants to pursue it specifically.
