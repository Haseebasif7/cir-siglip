# Phase 18b, Steps 2-3: Loss Balancing Check (Four Corners, Contrastive Sub-Tail)

Measured fresh on the freshly-initialized (untrained) `FourHeadDedicatedCapacity`, seed=42 -- NOT assumed from phase 18's weights, since sub_tail's objective formulation changed entirely (KL ranking-distillation -> MNRL contrastive).

`comp_rel` remains the fixed reference (weight=1.0), unchanged reasoning from phase 18.

## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)

| Corner | Initial loss | Weight (phase 18b) | Weight (phase 18, for comparison) |
|---|---|---|---|
| sub_rel | 0.2153 | 22.1709 | 22.1712 |
| sub_tail | 4.7348 | 1.0080 | 23.7129 |
| comp_rel | 4.7728 | 1.0000 | 1.0000 |
| comp_tail | 4.8454 | 0.9850 | 0.9850 |

## Does the ~22-24x substitute/complement asymmetry persist under the new formulation?

Phase 18: substitute-side weights averaged 22.94x the complement reference (sub_rel=22.17x, sub_tail=23.71x -- the latter was ranking-distillation, a structurally much-smaller-scale loss at init). Phase 18b: substitute-side weights now average 11.59x (sub_rel=22.17x, unchanged formulation so similar scale expected; sub_tail=1.01x, now MNRL, the same loss family as comp_rel/comp_tail, so a much smaller weight is expected since its initial scale is now comparable to theirs).
**The asymmetry does NOT persist for sub_tail specifically** -- its weight dropped from 23.71x to 1.01x, confirming the asymmetry was a property of the ranking-distillation LOSS FORMULATION (a KL-divergence term naturally starts near zero for an untrained model), not something inherent to the sub_tail corner or the tail-exposure axis itself. sub_rel (unchanged formulation) still carries a comparable weight to before (22.17x vs 22.17x), confirming the asymmetry really was formulation-specific, not corner-specific.

## Gradient-norm verification (shared layer only, each corner isolated)

Gradient L2-norm into `model.shared.parameters()` -- the only shared submodule -- one fixed batch of 128 anchor edges per stream:

| Corner | Grad norm, UNWEIGHTED | Grad norm, WEIGHTED | Ratio to reference (weighted) |
|---|---|---|---|
| sub_rel | 0.074100 | 1.647725 | 0.49x |
| sub_tail | 0.859650 | 0.870710 | 0.93x |
| comp_rel | 0.812566 | 0.812566 | 1.0x (reference) |
| comp_tail | 0.519985 | 0.503432 | 1.61x |

**Verification PASSED for all 3 non-reference corners**: after weighting, every corner's gradient into the shared layer is within the [0.1, 10] band relative to `comp_rel`. Proceeding to full training with these fixed weights.
