# Phase 18, Steps 4-5: Loss Balancing Check (Four Corners)

Measured fresh on the freshly-initialized (untrained) `FourHeadDedicatedCapacity`, seed=42 -- not assumed from any prior phase's weights, per this project's standing rule.

`comp_rel` (complement-relevance, MNRL on the full also_buy set) is the fixed reference (weight=1.0) -- the double-protected objective, matching both phase 16/16d's 'relevance' convention and phase 12c/17's 'complement' convention.

## Initial loss magnitude (averaged over 20 calibration batches, no optimizer step taken)

| Corner | Initial loss | Weight |
|---|---|---|
| sub_rel | 0.2153 | 22.1712 |
| sub_tail | 0.2013 | 23.7129 |
| comp_rel | 4.7729 | 1.0000 |
| comp_tail | 4.8456 | 0.9850 |

## Gradient-norm verification (shared layer only, each corner isolated)

Gradient L2-norm into `model.shared.parameters()` -- the only shared submodule -- one fixed batch of 128 anchor edges per axis population:

| Corner | Grad norm, UNWEIGHTED | Grad norm, WEIGHTED | Ratio to reference (weighted) |
|---|---|---|---|
| sub_rel | 0.070860 | 1.575129 | 0.52x |
| sub_tail | 0.073079 | 1.850874 | 0.44x |
| comp_rel | 0.812566 | 0.812566 | 1.0x (reference) |
| comp_tail | 0.519673 | 0.505728 | 1.61x |

**Verification PASSED for all 3 non-reference corners**: after weighting, every corner's gradient into the shared layer is within the [0.1, 10] band relative to `comp_rel`. Proceeding to full training with these fixed weights.
