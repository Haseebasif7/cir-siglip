# Phase 20, Step 3: The Professor's Point 7 Bar, Checked Explicitly

Point 7 asks: does the best approach consistently outperform OutfitTransformer, at every K, not just one favorable one. Phase 9's plain ProjectionHead is the strongest configuration in `final_comparison_table.md`, so it is the one checked here, at K=10, K=30, and K=50 separately, both ways the brief asks for.

## Check 1: against phase 14b's OutfitTransformer reproduction (fully matched, same frozen SigLIP backbone -- no caveat needed, this is a direct, fair comparison)

| K | Phase 9 | Phase 14b OutfitTransformer repro | Phase 9 - repro | Phase 9 / repro | Outperforms? |
|---|---|---|---|---|---|
| 10 | 0.1317 | 0.0588 | +0.0729 | 2.24x | **YES** |
| 30 | 0.2464 | 0.1286 | +0.1178 | 1.92x | **YES** |
| 50 | 0.3216 | 0.1809 | +0.1407 | 1.78x | **YES** |

**Clears the bar at all three K values, by a wide and increasing-toward-K10 margin (1.78x to 2.24x). This is the strongest, cleanest form of the claim: same backbone, same benchmark, same protocol, no caveat needed anywhere.**

## Check 2: against OutfitTransformer's own published numbers (different backbone -- CLIP vs. SigLIP -- and candidate-pool construction, properly caveated, directional context only per this project's standing convention)

| K | Phase 9 | OutfitTransformer published | Phase 9 - published | Phase 9 / published | Outperforms? |
|---|---|---|---|---|---|
| 10 | 0.1317 | 0.0958 | +0.0359 | 1.37x | **YES** |
| 30 | 0.2464 | 0.1796 | +0.0668 | 1.37x | **YES** |
| 50 | 0.3216 | 0.2198 | +0.1018 | 1.46x | **YES** |

**Clears the bar at all three K values here too, consistently around 1.4-1.5x, not just at one favorable K.** Properly caveated: this is not a matched-protocol comparison (different backbone, different candidate-pool construction), so it is reported as directional context, the same standing convention this project has used for every literature-anchor comparison since phase 12.

## Plain verdict

**Phase 9's plain projection mechanism consistently outperforms OutfitTransformer at every single K, both against the fully matched, same-backbone reproduction (phase 14b) and against OutfitTransformer's own published numbers (properly caveated).** No K where this reverses, no ambiguous or marginal K rounded up -- every comparison in both checks lands clearly above 1.0x, most well above it. This is not a mixed or partial result.
