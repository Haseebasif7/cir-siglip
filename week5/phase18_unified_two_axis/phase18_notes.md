# Phase 18: Unified Two-Axis Controllable Model

## The verdict, stated plainly

**Mixed, not a clean generalization of the dedicated-capacity principle to two simultaneous axes.**
The architecture itself works exactly as designed -- verified endpoint identity, exact gradient
isolation, near-orthogonal heads, no collapse, and real, substantial smoothness along both axes and
the diagonal. But the retrieval-level behavior does NOT show the two axes are genuinely independent:
axis 2 (relevance/tail-exposure) only moves in the correct direction when the substitute/complement
axis is leaning complement, and reverses when it leans substitute. A second, separate honest finding:
hit-rate turned out to be an unreliable diagnostic for the substitute/complement axis on this data --
the substitute corners score HIGHER hit-rate than the complement corner that was actually trained on
the co-occurrence signal, traced to a real capacity-competition cost (comp_rel's own validation loss
degraded further than either 2-head predecessor). Both findings are reported directly, not smoothed
over -- this phase's central question (do the two axes behave independently once unified) gets a real,
substantive "not fully" answer, which is itself a useful, citable result.

## Steps 1-2: the new training signals, built and coverage-checked

**Step 1 (the one genuinely new signal): tail-restricted nearest-neighbor lookup.** Of phase 7's
24,719-item Amazon pool, 11,190 items (45.3%) are tail-tier by phase 3's definition -- confirmed
directly, matching phase 16c's own item-level composition finding for this same pool. Built raw-SigLIP
top-50 neighbor lookups restricted to tail-tier columns only for every item in the pool. **Coverage:
0/24,719 anchors needed padding** -- every anchor has a full set of 50 distinct tail-tier visual
neighbors. No supplementary sampling was needed. `signal_construction_summary.md`.

**Step 2: substitute-relevance signal, rebuilt on Amazon.** Phase 12c's unrestricted ranking-
distillation method, previously only run on Polyvore's 251,008-item pool, applied here to the same
24,719-item Amazon pool for the first time. Both lookups built in seconds locally (no Modal needed at
this pool size).

## Step 3: architecture -- four dedicated heads, bilinear interpolation

Direct extension of phase 16d/17's dedicated-capacity pattern from 2 heads to 4: one minimal shared
`Linear(768,256)` layer feeding four fully independent `Linear(256,128)` heads (`sub_rel_head`,
`sub_tail_head`, `comp_rel_head`, `comp_tail_head`), blended via bilinear interpolation across
`alpha1` (substitute/complement) and `alpha2` (relevance/tail-exposure).

**Verified directly, not assumed**: at each of the 4 grid corners, the bilinear blend collapses to
exactly that corner's own head output (max abs diff <= 2.98e-08) and backpropagation produces exactly
zero gradient into the other three heads -- the 2D generalization of phase 16d/17's own verified
endpoint-identity and gradient-isolation properties. `architecture_notes.md`.

## Step 4-5: training and fresh loss-balancing calibration

`comp_rel` (MNRL on the full also_buy set) chosen as the fixed reference (weight=1.0) -- the natural
double-reference point, since it's both phase 16/16d's "relevance" signal and phase 12/17's
"complement"-family signal. Calibration measured fresh: `initial_sub_rel=0.2153`,
`initial_sub_tail=0.2013`, `initial_comp_rel=4.7729`, `initial_comp_tail=4.8456` ->
`weight_sub_rel=22.17`, `weight_sub_tail=23.71`, `weight_comp_tail=0.99`. **Gradient-norm verification
PASSED for all 3 non-reference corners** (ratios 0.44x-1.61x, comfortably inside the [0.1, 10] band).
`loss_balancing_check.md`.

Training (each corner trained only at its own fixed extreme, per phase 15's finding that continuous
interpolation dilutes per-corner signal, mirroring phase 16d/17's joint-step convention extended to 4
terms): converged cleanly, **57 epochs, early-stopped, no collapse in any of the 4 heads at any point**
(mean pairwise cosine stayed in a healthy 0.02-0.69 range across heads throughout,
`models/training_curves.json`). Best checkpoint at epoch 51, **best_val_comp_rel = 1.1746**.

**This is measurably worse than either 2-head predecessor's equivalent objective**: phase 16's original
unweighted relevance loss reached 0.9735, phase 16d's 2-head dedicated-capacity version reached 1.0750,
and this 4-head version reaches 1.1746 -- a real, honestly-reported cost of adding two more competing
objectives (`sub_rel`, `sub_tail`) onto the same minimal shared trunk. Unlike phase 17 (which found
dedicated capacity fully absorbed a second competing objective with NO quality cost to the protected
one), here a third and fourth competing objective were added to the same-sized shared layer, and some
capacity pressure returns -- the shared trunk's dimensionality reduction step is a genuinely finite
resource, and 4-way sharing costs more than 2-way sharing did. This is directly relevant to
interpreting the retrieval-level findings below, not a side note.

## Step 6: corner sanity checks -- 3 of 4 clean, 1 surprising

| Corner | (alpha1, alpha2) | Hit Rate@10 | Mean ref_count@10 | Tail fraction@10 | Overlap w/ raw SigLIP@10 |
|---|---|---|---|---|---|
| sub_rel | (1.0, 1.0) | 0.4594 | 16.86 | 0.3415 | 0.6674 |
| sub_tail | (1.0, 0.0) | 0.4546 | 17.23 | 0.3303 | 0.6638 |
| comp_rel | (0.0, 1.0) | 0.4311 | 18.84 | 0.3417 | 0.4343 |
| comp_tail | (0.0, 0.0) | 0.3381 | 15.39 | 0.3675 | 0.3047 |

- **sub_rel**: clean pass. Overlap-with-raw-SigLIP (0.6674) is clearly the highest of all 4 corners,
  exactly as a substitute-mode corner should look.
- **comp_tail**: clean pass. Lowest ref_count (15.39) and lowest overlap-with-raw of all 4 corners,
  matching phase 16d's tail-mode signature directionally (phase 16d tail: ref_count@5=15.86,
  Hit Rate@10=0.3029 -- comp_tail here: 15.39 and 0.3381, same direction, different exact magnitudes
  as expected from a different, more heavily shared architecture).
- **sub_tail (the new corner)**: clean pass on its own explicit bar. Both required properties hold
  simultaneously: more visually similar to the query than comp_rel (0.6638 vs 0.4343 overlap) AND
  lower ref_count than comp_rel (17.23 vs 18.84) -- checked directly, not assumed from sub_rel and
  comp_tail's individual behaviors.
- **comp_rel -- the one surprising result**: Hit Rate@10 (0.4311) is NOT the highest among the 4
  corners -- both sub_rel (0.4594) and sub_tail (0.4546) score higher hit-rate against the real
  also_buy/also_viewed ground truth than comp_rel, the corner literally trained on that co-occurrence
  signal. Flagged directly rather than glossed over; explained below alongside step 7.

Full detail: `corner_sanity_checks.md`.

## Step 7: axis independence -- the central question, answered "not fully"

**Axis 1 (substitute/complement) via overlap-with-raw-SigLIP: genuinely independent of alpha2.**
Moving alpha1 from 0 to 1 increases overlap-with-raw at every single value of alpha2 (deltas
+0.176 to +0.359, all positive) -- this half of axis 1 is a real, robust, axis-2-independent dial.

**Axis 1 via hit-rate: consistently WRONG direction, but consistently so.** At every value of alpha2,
hit-rate goes UP (not down) as alpha1 increases toward substitute (deltas -0.005 to -0.117, all
negative relative to the expected sign). This ties directly to the comp_rel finding above: comp_rel's
own real quality was diluted by the 4-way capacity competition (val_comp_rel 1.1746, the worst of any
comparable corner in this project's history), while sub_rel's ranking-distillation objective converged
very cleanly (val_sub_rel=0.0179) and evidently produces retrievals that, on this specific
Clothing/Shoes/Jewelry catalog, correlate reasonably well with real co-purchase anyway (visually similar
items are often bought together in this category -- a plausible, catalog-specific explanation, not
tested further here). The takeaway: hit-rate is not a reliable diagnostic for this project's
substitute/complement axis on Amazon specifically; overlap-with-raw-SigLIP is the more trustworthy one
for this dataset, and the two disagreeing so sharply is itself informative.

**Axis 2 (relevance/tail-exposure): direction flips depending on alpha1 -- a real cross-axis
interaction, not independence.** Ref_count and tail_fraction both move correctly (relevance retrieves
higher ref_count / lower tail_fraction than tail-exposure) at alpha1 = 0, 0.25, 0.5 -- but REVERSE at
alpha1 = 0.75, 1.0 (ref_count delta flips from +3.46 at alpha1=0 to -0.38 at alpha1=1.0). In other
words: the tail-exposure dial only works as intended when the model is leaning complement; once leaning
substantially toward substitute, "tail-exposure" mode actually retrieves slightly MORE popular items
than "relevance" mode does at that same alpha1. This is the phase's clearest evidence that the two axes
are not fully independent once unified into one model.

Full tables: `axis_independence_check.md`.

## Step 8: smoothness extended to 2D -- real, but weaker than the pure 1D mechanisms

| Mechanism | Gap |
|---|---|
| Phase 12d (substitute/complement, shared trunk) | 0.4751 |
| Phase 16d (relevance/tail, dedicated capacity) | 0.6110 |
| Phase 17 (substitute/complement, dedicated capacity, Polyvore) | 0.6646 |
| **Phase 18, axis 1** | **0.3476** |
| **Phase 18, axis 2** | **0.2825** |
| **Phase 18, diagonal** | **0.2895** |

All three phase-18 gaps are real and substantial (computed from a coarser 5-point grid, not directly
apples-to-apples with the 11-point 1D sweeps, but comparable in spirit) -- moving smoothly through
either axis, or diagonally through both at once, produces smoothly decaying retrieval overlap, not a
sharp discontinuity. But all three gaps are meaningfully smaller than any single-axis mechanism this
project has built. This is consistent with the training-quality finding above: four objectives sharing
one minimal trunk leaves a somewhat less decisively separated space than two objectives did. Full
detail: `smoothness_2d.md`.

## Bonus: head-similarity diagnostic

All 6 pairwise combinations of the 4 heads' own outputs are near-orthogonal (|cosine| <= 0.0235,
`logs/head_similarity_diagnostic.md`) -- representationally, the dedicated-capacity fix worked exactly
as it did in phases 16d/17, even under 4-way sharing. This is an important, separate finding from step
7's retrieval-level result: **the heads themselves are cleanly independent in representation space, but
the axis-2 cross-axis interaction still shows up at the retrieval level.** The likely explanation is
that bilinear blending of (even orthogonal) normalized vectors, followed by renormalization, is not a
purely additive operation on the resulting similarity RANKINGS -- interpolating in embedding space does
not guarantee interpolating in retrieval-rank space. This is flagged as the most plausible mechanism,
not fully diagnosed further here (out of this phase's scope).

## What this means for the project's design-principle claim

**The dedicated-capacity architectural fix itself continues to hold up mechanistically** -- a third
independent confirmation (after phase 16d's relevance/tail axis and phase 17's substitute/complement
axis) that giving each mode/corner its own head, off a minimal shared trunk, produces cleanly
independent, non-collapsing, near-orthogonal representations, now even at 4-way scale. That specific,
narrower claim generalizes again.

**But the STRONGER claim this phase set out to test -- that unifying two already-independently-proven
axes into one model preserves full behavioral independence between them -- is NOT supported.** Two
concrete costs were found and reported honestly: (1) capacity competition returns, measurably, once a
third and fourth objective share the same minimal trunk (comp_rel's validation loss is the worst
equivalent-objective number in this project's history); (2) the tail-exposure axis's correct direction
depends on where the substitute/complement axis is set -- a real, substantive interaction, not just
noise (directionally consistent reversal at both alpha1=0.75 and alpha1=1.0). A genuinely unified,
fully-independent 2-axis controllable mechanism is not yet achieved by this specific design; the most
plausible next lever (not attempted here, out of this phase's scope) would be a wider shared
dimensionality-reduction layer, or moving that layer's width itself into the calibration search space,
to give 4-way sharing more room before competition sets back in.

## Required-output checklist

- `signal_construction_summary.md` -- done, both new/rebuilt signals, coverage checks passed cleanly.
- `architecture_notes.md` -- done, endpoint-identity and gradient-isolation checks passed for all 4 corners.
- `loss_balancing_check.md` -- done, fresh 4-way calibration, all 3 non-reference corners passed verification.
- `corner_sanity_checks.md` -- done, 3 of 4 corners clean, 1 surprising result (comp_rel's hit-rate) reported directly.
- `axis_independence_check.md` -- done, this phase's central question: answered "not fully independent," with the exact tables showing where and how.
- `smoothness_2d.md` -- done, all three gaps (axis 1, axis 2, diagonal) real and substantial but smaller than the pure 1D mechanisms.
- This file -- verdict: **mixed**. Architecture mechanism generalizes a third time; full 2-axis behavioral independence does not (yet) hold.
- `logs/head_similarity_diagnostic.md` -- bonus mechanistic check, included for traceability.

## What comes next (not decided here, per this phase's own "do not do yet" instruction)

No paper writeup was started. The most concrete open thread this phase leaves behind: whether a wider
shared layer (trading off some of the dedicated-capacity philosophy) recovers full axis independence at
4-way scale, or whether the interaction is more fundamental to bilinear blending itself -- neither
tested here, both out of scope for this phase.
