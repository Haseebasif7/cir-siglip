# Phase 12d: Alpha Interpolation Sweep -- Overall Verdict

## Context

Phase 12c confirmed the two mode endpoints (alpha=0 complement, alpha=1 substitute) are
genuinely, correctly distinct. But everything tested through phase 12c only checked those
two fixed points plus one midpoint (alpha=0.5). This phase evaluates phase 12c's
already-trained checkpoint at 11 alpha values (0.0 to 1.0, step 0.1) -- no retraining --
to find out whether "controllable" describes a real dial across the full range, or just
two good settings with unknown behavior in between.

## Step 1: the sweep

Full results: `results_table.md`, `data/alpha_sweep_results.json`. All four diagnostic
metrics plus full-benchmark Recall@10/30/50 computed at every alpha, on fixed query
samples (same seeds as phases 12/12b/12c) so differences across alpha reflect alpha
itself, not sampling noise. The endpoint values (alpha=0.0 and alpha=1.0) match phase
12c's own reported numbers exactly, confirming this sweep is evaluating the same
checkpoint the same way.

## Step 2: monotonicity, checked two ways (`monotonicity_check.md`)

Beyond the standard step-wise check (does any single step reverse beyond a noise floor),
an extremum-location check was added after inspecting the raw numbers: for a metric
expected to move monotonically, its global min/max should sit at the alpha=0.0 endpoint.
A metric can drift the wrong way across several small steps in a row -- none large enough
individually to trip a per-step noise floor -- and still have a real, interior extremum,
which the step-wise check alone would miss. Both checks were needed here:

- **Overlap with complement mode (d)**: perfectly monotonic, both checks pass cleanly
  (1.000 -> 0.869 -> ... -> 0.402, strictly decreasing at every single step).
- **Visual similarity to query (a)**: no single step exceeds the noise floor, but the
  global minimum sits at alpha=0.1 (0.7177), not alpha=0.0 (0.7217) -- a small dip in the
  first step before rising monotonically from alpha=0.1 onward.
- **Co-occurrence hit rate (b)**: no single step exceeds the noise floor, but the global
  maximum sits at alpha=0.4 (0.108), not alpha=0.0 (0.102) -- the metric actually *rises*
  gently from alpha=0.0 to alpha=0.4 before falling as expected from 0.4 to 1.0.
- **Overlap with raw SigLIP (c)**: the clearest violation -- one single step (alpha
  0.0->0.1) drops by -0.0188, beyond the noise floor, and the global minimum sits at
  alpha=0.2 (0.143), not alpha=0.0 (0.171). The metric dips for the first two steps before
  rising steadily and substantially from alpha=0.2 to 1.0.

**Three of four metrics are not cleanly monotonic across the full range -- specifically,
all three show the same shape: a gentle, real (not noise-level, confirmed by the
extremum-location check) drift in the wrong direction across roughly alpha 0.0-0.2/0.4,
before behaving as expected for the rest of the range.** This is a genuine, consistent
"warm-up zone" pattern, not scattered random noise across different parts of the range for
different metrics -- all three affected metrics dip/hump in the SAME low-alpha region.

## Step 3: retrieval-level smoothness (`adjacent_vs_distant_overlap.md`)

This is a different, complementary question from step 2's monotonicity check: does moving
the dial a LITTLE change results a little, and moving it a LOT change results a lot? Full
11x11 pairwise overlap matrix, 500-query sample. **Answer: yes, clearly.** Mean overlap
decays smoothly and consistently as |delta alpha| grows: 0.877 (adjacent, delta=0.1) ->
0.769 -> 0.669 -> 0.582 -> 0.509 -> 0.453 -> 0.415 -> 0.397 -> 0.394 -> 0.402 (delta=1.0,
the two endpoints). This is a real, monotonic-within-noise decay across all 10 delta
values, not just a binary near/far split -- the actual operational definition of a usable
control. Notably, this property holds even though step 2 found the individual diagnostic
metrics aren't perfectly monotonic in the low-alpha region -- the retrieval SETS still
change gradually and predictably step to step throughout the whole range, including inside
the "warm-up zone."

## Step 4: qualitative confirmation (`qualitative_examples/`)

4 queries (bags, jewellery, tops, shoes), top-5 at alpha = 0.0, 0.33, 0.67, 1.0. The shoes
example shows the shift particularly clearly: alpha=0.0's results are a stylistically
eclectic mix (a pink flat with graphic text, red pointed flats, black suede pumps in
different styles) that gradually consolidates -- by alpha=0.67-1.0 the results have
converged on a visually cohesive set of black strap/bow heels. The transition is visibly
*gradual* across the intermediate steps (0.33, 0.67), not an abrupt jump between two
different-looking extremes -- consistent with step 3's quantitative finding.

## Step 5: visualization

`plots/alpha_sweep.png` -- one plot, four lines (Okabe-Ito CVD-safe categorical palette,
validated with the dataviz skill's own `validate_palette.js`, all checks passed). The
shapes described in step 2 are immediately visible: overlap-with-complement descends in a
single clean curve; the other three lines show a small dip/hump in the first 1-3 points
before settling into the expected trend for the rest of the range.

## Overall verdict: a real dial, with a caveat about its low end

**This is a genuine, usable control dial across most of its range, not just two points
that happen to work -- but it is not a perfectly clean dial across the ENTIRE range, and
that should be stated plainly rather than rounded up to "fully monotonic."**

The evidence splits cleanly into two findings that both matter:

1. **The retrieval-level smoothness property (step 3) holds essentially perfectly across
   the whole range.** Moving alpha a little always changes retrieval a little; moving it a
   lot always changes retrieval a lot. This is arguably the more fundamental property for
   "is this a usable dial" -- a user turning the knob a small amount will always see a
   correspondingly small, predictable change in results, everywhere in the range.
2. **But the four *diagnostic* metrics are not all monotonic everywhere** -- three of four
   show a consistent, real (not noise) drift in the wrong direction across roughly
   alpha=0.0-0.2/0.4 before behaving as expected. Practically, this means: alpha values in
   the very low end (roughly 0.0 to 0.2-0.3) do not reliably deliver "slightly more
   visual-similarity-flavored than pure complement" -- they can, on average, actually be
   marginally LESS visually similar / LESS raw-SigLIP-like than pure alpha=0.0 itself, a
   small but real inversion right at the low end of the dial.

**Practical recommendation**: the dial is trustworthy and safe to expose across roughly
alpha in [0.3, 1.0] -- fully monotonic, smooth, and qualitatively coherent in that range.
The [0.0, 0.3) region should either be presented only via its two "anchor" settings
(alpha=0.0 pure complement, perhaps alpha=0.3 as the first fully-monotonic point) rather
than as a freely-adjustable sub-range, or investigated further before being exposed as a
fine-grained control -- the underlying cause of this low-end warm-up zone is not yet
diagnosed (candidate explanations not tested here: possibly an artifact of how the
additive mode-vector geometry behaves for small substitute contributions, or a genuine
property of the trained model near alpha=0 that a different training setup might avoid).

This is evaluation-only, consistent with the brief -- no retraining was done, and per the
brief's own scope, the CSA-Net baseline reproduction and the complement-mode capacity
tradeoff identified in phase 12c remain separate, not-yet-started next steps.

## Required-output checklist

- `results_table.md` -- done, all metrics + Recall@K at all 11 alpha values.
- `monotonicity_check.md` -- done, step-wise AND extremum-location checks for all 4 metrics.
- `adjacent_vs_distant_overlap.md` -- done, full 11x11 matrix + delta-alpha decay curve.
- `plots/alpha_sweep.png` -- done, validated CVD-safe palette.
- `qualitative_examples/` -- done, 4 queries x 4 alpha values.
- This file -- verdict: **real dial across most of the range (roughly alpha >= 0.3), with
  an honest, evidenced caveat about a low-end (0.0-0.3) warm-up zone that is not yet
  understood.**
