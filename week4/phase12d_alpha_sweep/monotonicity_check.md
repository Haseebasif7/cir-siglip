# Phase 12d, Step 2: Monotonicity Check

For each metric, checked step-by-step across all 11 alpha values (0.0 to 1.0, step 0.1) whether it moves in the expected direction consistently, not just between the two endpoints. A step-to-step change smaller than 0.01 is treated as noise/flat, not a violation, given the underlying 500-1,000-query sample sizes (see script docstring for the noise-floor justification).

## Visual similarity to query (a)

Values across alpha 0.0 -> 1.0: [0.7217, 0.7177, 0.7182, 0.7233, 0.7304, 0.7373, 0.7425, 0.7458, 0.7479, 0.7491, 0.7498]
Expected direction as alpha rises: **increase**. Net change (alpha=1.0 minus alpha=0.0): +0.0281 (correct endpoint direction).
No single step moves against the expected direction beyond the 0.01 noise floor.
**Global minimum sits at alpha=0.1 (0.7177), NOT at the expected alpha=0.0 endpoint** -- even though no single step exceeded the noise floor, the metric drifts the wrong way across several consecutive small steps before turning around, a real (if gentle) hump/dip a pure step-wise check would miss.

## Co-occurrence hit rate (b)

Values across alpha 0.0 -> 1.0: [0.102, 0.103, 0.104, 0.105, 0.108, 0.108, 0.102, 0.096, 0.085, 0.079, 0.076]
Expected direction as alpha rises: **decrease**. Net change (alpha=1.0 minus alpha=0.0): -0.0260 (correct endpoint direction).
No single step moves against the expected direction beyond the 0.01 noise floor.
**Global maximum sits at alpha=0.4 (0.1080), NOT at the expected alpha=0.0 endpoint** -- even though no single step exceeded the noise floor, the metric drifts the wrong way across several consecutive small steps before turning around, a real (if gentle) hump/dip a pure step-wise check would miss.

## Overlap with raw SigLIP (c)

Values across alpha 0.0 -> 1.0: [0.1708, 0.152, 0.143, 0.1536, 0.18, 0.2138, 0.2584, 0.2908, 0.3126, 0.332, 0.3428]
Expected direction as alpha rises: **increase**. Net change (alpha=1.0 minus alpha=0.0): +0.1720 (correct endpoint direction).
**1 single step(s) move against the expected direction beyond the 0.01 noise floor**: alpha 0.0->0.1 (delta -0.0188)
**Global minimum sits at alpha=0.2 (0.1430), NOT at the expected alpha=0.0 endpoint** -- even though no single step exceeded the noise floor, the metric drifts the wrong way across several consecutive small steps before turning around, a real (if gentle) hump/dip a pure step-wise check would miss.

## Overlap with complement mode (d)

Values across alpha 0.0 -> 1.0: [1.0, 0.8692, 0.7738, 0.6954, 0.637, 0.5746, 0.5336, 0.4894, 0.457, 0.426, 0.4022]
Expected direction as alpha rises: **decrease**. Net change (alpha=1.0 minus alpha=0.0): -0.5978 (correct endpoint direction).
No single step moves against the expected direction beyond the 0.01 noise floor.
Global maximum sits at alpha=0.0 (the expected endpoint) -- no multi-step hump/dip either.

## Overall verdict

**Not cleanly monotonic across the full range** -- at least one metric has a single step that reverses beyond the noise floor; and at least one metric has a real multi-step hump/dip even where no single step alone exceeded the noise floor. See `../phase12d_notes.md` for what this means for whether the mechanism is a genuine, usable dial across its full range.

