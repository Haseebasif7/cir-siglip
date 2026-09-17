# Phase 12, Step 4.2: Control-Effectiveness Diagnostic

Sample: 1000 CIR benchmark queries (seed=42, target sample size 1000), top-10 results compared under substitute mode (alpha=1.0) and complement mode (alpha=0.0), same shared checkpoint, same queries.

## Axis 1: visual similarity to the query (raw SigLIP cosine, averaged over top-10 results)

Premise: substitute mode should look MORE like plain visual similarity, so its top-10 results should have higher average raw-SigLIP similarity to the query than complement mode's.

- Substitute mode: 0.7193
- Complement mode: 0.7120
- **Direction CONFIRMED** (substitute > complement).

## Axis 2: match rate against real outfit co-occurrence ground truth (hit@10 on the true held-out target)

Premise: complement mode should better match real 'goes well with' outfit assembly, so its hit rate against the CIR benchmark's true target (a real Polyvore co-outfit partner, by construction) should be higher than substitute mode's.

- Substitute mode: 0.1280 (128/1000)
- Complement mode: 0.1280 (128/1000)
- **Direction NOT CONFIRMED** (complement <= substitute).

## Verdict

**Only one direction confirmed, and even that one is a weak effect.** The control knob moves
behavior as intended on axis 1 (0.7193 vs 0.7120, a ~1% relative gap) but the two modes' hit
rates on axis 2 are not just "not significantly different" -- they are *exactly* tied (128/1000
each). That exact tie is itself a strong clue about the mechanism, followed up directly below.

## Why: a direct mechanistic check on how different the two modes actually are

The near-identical Recall@K numbers in `results_table.md` (substitute 0.1329 vs complement
0.1335 @10, vs raw SigLIP's 0.0553) and axis 2's exact tie above both point the same
direction -- the two modes might not be producing meaningfully different rankings at all,
despite training with two different loss terms. Checked directly, on top of the two premises
already tested above:

- **Top-10 overlap between substitute mode's and complement mode's retrieved lists**, same
  500-query sample: **mean 76.4% of the top-10 slots are identical between the two modes**
  (median 80%, min 20%, max 100% -- every query checked, not a cherry-picked subset).
- **Per-item embedding similarity**: for the same 5,000 items, the average cosine similarity
  between an item's substitute-mode projection and its complement-mode projection is **0.829**
  -- the two "modes" of the same item point in nearly the same direction in the shared 128-d
  space.
- **Vector magnitudes, checked directly from the trained checkpoint**: the shared base
  projection's own output norm (before either mode vector is added) averages **2.356**; the
  learned `mode_substitute` vector has norm **1.186** and `mode_complement` has norm **0.256**.
  The mode vectors are real and non-zero (training did move them), but both are smaller than
  the base projection they're added to -- complement's mode vector in particular is roughly
  9x smaller than the base projection's own norm, so adding it barely rotates the final
  L2-normalized direction at all.

**Root cause**: the two loss terms are wildly mismatched in scale. Complement-mode's MNRL loss
starts around 4.4 and settles near 4.0-4.3 (cross-entropy over ~137 classes -- B-1 in-batch
negatives plus 8 explicit negatives); substitute-mode's MSE distillation loss starts near
0.015 and settles near 0.011 -- **roughly 400x smaller in raw magnitude**. Since both losses
are summed unweighted and backpropagate into the *same shared* `net` every step, the shared
base projection's weights are almost entirely shaped by the complement objective's much larger
gradient signal; the substitute loss mostly gets absorbed by the small `mode_substitute`
offset rather than by reshaping the shared representation itself. The result: both "modes" are
really just small, unequal perturbations of one underlying (complement-dominated) projection,
not two genuinely different embedding spaces -- which is exactly what the 76% top-10 overlap
and the exact-tie axis-2 hit rate are showing directly.

