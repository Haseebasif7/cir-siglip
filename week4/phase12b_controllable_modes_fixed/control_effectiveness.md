# Phase 12b, Step 4.2: Control-Effectiveness Diagnostic, Before/After Comparison

Same procedure as phase 12's `control_effectiveness.md` (sample size 1000, seed=42, top-10), on the retrained `controllable_modes_fixed.pt` checkpoint.

## Axis 1: visual similarity to the query (raw SigLIP cosine, top-10 average)

- Substitute mode: phase 12 = 0.7193, **phase 12b = 0.7233** (delta +0.0040)
- Complement mode: phase 12 = 0.7120, **phase 12b = 0.7228** (delta +0.0108)
- **Direction CONFIRMED** (substitute > complement); gap = 0.0005 (phase 12's gap was 0.0073).

## Axis 2: match rate against real outfit co-occurrence (hit@10 on the true target)

- Substitute mode: phase 12 = 0.1280, **phase 12b = 0.1250** (delta -0.0030)
- Complement mode: phase 12 = 0.1280, **phase 12b = 0.1180** (delta -0.0100)
- **Direction NOT CONFIRMED** (complement <= substitute); phase 12 was an exact tie (128/1000 each) -- this tie is now broken.

## Follow-up checks (these are what actually exposed the problem in phase 12)

- Mean top-10 overlap between the two modes (n=500 queries): phase 12 = 0.7636, **phase 12b = 0.5102** (delta -0.2534)
- Mean per-item cosine(z_sub, z_comp) across 5,000 items: phase 12 = 0.8288, **phase 12b = 0.5844** (delta -0.2444)

## Extra follow-up: does substitute mode's retrieval now actually resemble RAW SigLIP's?

The follow-up checks above show the two modes genuinely diverged from EACH OTHER (overlap
dropped 76.4% -> 51.0%). But that alone doesn't confirm substitute mode moved *toward*
genuine visual similarity specifically -- it could have diverged toward something else
entirely. Checked directly, not asked for by the brief but necessary to settle this: top-10
overlap between each mode's retrieval and RAW SigLIP's own top-10 retrieval, same 500-query
sample:

- Substitute mode vs raw SigLIP: mean overlap = **0.1612**
- Complement mode vs raw SigLIP: mean overlap = **0.1754**

**Substitute mode's retrieval is not more similar to raw SigLIP's than complement mode's is
-- if anything, slightly less.** This directly answers the open question the axis-1/axis-2
results leave hanging: substitute mode moved away from complement mode, but not toward raw
SigLIP. The PCA-128 target (73.7% of raw SigLIP's total variance retained, see
`../data/pca_variance_check.md`) apparently preserves enough of raw SigLIP's *global*
structure to give substitute mode a genuinely different geometry from complement mode's, but
not enough of its *local nearest-neighbor* structure -- the fine-grained relationships that
actually determine which specific items rank in a top-10 -- to make substitute mode's
retrieval behave like real visual-similarity search. Variance retention is a global,
aggregate property; retrieval quality depends on local neighborhood structure, and PCA
truncation does not guarantee the latter survives just because most of the former does.

## Verdict

**Partial improvement, with an important caveat that changes the overall picture.** The two
modes are now genuinely, substantially different from each other (overlap and per-item
cosine both dropped by ~25 points) -- that part of the fix worked. But the specific
behavioral premises this mechanism is supposed to deliver did NOT clearly improve: axis 1's
gap actually shrank in absolute terms (0.0073 -> 0.0005) even though both raw values rose,
axis 2 flipped to the wrong direction (a small, likely noise-level reversal, not a large
one), and -- the decisive check -- substitute mode's retrieval is not measurably closer to
raw SigLIP's own retrieval than complement mode's is. The fix succeeded at making the two
modes *different*; it did not succeed at making substitute mode specifically resemble
*plain visual similarity*, which was the actual goal. See `../phase12b_notes.md` for the
full go/no-go verdict and what this means for whether to pursue the fuller architecture.

