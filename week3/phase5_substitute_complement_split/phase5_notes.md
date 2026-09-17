# Phase 5: Substitute vs Complement Ground Truth Split

## Context

The professor's actual goal is a hybrid recommender balancing visually similar
(substitute) products against complementary (goes-with-but-doesn't-look-alike)
products. McAuley 2015 / VBPR frame `also_viewed` as plausibly substitute-like
and `also_buy` as plausibly complement-like. Every phase so far has unioned
`also_buy` and `also_viewed` into one combined ground truth signal. This phase
was meant to check, before any training effort, whether the two fields behave
differently enough in this data to justify separate learned representations.

Diagnostic only, per the brief: no training, no new embeddings, no new
downloads. Reused phase 1b's exact sample and phase 1b's existing SigLIP/
FashionCLIP top-10 retrieval lists throughout.

## Step 1 finding: also_viewed is not merely low-overlap with also_buy, it is completely empty

Before scoring anything, step 1 checked pair-level overlap between the two
fields (`overlap_check.md`). Every one of the 1,872 sampled products has an
**empty** `also_viewed` list -- 0 nonempty out of 1,872. The same check against
phase 1's original 776-product sample also returns 0 nonempty out of 776.

This was verified as a genuine data property, not a sampling or download bug:
streaming the first 50,000 raw records directly from
`meta_Clothing_Shoes_and_Jewelry.json.gz` (the same source file every sample in
this project has come from since phase 1) found 0 nonempty `also_viewed`
fields against 8,092 nonempty `also_buy` fields. `also_viewed` exists as a key
in the JSON schema (documented in `week1/dataset_summary_week1.md`) but its
value is always `[]` in this specific hosted 2018 metadata dump.

**Practical consequence: this project has never actually had an also_viewed
signal.** Every "combined also_buy/also_viewed" ground truth used in phases
1, 1b, 2, 3, and 4 has, in effect, only ever been also_buy. The union operation
was harmless (unioning with the empty set changes nothing) but the intended
dual-signal design was never realized.

## Step 2 result: also_buy_only is numerically identical to combined; also_viewed_only is trivially zero

`results_table.md` has the full table. Summary:

| Technique | also_viewed only | also_buy only | Combined (reference) |
|---|---|---|---|
| SigLIP | HR@5 = 0.000 (flagged, 0 overlap refs) | HR@5 = 0.502 | HR@5 = 0.502 |
| FashionCLIP | HR@5 = 0.000 (flagged, 0 overlap refs) | HR@5 = 0.468 | HR@5 = 0.468 |

`also_buy_only` and `combined` match exactly (to rounding) for both
techniques, as expected once also_viewed is confirmed empty: combined =
also_buy UNION also_viewed = also_buy UNION {} = also_buy. `also_viewed_only`
is 0.000/0.000 for every K and every technique, with 0 within-sample overlap
references -- this fails the phase 1b trustworthy-reference threshold
(>=20 refs) as badly as a signal can fail it. It is not a genuine measurement
of "does visual similarity correlate with also_viewed" -- there is no
also_viewed ground truth to correlate against.

## Step 3: interpretation

The brief asks directly whether frozen visual similarity correlates more
strongly with `also_viewed` than `also_buy`. **This cannot be answered from
this data.** It isn't that the two scores are close together (which would
itself be an answerable, if less interesting, result) -- one side of the
comparison doesn't exist. Every "also_buy/also_viewed combined" result
reported in phases 1 through 4 has, in substance, been an also_buy-only
result the whole time.

## Step 4: category breakdown

**Skipped, per the brief's own condition** ("if step 2 finds a meaningful
difference between the two signals, break down by category"). Step 2 didn't
find a meaningful difference between two measurable signals -- it found that
one of the two signals doesn't exist in this data. A category breakdown of
"0.000 for every category, everywhere" would add nothing not already stated
here; `category_breakdown.md` was intentionally not produced for this phase.

## Recommendation

**Do not pursue learned substitute/complement representations against
`also_viewed` vs `also_buy` as currently framed, because the data doesn't
support the split.** This isn't a "signal isn't clean enough" finding in the
usual noisy-data sense (like the Uniforms/Traditional categories' sparse
ground truth from phase 1b) -- it's an outright absence of one of the two
signals in the specific metadata file this project has used throughout.

Before any training effort aimed at a substitute/complement distinction is
committed to, one of the following would need to happen first:

1. **Find a metadata source that actually populates also_viewed.** The 2018
   Amazon Review Data has more than one hosted copy/version; it's possible a
   different mirror or file (e.g. the original UCSD 2018 release rather than
   this "amazon_v2" host) has also_viewed populated. Worth a quick check
   before concluding the whole 2018 dataset lacks it -- only this specific
   hosted file has been confirmed empty here.
2. **Find a different proxy for the substitute/complement distinction** that
   doesn't depend on also_viewed at all -- e.g. within `also_buy` itself,
   using category/subcategory agreement between a product and its also_buy
   targets as a rough substitute (same subcategory) vs. complement (different
   subcategory) heuristic. Untested, but doesn't require new data.
3. **Reframe the hybrid recommender goal around what the data actually
   supports** -- treat `also_buy` as the only reliable relatedness signal in
   this dataset and build the substitute/complement balance some other way
   (e.g. visual-similarity threshold as the substitute axis, co-purchase
   frequency or category-diversity as the complement axis, rather than two
   separate ground-truth fields).

None of these has been attempted in this phase -- it was diagnostic only, per
the brief, and stops here with the recommendation above.
