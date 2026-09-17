# Phase 16: Training Pool / Candidate Gallery Summary

## Reused infrastructure

- Phase 7 training pool embeddings: `week3/phase7_learned_compatibility/embeddings/siglip_base.npz` (24719 items, 768-d SigLIP)
- Phase 1b eval sample embeddings: `week2/phase1b_category_balanced/embeddings/siglip_base.npz` (1872 items, 768-d SigLIP)
- Phase 3 popularity/tier lookup: `week2/phase3_popularity_eda/data/popularity_lookup.csv` (3777545 catalog-wide asins)

## Disjointness (phase 7 pool vs phase 1b eval sample)

Overlap between phase 7's 24719-item pool and phase 1b's 1872-item eval sample: **0**. This re-asserts, directly against the embedding files this phase actually loads, the same zero-overlap result already proven and recorded at `week3/phase7_learned_compatibility/data/exclusion_check.md` (checked against the full 27,970-product raw pool, of which the 24,719-item cleaned pool used here is a strict subset).

## Candidate gallery

Merged gallery size: **26591** items (24719 + 1872, 0 duplicates found).

Role split: phase 1b's 1,872 items remain the **queries** (clean, disjoint-from-training, this project's standard held-out eval set, unchanged role). The full merged gallery of 26591 items is the **retrievable candidate pool** every query searches against for both the Hit-Rate/CIR-style accuracy sweep and the field-standard popularity metrics.

## Tier composition of the merged gallery

| Tier | Count | % of gallery |
|---|---|---|
| head | 11366 | 42.74% |
| mid | 3732 | 14.03% |
| tail | 11493 | 43.22% |
| unknown | 0 | 0.00% |

Tail-tier items: **11493** (43.22% of the 26591-item gallery), against a decision rule of >= 2% and >= 200 absolute tail items before this gallery is considered adequate for the field-standard tail metrics.

**Verdict: adequate.** No supplementary sampling needed -- proceeding with this 26591-item merged gallery as-is for the rest of the phase.

## Important scoping note for the field-standard metrics (step 5)

Coverage@N and Tail-Coverage@N are computed later against the TRUE catalog-wide denominators from phase 3 (3,777,545 total items; 1,888,773 tail-tier items), not against this gallery's size (26591 items) -- the brief requires the field's own catalog-relative definitions, not an approximation. This gallery only affects the NUMERATOR (what items are reachable to be recommended at all); expect resulting Coverage@N/Tail-Coverage@N percentages to be small (upper-bounded by roughly 0.704% for any metric with this gallery as its reachable-item source), reported as an honest structural ceiling of this pipeline's finite embedded pool, not normalized away.
