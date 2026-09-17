# Phase 5, Step 2: Hit Rate@K by Ground-Truth Signal (SigLIP and FashionCLIP)

Same 1,872-product sample and same top-10 retrieved lists as phase 1b (no recomputation) -- only the ground truth used for scoring changes per row.

| Technique | Signal | Overlap refs (within sample) | Trustworthy (>=20 refs)? | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|---|---|---|
| SigLIP | also_viewed only | 0 | **NO -- too few refs** | n/a | n/a | n/a | n/a |
| SigLIP | also_buy only | 9450 | yes | 0.502 | 0.578 | 0.191 | 0.144 |
| SigLIP | combined (reference) | 9450 | yes | 0.502 | 0.578 | 0.191 | 0.144 |
| FashionCLIP | also_viewed only | 0 | **NO -- too few refs** | n/a | n/a | n/a | n/a |
| FashionCLIP | also_buy only | 9450 | yes | 0.468 | 0.540 | 0.180 | 0.136 |
| FashionCLIP | combined (reference) | 9450 | yes | 0.468 | 0.540 | 0.180 | 0.136 |

**also_viewed_only is flagged not trustworthy for both techniques (0 overlap refs, far below the 20-ref threshold) -- there is no also_viewed ground truth anywhere in this sample (see overlap_check.md), so Hit Rate@K against it is 0.000/0.000 by construction (no query ever has a nonempty also_viewed ground truth to hit), not a genuine measurement of retrieval quality against a substitute-like signal.**

also_buy_only is numerically identical to combined for both techniques -- expected, since combined = also_buy UNION also_viewed, and also_viewed contributes the empty set for every product in this sample.

