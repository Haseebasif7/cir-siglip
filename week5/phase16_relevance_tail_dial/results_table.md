# Phase 16: Results Table

## Retrieval accuracy and axis checks across the alpha sweep

| alpha | Hit Rate@5 | Hit Rate@10 | Mean ref_count@5 | Mean ref_count@10 |
|---|---|---|---|---|
| 0.0 | 0.3194 | 0.3958 | 16.90 | 16.58 |
| 0.1 | 0.3194 | 0.3953 | 16.89 | 16.60 |
| 0.2 | 0.3184 | 0.3937 | 16.95 | 16.65 |
| 0.3 | 0.3184 | 0.3942 | 17.06 | 16.73 |
| 0.4 | 0.3178 | 0.3937 | 17.08 | 16.74 |
| 0.5 | 0.3184 | 0.3953 | 17.12 | 16.80 |
| 0.6 | 0.3189 | 0.3953 | 17.21 | 16.82 |
| 0.7 | 0.3184 | 0.3953 | 17.33 | 16.92 |
| 0.8 | 0.3173 | 0.3948 | 17.38 | 16.97 |
| 0.9 | 0.3184 | 0.3953 | 17.50 | 17.13 |
| 1.0 | 0.3184 | 0.3948 | 17.49 | 17.21 |

## Axis checks (endpoints, alpha=1.0 relevance vs alpha=0.0 tail-exposure)

- **Axis 1 (relevance mode's Hit Rate@K should exceed tail mode's, every K): FAIL** -- K=5: relevance=0.3184 vs tail=0.3194, K=10: relevance=0.3948 vs tail=0.3958
- **Axis 2 (tail mode's mean retrieved ref_count should be lower than relevance mode's, every K): PASS** -- K=5: relevance=17.49 vs tail=16.90, K=10: relevance=17.21 vs tail=16.58

## Field-standard long-tail metrics

Formulas from the week 5 literature review (Saha, Biswas, Das & Paitya's systematic review). Coverage@N and Tail-Coverage@N denominators are the TRUE catalog-wide counts from phase 3 (3,777,545 total items, 1,888,773 tail-tier items) -- not the 26,591-item reachable gallery. Expect small absolute Coverage/Tail-Coverage percentages as a result; this is a structural ceiling from this pipeline's finite embedded pool, not a computation error.

### N=10

| alpha | Coverage@N | Tail-Coverage@N | APRI | RPI |
|---|---|---|---|---|
| 0.0 | 0.2382% (8997) | 0.1910% (3608) | 16.58 | 0.5230 |
| 0.1 | 0.2382% (8998) | 0.1910% (3608) | 16.60 | 0.5231 |
| 0.2 | 0.2386% (9012) | 0.1909% (3605) | 16.65 | 0.5235 |
| 0.3 | 0.2386% (9012) | 0.1909% (3606) | 16.73 | 0.5237 |
| 0.4 | 0.2385% (9008) | 0.1909% (3605) | 16.74 | 0.5230 |
| 0.5 | 0.2387% (9016) | 0.1912% (3612) | 16.80 | 0.5232 |
| 0.6 | 0.2385% (9008) | 0.1913% (3613) | 16.82 | 0.5235 |
| 0.7 | 0.2386% (9015) | 0.1912% (3611) | 16.92 | 0.5234 |
| 0.8 | 0.2387% (9017) | 0.1916% (3618) | 16.97 | 0.5234 |
| 0.9 | 0.2383% (9003) | 0.1911% (3609) | 17.13 | 0.5231 |
| 1.0 | 0.2384% (9004) | 0.1904% (3597) | 17.21 | 0.5233 |

### N=20

| alpha | Coverage@N | Tail-Coverage@N | APRI | RPI |
|---|---|---|---|---|
| 0.0 | 0.3584% (13537) | 0.2977% (5623) | 16.23 | 0.5149 |
| 0.1 | 0.3588% (13554) | 0.2978% (5624) | 16.31 | 0.5154 |
| 0.2 | 0.3588% (13553) | 0.2974% (5618) | 16.43 | 0.5158 |
| 0.3 | 0.3592% (13569) | 0.2976% (5621) | 16.47 | 0.5159 |
| 0.4 | 0.3591% (13564) | 0.2972% (5613) | 16.53 | 0.5163 |
| 0.5 | 0.3589% (13557) | 0.2971% (5612) | 16.62 | 0.5162 |
| 0.6 | 0.3589% (13557) | 0.2968% (5606) | 16.66 | 0.5163 |
| 0.7 | 0.3586% (13548) | 0.2970% (5610) | 16.71 | 0.5163 |
| 0.8 | 0.3585% (13542) | 0.2965% (5601) | 16.72 | 0.5167 |
| 0.9 | 0.3580% (13525) | 0.2964% (5599) | 16.80 | 0.5170 |
| 1.0 | 0.3583% (13534) | 0.2968% (5606) | 16.92 | 0.5171 |

### Endpoint deltas (alpha=1.0 relevance vs alpha=0.0 tail-exposure)

- N=10: Coverage@N 0.2382% (tail) vs 0.2384% (relevance); Tail-Coverage@N 0.1910% (tail) vs 0.1904% (relevance); APRI 16.58 (tail) vs 17.21 (relevance); RPI 0.5230 (tail) vs 0.5233 (relevance).
- N=20: Coverage@N 0.3584% (tail) vs 0.3583% (relevance); Tail-Coverage@N 0.2977% (tail) vs 0.2968% (relevance); APRI 16.23 (tail) vs 16.92 (relevance); RPI 0.5149 (tail) vs 0.5171 (relevance).

## Contextual comparison: GUME (CIKM 2024), Clothing/Shoes/Jewelry category

Reported as context, not a matched-protocol baseline claim -- GUME requires a full user-item interaction graph, this project's mechanism is item-only; the two are not the same evaluation protocol (GUME: Recall@K on a held-out user-item interaction split; this phase: Hit-Rate@K on an item-to-item also_buy retrieval task over a different, smaller candidate pool). Numbers below are GUME's own published results on the same Amazon Clothing, Shoes, and Jewelry category this project uses throughout, confirmed directly from the paper's full text (`week5/literature_review/longtail_direction_literature_review.md`):

- GUME Recall@10 = 0.0703, Recall@20 = 0.1024 (vs. MENTOR, its strongest baseline: 0.0668 / 0.0989).
