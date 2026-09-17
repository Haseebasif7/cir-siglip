# Phase 18b, Step 4 & Step 6: Corner Sanity Checks

## Step 4 (this phase's specific, narrow target -- checked FIRST, before anything broader)

Does sub_tail's PURE corner now retrieve lower mean ref_count / higher tail fraction than sub_rel's PURE corner? Phase 18 found this reversed even unblended (sub_tail ref_count=17.23 > sub_rel's 16.86).

| | sub_rel (alpha1=1.0, alpha2=1.0) | sub_tail (alpha1=1.0, alpha2=0.0) |
|---|---|---|
| Mean ref_count@10 -- phase 18 | 16.86 | 17.23 |
| Mean ref_count@10 -- **phase 18b** | 16.94 | **16.53** |
| Tail fraction@10 -- phase 18 | 0.3415 | 0.3303 |
| Tail fraction@10 -- **phase 18b** | 0.3510 | **0.3596** |

**FIXED**: sub_tail's ref_count (16.53) is now lower than sub_rel's (16.94), and sub_tail's tail_fraction (0.3596) is higher than sub_rel's (0.3510).

## Step 6: full corner sanity table (same format as phase 18)

| Corner | (alpha1, alpha2) | Hit Rate@10 | Mean ref_count@10 | Tail fraction@10 | Overlap w/ raw SigLIP@10 |
|---|---|---|---|---|---|
| sub_rel | (1.0, 1.0) | 0.4418 | 16.94 | 0.3510 | 0.6462 |
| sub_tail | (1.0, 0.0) | 0.4359 | 16.53 | 0.3596 | 0.5769 |
| comp_rel | (0.0, 1.0) | 0.4204 | 17.58 | 0.3543 | 0.4221 |
| comp_tail | (0.0, 0.0) | 0.3307 | 15.44 | 0.3757 | 0.2939 |

## Direct before/after comparison to phase 18

| Corner | Metric | Phase 18 | Phase 18b |
|---|---|---|---|
| sub_rel | hit_rate | 0.4594 | 0.4418 |
| sub_rel | mean_ref_count | 16.86 | 16.94 |
| sub_rel | tail_fraction | 0.3415 | 0.3510 |
| sub_rel | overlap_with_raw | 0.6674 | 0.6462 |
| sub_tail | hit_rate | 0.4546 | 0.4359 |
| sub_tail | mean_ref_count | 17.23 | 16.53 |
| sub_tail | tail_fraction | 0.3303 | 0.3596 |
| sub_tail | overlap_with_raw | 0.6638 | 0.5769 |
| comp_rel | hit_rate | 0.4311 | 0.4204 |
| comp_rel | mean_ref_count | 18.84 | 17.58 |
| comp_rel | tail_fraction | 0.3417 | 0.3543 |
| comp_rel | overlap_with_raw | 0.4343 | 0.4221 |
| comp_tail | hit_rate | 0.3381 | 0.3307 |
| comp_tail | mean_ref_count | 15.39 | 15.44 |
| comp_tail | tail_fraction | 0.3675 | 0.3757 |
| comp_tail | overlap_with_raw | 0.3047 | 0.2939 |
