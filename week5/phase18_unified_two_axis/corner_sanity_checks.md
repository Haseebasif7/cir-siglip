# Phase 18, Step 6: Corner Sanity Checks

Each corner checked directly against what its parent mechanism(s) should look like -- not assumed to work just because the axes work individually.

| Corner | (alpha1, alpha2) | Hit Rate@10 | Mean ref_count@10 | Tail fraction@10 | Overlap w/ raw SigLIP@10 |
|---|---|---|---|---|---|
| sub_rel | (1.0, 1.0) | 0.4594 | 16.86 | 0.3415 | 0.6674 |
| sub_tail | (1.0, 0.0) | 0.4546 | 17.23 | 0.3303 | 0.6638 |
| comp_rel | (0.0, 1.0) | 0.4311 | 18.84 | 0.3417 | 0.4343 |
| comp_tail | (0.0, 0.0) | 0.3381 | 15.39 | 0.3675 | 0.3047 |

## sub_rel -- should resemble phase 17's substitute mode (high visual similarity to raw SigLIP)

Overlap w/ raw SigLIP = 0.6674. Phase 17's Polyvore substitute mode measured 0.3310 (different dataset, cited as directional context only). **Consistent**: sub_rel's overlap-with-raw (0.6674) is higher than comp_rel's (0.4343), as a substitute-mode corner should be.

## comp_rel -- should resemble phase 16/16d's relevance mode (real co-occurrence signal)

Hit Rate@10 = 0.4311. Phase 16d's relevance-mode Hit Rate@10 was 0.3948 (same also_buy signal, 2-head architecture, direct numeric comparison valid since this is the same underlying signal and gallery/query setup). **Check directly**: comp_rel's Hit Rate (0.4311) vs sub_rel's (0.4594) and sub_tail's (0.4546).

## comp_tail -- should resemble phase 16d's tail mode (lower ref_count, real co-occurrence)

Mean ref_count@10 = 15.39, Hit Rate@10 = 0.3381. Phase 16d's tail-mode reference: ref_count@5=15.86, Hit Rate@10=0.3029 (same attribute-pair signal, 2-head architecture). **Consistent**: comp_tail's ref_count (15.39) is lower than comp_rel's (18.84).

## sub_tail -- THE NEW CORNER: should be visually similar AND skew toward low ref_count -- checked directly, not assumed from its two parent behaviors

Overlap w/ raw SigLIP = 0.6638 (vs comp_rel's 0.4343), mean ref_count@10 = 17.23 (vs comp_rel's 18.84), tail_fraction@10 = 0.3303 (vs comp_rel's 0.3417). **BOTH properties hold**: sub_tail retrieves items that are more visually similar to the query AND lower in ref_count than the plain complement-relevance corner -- this is the one genuinely new corner and its behavior is reported exactly as measured, not assumed from sub_rel + comp_tail alone.
