# Phase 18b, Step 5: Axis Independence Check (Re-Run)

Same method as phase 18's step 7. For each fixed alpha2, does moving alpha1 alone reproduce the same axis-1 behavior regardless of alpha2? And the reverse for axis 2.

## Axis 1 (substitute <-> complement) at each fixed alpha2

| alpha2 | overlap_raw(a1=0) | overlap_raw(a1=1) | delta | hit_rate(a1=0) | hit_rate(a1=1) | delta |
|---|---|---|---|---|---|---|
| 0.0 | 0.2939 | 0.5769 | +0.2831 | 0.3307 | 0.4359 | -0.1052 |
| 0.25 | 0.3294 | 0.5779 | +0.2485 | 0.3536 | 0.4332 | -0.0796 |
| 0.5 | 0.4218 | 0.5848 | +0.1630 | 0.4151 | 0.4290 | -0.0139 |
| 0.75 | 0.4345 | 0.6166 | +0.1821 | 0.4263 | 0.4407 | -0.0144 |
| 1.0 | 0.4221 | 0.6462 | +0.2241 | 0.4204 | 0.4418 | -0.0214 |

**Axis 1 direction consistent across every alpha2 value: overlap-with-raw YES (deltas: [0.2831, 0.2485, 0.163, 0.1821, 0.2241]), hit-rate NO (deltas: [-0.1052, -0.0796, -0.0139, -0.0144, -0.0214]). Delta range: overlap [0.1630, 0.2831], hit-rate [-0.1052, -0.0139].**

## Axis 2 (relevance <-> tail-exposure) at each fixed alpha1

| alpha1 | ref_count(a2=0) | ref_count(a2=1) | delta | tail_frac(a2=0) | tail_frac(a2=1) | delta |
|---|---|---|---|---|---|---|
| 0.0 | 15.44 | 17.58 | +2.13 | 0.3757 | 0.3543 | +0.0214 |
| 0.25 | 15.99 | 17.84 | +1.85 | 0.3665 | 0.3532 | +0.0134 |
| 0.5 | 16.24 | 17.45 | +1.21 | 0.3545 | 0.3526 | +0.0019 |
| 0.75 | 16.48 | 17.17 | +0.70 | 0.3562 | 0.3521 | +0.0041 |
| 1.0 | 16.53 | 16.94 | +0.41 | 0.3596 | 0.3510 | +0.0087 |

**Axis 2 direction consistent across every alpha1 value: ref_count YES (deltas: [2.13, 1.85, 1.21, 0.7, 0.41]), tail_fraction YES (deltas: [0.0214, 0.0134, 0.0019, 0.0041, 0.0087]). Delta range: ref_count [0.41, 2.13], tail_fraction [0.0019, 0.0214].**

## Verdict

**The two axes do NOT behave fully independently** -- at least one directional check failed at some value of the other axis. Reported plainly.
