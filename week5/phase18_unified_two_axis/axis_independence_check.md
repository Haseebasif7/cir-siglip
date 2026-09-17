# Phase 18, Step 7: Axis Independence Check -- The Central Question

For each fixed alpha2, does moving alpha1 alone reproduce the same axis-1 behavior (overlap-with-raw increasing, hit rate decreasing) regardless of alpha2's value? And the reverse: for each fixed alpha1, does moving alpha2 alone reproduce the same axis-2 behavior (ref_count decreasing, tail_fraction increasing) regardless of alpha1's value?

## Axis 1 (substitute <-> complement) at each fixed alpha2

| alpha2 | overlap_raw(a1=0) | overlap_raw(a1=1) | delta | hit_rate(a1=0) | hit_rate(a1=1) | delta |
|---|---|---|---|---|---|---|
| 0.0 | 0.3047 | 0.6638 | +0.3591 | 0.3381 | 0.4546 | -0.1165 |
| 0.25 | 0.3443 | 0.6429 | +0.2987 | 0.3611 | 0.4503 | -0.0892 |
| 0.5 | 0.4350 | 0.6114 | +0.1764 | 0.4284 | 0.4364 | -0.0080 |
| 0.75 | 0.4462 | 0.6495 | +0.2033 | 0.4412 | 0.4460 | -0.0048 |
| 1.0 | 0.4343 | 0.6674 | +0.2331 | 0.4311 | 0.4594 | -0.0283 |

**Axis 1 direction consistent across every alpha2 value: overlap-with-raw YES (deltas: [0.3591, 0.2987, 0.1764, 0.2033, 0.2331]), hit-rate NO (deltas: [-0.1165, -0.0892, -0.008, -0.0048, -0.0283]). Delta range: overlap [0.1764, 0.3591], hit-rate [-0.1165, -0.0048].**

## Axis 2 (relevance <-> tail-exposure) at each fixed alpha1

| alpha1 | ref_count(a2=0) | ref_count(a2=1) | delta | tail_frac(a2=0) | tail_frac(a2=1) | delta |
|---|---|---|---|---|---|---|
| 0.0 | 15.39 | 18.84 | +3.46 | 0.3675 | 0.3417 | +0.0257 |
| 0.25 | 15.40 | 18.45 | +3.04 | 0.3646 | 0.3435 | +0.0211 |
| 0.5 | 15.96 | 17.37 | +1.41 | 0.3502 | 0.3443 | +0.0059 |
| 0.75 | 17.41 | 16.95 | -0.46 | 0.3362 | 0.3415 | -0.0052 |
| 1.0 | 17.23 | 16.86 | -0.38 | 0.3303 | 0.3415 | -0.0112 |

**Axis 2 direction consistent across every alpha1 value: ref_count NO (deltas: [3.46, 3.04, 1.41, -0.46, -0.38]), tail_fraction NO (deltas: [0.0257, 0.0211, 0.0059, -0.0052, -0.0112]). Delta range: ref_count [-0.46, 3.46], tail_fraction [-0.0112, 0.0257].**

## Verdict

**The two axes do NOT behave fully independently** -- at least one directional check failed at some value of the other axis. Reported plainly: this means the two dedicated-capacity axes interact to some degree even though each one works on its own at its own endpoints.
