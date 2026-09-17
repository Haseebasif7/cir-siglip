# Phase 36: Equivalence Testing (paired bootstrap over test queries)

Per-query hit/miss captured in step 1 for all 29681 test queries, identical query set for every system (`n_skipped=0` everywhere, aggregates reproduced -- see `data/reproduction_check.json`). Paired percentile bootstrap over query indices, B=10000, seed 20260915, the same resamples for every comparison. **Pre-declared equivalence bound (phase36_notes.md): delta = 2% relative of `ours_ens`'s Recall@K = +/-0.0038 / +/-0.0065 / +/-0.0082 absolute at K=10/30/50.** Two systems are equivalent at K iff the 90% CI of their difference lies inside [-delta, +delta] (TOST, alpha=0.05).

## Per-system Recall@K with 95% bootstrap CI

| System | R@10 [95% CI] | R@30 [95% CI] | R@50 [95% CI] |
|---|---|---|---|
| `ours_ens` -- Ours, mean-pooled projection, 10-seed ensemble (phase 28) | 0.1904 [0.1859, 0.1949] | 0.3267 [0.3214, 0.3320] | 0.4079 [0.4023, 0.4135] |
| `ours_solo` -- Ours, mean-pooled projection, single model (phase 27 text_only) | 0.1656 [0.1613, 0.1699] | 0.2947 [0.2895, 0.2999] | 0.3733 [0.3677, 0.3787] |
| `ot_ens` -- OutfitTransformer mechanism, frozen backbone, matched optimization, 3-seed ensemble (phase 32) | 0.1897 [0.1852, 0.1942] | 0.3246 [0.3192, 0.3299] | 0.4019 [0.3963, 0.4073] |
| `ot_solo` -- OutfitTransformer mechanism, frozen backbone, matched optimization, single model (phase 31/32 seed 42) | 0.1799 [0.1755, 0.1842] | 0.3111 [0.3057, 0.3163] | 0.3844 [0.3789, 0.3899] |
| `csa_ens` -- CSA-Net mechanism, frozen backbone, matched optimization + random negatives, 3-seed ensemble (phase 34) | 0.1674 [0.1632, 0.1716] | 0.2860 [0.2809, 0.2911] | 0.3586 [0.3532, 0.3639] |

## Paired differences (A minus B)

| Pair | K | R@K A | R@K B | diff | diff % of B | 95% CI | 90% CI | +/-delta | TOST | McNemar b / c | McNemar p |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ours_ens` - `ot_ens` | 10 | 0.1904 | 0.1897 | +0.0007 | +0.36% | [-0.0030, +0.0043] | [-0.0025, +0.0037] | 0.0038 | **equivalent** | 1590 / 1570 | 0.735 |
| `ours_ens` - `ot_ens` | 30 | 0.3267 | 0.3246 | +0.0022 | +0.66% | [-0.0021, +0.0065] | [-0.0014, +0.0057] | 0.0065 | **equivalent** | 2066 / 2002 | 0.323 |
| `ours_ens` - `ot_ens` | 50 | 0.4079 | 0.4019 | +0.0060 | +1.49% | [+0.0016, +0.0103] | [+0.0024, +0.0097] | 0.0082 | not equivalent | 2293 / 2115 | 0.00767 |
| `ours_ens` - `csa_ens` | 10 | 0.1904 | 0.1674 | +0.0230 | +13.75% | [+0.0187, +0.0272] | [+0.0194, +0.0265] | 0.0038 | not equivalent | 2390 / 1707 | 1.27e-26 |
| `ours_ens` - `csa_ens` | 30 | 0.3267 | 0.2860 | +0.0407 | +14.24% | [+0.0360, +0.0457] | [+0.0367, +0.0449] | 0.0065 | not equivalent | 3336 / 2127 | 1.59e-60 |
| `ours_ens` - `csa_ens` | 50 | 0.4079 | 0.3586 | +0.0493 | +13.75% | [+0.0444, +0.0543] | [+0.0451, +0.0536] | 0.0082 | not equivalent | 3711 / 2248 | 8.58e-81 |
| `ot_ens` - `csa_ens` | 10 | 0.1897 | 0.1674 | +0.0223 | +13.35% | [+0.0187, +0.0260] | [+0.0193, +0.0254] | 0.0038 | not equivalent | 1864 / 1201 | 3.36e-33 |
| `ot_ens` - `csa_ens` | 30 | 0.3246 | 0.2860 | +0.0386 | +13.49% | [+0.0344, +0.0428] | [+0.0351, +0.0421] | 0.0065 | not equivalent | 2649 / 1504 | 2.13e-71 |
| `ot_ens` - `csa_ens` | 50 | 0.4019 | 0.3586 | +0.0433 | +12.07% | [+0.0389, +0.0478] | [+0.0396, +0.0470] | 0.0082 | not equivalent | 2889 / 1604 | 6.56e-83 |
| `ours_solo` - `ot_solo` | 10 | 0.1656 | 0.1799 | -0.0143 | -7.96% | [-0.0184, -0.0104] | [-0.0178, -0.0110] | 0.0038 | not equivalent | 1611 / 2036 | 2.08e-12 |
| `ours_solo` - `ot_solo` | 30 | 0.2947 | 0.3111 | -0.0164 | -5.26% | [-0.0210, -0.0118] | [-0.0202, -0.0126] | 0.0065 | not equivalent | 2193 / 2679 | 3.55e-12 |
| `ours_solo` - `ot_solo` | 50 | 0.3733 | 0.3844 | -0.0111 | -2.89% | [-0.0160, -0.0064] | [-0.0153, -0.0071] | 0.0082 | not equivalent | 2481 / 2811 | 6.07e-06 |

McNemar `b` = queries A hits and B misses; `c` = A misses and B hits; exact two-sided binomial test on the discordant pairs. `TOST` is the pre-declared equivalence verdict; the McNemar p answers the different question of whether the two systems' hit sets differ at all.

## Recorded cross-seed variability (validation Recall@10 std, from each phase's `individual_seeds.md`)

| System | per-seed val R@10 std |
|---|---|
| ours (10 seeds, phase 28 individual_seeds.md) | 0.0016 |
| OutfitTransformer (3 seeds, phase 32 individual_seeds.md) | 0.00074 |
| CSA-Net random negatives (3 seeds, phase 34 individual_seeds.md) | 0.00087 |

Query-sampling uncertainty (the bootstrap CIs above) and training-seed uncertainty (this table) are different sources; a reader needs both. The seed spreads are on the validation benchmark and were not recomputed here.

## Per-category difference, `ours_ens` minus `ot_ens` (Recall@10, 95% CI)

| Category | n | R@10 A | R@10 B | diff | 95% CI |
|---|---|---|---|---|---|
| accessories | 1361 | 0.2403 | 0.2373 | +0.0029 | [-0.0162, +0.0206] |
| all-body | 3182 | 0.2316 | 0.2376 | -0.0060 | [-0.0179, +0.0060] |
| bags | 3382 | 0.1928 | 0.1886 | +0.0041 | [-0.0065, +0.0151] |
| bottoms | 3361 | 0.1663 | 0.1702 | -0.0039 | [-0.0140, +0.0060] |
| hats | 1401 | 0.2684 | 0.2463 | +0.0221 | [+0.0021, +0.0421] |
| jewellery | 3352 | 0.1429 | 0.1489 | -0.0060 | [-0.0167, +0.0048] |
| outerwear | 3244 | 0.1643 | 0.1720 | -0.0077 | [-0.0182, +0.0031] |
| scarves | 914 | 0.3042 | 0.3129 | -0.0088 | [-0.0328, +0.0153] |
| shoes | 3315 | 0.2287 | 0.2175 | +0.0112 | [-0.0003, +0.0223] |
| sunglasses | 2906 | 0.1242 | 0.1101 | +0.0141 | [+0.0028, +0.0255] |
| tops | 3263 | 0.1811 | 0.1879 | -0.0067 | [-0.0178, +0.0043] |

## Per-category difference, `ours_ens` minus `csa_ens` (Recall@10, 95% CI)

| Category | n | R@10 A | R@10 B | diff | 95% CI |
|---|---|---|---|---|---|
| accessories | 1361 | 0.2403 | 0.2226 | +0.0176 | [-0.0044, +0.0389] |
| all-body | 3182 | 0.2316 | 0.2027 | +0.0289 | [+0.0148, +0.0434] |
| bags | 3382 | 0.1928 | 0.1721 | +0.0207 | [+0.0080, +0.0334] |
| bottoms | 3361 | 0.1663 | 0.1503 | +0.0161 | [+0.0042, +0.0280] |
| hats | 1401 | 0.2684 | 0.2191 | +0.0493 | [+0.0278, +0.0714] |
| jewellery | 3352 | 0.1429 | 0.1441 | -0.0012 | [-0.0128, +0.0104] |
| outerwear | 3244 | 0.1643 | 0.1492 | +0.0151 | [+0.0031, +0.0274] |
| scarves | 914 | 0.3042 | 0.2659 | +0.0383 | [+0.0120, +0.0656] |
| shoes | 3315 | 0.2287 | 0.1816 | +0.0471 | [+0.0341, +0.0600] |
| sunglasses | 2906 | 0.1242 | 0.0967 | +0.0275 | [+0.0151, +0.0399] |
| tops | 3263 | 0.1811 | 0.1633 | +0.0178 | [+0.0052, +0.0297] |

## Per-category difference, `ot_ens` minus `csa_ens` (Recall@10, 95% CI)

| Category | n | R@10 A | R@10 B | diff | 95% CI |
|---|---|---|---|---|---|
| accessories | 1361 | 0.2373 | 0.2226 | +0.0147 | [-0.0037, +0.0331] |
| all-body | 3182 | 0.2376 | 0.2027 | +0.0349 | [+0.0223, +0.0478] |
| bags | 3382 | 0.1886 | 0.1721 | +0.0166 | [+0.0056, +0.0275] |
| bottoms | 3361 | 0.1702 | 0.1503 | +0.0199 | [+0.0101, +0.0298] |
| hats | 1401 | 0.2463 | 0.2191 | +0.0271 | [+0.0093, +0.0450] |
| jewellery | 3352 | 0.1489 | 0.1441 | +0.0048 | [-0.0054, +0.0146] |
| outerwear | 3244 | 0.1720 | 0.1492 | +0.0228 | [+0.0114, +0.0339] |
| scarves | 914 | 0.3129 | 0.2659 | +0.0470 | [+0.0241, +0.0711] |
| shoes | 3315 | 0.2175 | 0.1816 | +0.0359 | [+0.0250, +0.0471] |
| sunglasses | 2906 | 0.1101 | 0.0967 | +0.0134 | [+0.0031, +0.0234] |
| tops | 3263 | 0.1879 | 0.1633 | +0.0245 | [+0.0138, +0.0349] |

Interpretation is written in `phase36_notes.md` after reading these tables, not generated here.
