# Phase 6, Step 2: also_buy Edge Similarity Distribution (SIGLIP)

n = 9450 within-sample also_buy edges (directed pairs, phase 1b's 1,872-product sample, SigLIP embeddings).

## Summary statistics

| Statistic | Value |
|---|---|
| Mean | 0.6712 |
| Median | 0.6817 |
| Std dev | 0.1461 |
| Min | 0.1850 |
| Max | 1.0000 |
| Skewness | -0.1370 |
| Excess kurtosis | -0.6624 |
| Sarle's bimodality coefficient | 0.4358 (0.555 = rough bimodality threshold) |

## Detected KDE modes: 2

Peak location(s): 0.5896, 0.7504

![distribution plot](similarity_distribution_siglip.png)

## Shape reading

**Mixed signal**: the KDE shows 2 prominent peaks (at similarity ~0.590, ~0.750), suggesting some structure, but Sarle's bimodality coefficient (0.436) stays below the 0.555 rough threshold typically associated with bimodality -- the peaks may be real but modest, not a clean two-group split. Needs the qualitative check to interpret.

