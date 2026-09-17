# Phase 6, Step 2: also_buy Edge Similarity Distribution (FASHIONCLIP)

n = 9450 within-sample also_buy edges (directed pairs, phase 1b's 1,872-product sample, SigLIP embeddings).

## Summary statistics

| Statistic | Value |
|---|---|
| Mean | 0.5959 |
| Median | 0.6072 |
| Std dev | 0.1871 |
| Min | 0.1008 |
| Max | 1.0000 |
| Skewness | -0.0980 |
| Excess kurtosis | -0.7599 |
| Sarle's bimodality coefficient | 0.4507 (0.555 = rough bimodality threshold) |

## Detected KDE modes: 2

Peak location(s): 0.5427, 0.6985

![distribution plot](similarity_distribution_fashionclip.png)

## Shape reading

**Mixed signal**: the KDE shows 2 prominent peaks (at similarity ~0.543, ~0.698), suggesting some structure, but Sarle's bimodality coefficient (0.451) stays below the 0.555 rough threshold typically associated with bimodality -- the peaks may be real but modest, not a clean two-group split. Needs the qualitative check to interpret.

