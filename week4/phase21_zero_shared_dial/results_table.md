# Phase 21: Alpha Sweep -- Blending Two Fully Independent Networks

Full CIR benchmark (26494 pool slots, 29681 queries), 11-point alpha sweep, `z_blend = normalize((1-alpha)*z_complement + alpha*z_substitute)` -- the complement and substitute heads share ZERO parameters, trained fully independently, this blend is the only place their outputs ever interact. Diagnostic metrics a/b on the same 1000-query sample (seed=42) used throughout phases 12/12b/12c/12d/17; metrics c/d on the same 500-query overlap sample (seed=42).

| alpha | Recall@10 | Recall@30 | Recall@50 | Visual sim (a) | Co-occur hit rate (b) | Overlap w/ raw SigLIP (c) | Overlap w/ complement (d) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.1317 | 0.2464 | 0.3216 | 0.7101 | 0.1390 | 0.0968 | 1.0000 |
| 0.1 | 0.1305 | 0.2450 | 0.3189 | 0.7115 | 0.1390 | 0.1026 | 0.9120 |
| 0.2 | 0.1289 | 0.2418 | 0.3152 | 0.7146 | 0.1290 | 0.1110 | 0.8158 |
| 0.3 | 0.1252 | 0.2362 | 0.3081 | 0.7205 | 0.1250 | 0.1290 | 0.6976 |
| 0.4 | 0.1188 | 0.2246 | 0.2939 | 0.7281 | 0.1280 | 0.1622 | 0.5614 |
| 0.5 | 0.1083 | 0.2047 | 0.2728 | 0.7355 | 0.1080 | 0.2066 | 0.4188 |
| 0.6 | 0.0922 | 0.1797 | 0.2378 | 0.7418 | 0.0970 | 0.2646 | 0.2884 |
| 0.7 | 0.0774 | 0.1495 | 0.2011 | 0.7463 | 0.0830 | 0.3098 | 0.1904 |
| 0.8 | 0.0625 | 0.1229 | 0.1662 | 0.7488 | 0.0670 | 0.3344 | 0.1332 |
| 0.9 | 0.0523 | 0.1034 | 0.1421 | 0.7493 | 0.0560 | 0.3384 | 0.1008 |
| 1.0 | 0.0470 | 0.0923 | 0.1291 | 0.7488 | 0.0510 | 0.3230 | 0.0846 |

Expected directions per this project's standing convention: (a) increase as alpha rises toward 1; (b) increase as alpha falls toward 0; (c) increase as alpha rises toward 1; (d) decrease as alpha rises toward 1 (trivially 1.0 at alpha=0.0 itself).

## Endpoints against phase 9 (complement, alone) and phase 17 (this project's best prior dial)

| Configuration | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Phase 9 (standalone, no dial) | 0.1317 | -- | -- |
| Phase 17 complement endpoint (alpha=0.0, dedicated capacity, one shared layer) | 0.1202 | -- | -- |
| **Phase 21 complement endpoint (alpha=0.0, zero shared parameters)** | **0.1317** | **0.2464** | **0.3216** |
| Phase 17 substitute endpoint (alpha=1.0) | 0.0573 | -- | -- |
| Phase 21 substitute endpoint (alpha=1.0, zero shared parameters) | 0.0470 | 0.0923 | 0.1291 |

