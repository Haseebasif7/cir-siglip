# Phase 12d: Alpha Interpolation Sweep -- Results

Phase 12c's `ranking_distillation.pt` checkpoint, evaluated at 11 alpha values (0.0 to 1.0, step 0.1) -- no retraining. Full CIR benchmark: all 26494 pool slots, 29681 queries. Diagnostic metrics a/b on the same 1000-query sample (seed=42) used throughout phases 12/12b/12c; metrics c/d on the same 500-query overlap sample (seed=42).

| alpha | Recall@10 | Recall@30 | Recall@50 | Visual sim (a) | Co-occur hit rate (b) | Overlap w/ raw SigLIP (c) | Overlap w/ complement (d) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.0971 | 0.1875 | 0.2471 | 0.7217 | 0.1020 | 0.1708 | 1.0000 |
| 0.1 | 0.0989 | 0.1896 | 0.2488 | 0.7177 | 0.1030 | 0.1520 | 0.8692 |
| 0.2 | 0.1006 | 0.1899 | 0.2459 | 0.7182 | 0.1040 | 0.1430 | 0.7738 |
| 0.3 | 0.0983 | 0.1855 | 0.2413 | 0.7233 | 0.1050 | 0.1536 | 0.6954 |
| 0.4 | 0.0944 | 0.1804 | 0.2350 | 0.7304 | 0.1080 | 0.1800 | 0.6370 |
| 0.5 | 0.0893 | 0.1695 | 0.2255 | 0.7373 | 0.1080 | 0.2138 | 0.5746 |
| 0.6 | 0.0836 | 0.1597 | 0.2128 | 0.7425 | 0.1020 | 0.2584 | 0.5336 |
| 0.7 | 0.0781 | 0.1502 | 0.2018 | 0.7458 | 0.0960 | 0.2908 | 0.4894 |
| 0.8 | 0.0735 | 0.1421 | 0.1912 | 0.7479 | 0.0850 | 0.3126 | 0.4570 |
| 0.9 | 0.0696 | 0.1349 | 0.1815 | 0.7491 | 0.0790 | 0.3320 | 0.4260 |
| 1.0 | 0.0667 | 0.1307 | 0.1734 | 0.7498 | 0.0760 | 0.3428 | 0.4022 |

Expected directions per the brief: (a) increase as alpha rises toward 1; (b) increase as alpha falls toward 0; (c) increase as alpha rises toward 1; (d) decrease as alpha rises toward 1 (trivially 1.0 at alpha=0.0 itself, since that IS the complement-mode reference). See `monotonicity_check.md` for whether each metric actually holds this direction consistently across the full sweep, not just at the two endpoints.

