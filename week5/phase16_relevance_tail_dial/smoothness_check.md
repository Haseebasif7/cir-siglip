# Phase 16, Step 6: Adjacent vs Distant Alpha Overlap

Full pairwise top-10 overlap matrix across all 11 alpha values, all 1872 phase 1b queries, computed from 04_evaluate_alpha_sweep.py's cached per-alpha retrieval lists -- same method as phase 12d's `03_adjacent_vs_distant_overlap.py`.

## Mean overlap as a function of |delta alpha|

| |delta alpha| | Mean top-10 overlap |
|---|---|
| 0.1 | 0.9931 |
| 0.2 | 0.9866 |
| 0.3 | 0.9805 |
| 0.4 | 0.9746 |
| 0.5 | 0.9689 |
| 0.6 | 0.9632 |
| 0.7 | 0.9573 |
| 0.8 | 0.9514 |
| 0.9 | 0.9453 |
| 1.0 | 0.9396 |

**Adjacent steps (delta=0.1): mean overlap = 0.9931**
**Most distant (delta=1.0, alpha=0.0 vs alpha=1.0): mean overlap = 0.9396**
**Gap (adjacent - distant): 0.0534**

## Verdict

**Smoothness check PARTIALLY confirms**: overlap does decline from adjacent (0.9931) to distant (0.9396) steps (gap=0.0534), and the decay is monotonic -- real movement exists, but the gap is much smaller than phase 12c/12d's own reference mechanism (gap=0.4751 there), consistent with the weaker axis-check/field-metric movement found in 04/05. This mechanism produces a real but structurally weak dial, not an inert one.

## Full 11x11 matrix (rows/cols = alpha, values = mean top-10 overlap)

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.993 | 0.986 | 0.979 | 0.973 | 0.967 | 0.963 | 0.957 | 0.951 | 0.945 | 0.940 |
| 0.1 | 0.993 | 1.000 | 0.993 | 0.986 | 0.980 | 0.974 | 0.969 | 0.963 | 0.957 | 0.951 | 0.945 |
| 0.2 | 0.986 | 0.993 | 1.000 | 0.993 | 0.986 | 0.980 | 0.975 | 0.969 | 0.964 | 0.957 | 0.952 |
| 0.3 | 0.979 | 0.986 | 0.993 | 1.000 | 0.993 | 0.986 | 0.982 | 0.976 | 0.970 | 0.963 | 0.957 |
| 0.4 | 0.973 | 0.980 | 0.986 | 0.993 | 1.000 | 0.993 | 0.988 | 0.982 | 0.976 | 0.969 | 0.963 |
| 0.5 | 0.967 | 0.974 | 0.980 | 0.986 | 0.993 | 1.000 | 0.994 | 0.988 | 0.982 | 0.975 | 0.969 |
| 0.6 | 0.963 | 0.969 | 0.975 | 0.982 | 0.988 | 0.994 | 1.000 | 0.993 | 0.987 | 0.980 | 0.974 |
| 0.7 | 0.957 | 0.963 | 0.969 | 0.976 | 0.982 | 0.988 | 0.993 | 1.000 | 0.993 | 0.986 | 0.980 |
| 0.8 | 0.951 | 0.957 | 0.964 | 0.970 | 0.976 | 0.982 | 0.987 | 0.993 | 1.000 | 0.993 | 0.986 |
| 0.9 | 0.945 | 0.951 | 0.957 | 0.963 | 0.969 | 0.975 | 0.980 | 0.986 | 0.993 | 1.000 | 0.993 |
| 1.0 | 0.940 | 0.945 | 0.952 | 0.957 | 0.963 | 0.969 | 0.974 | 0.980 | 0.986 | 0.993 | 1.000 |
