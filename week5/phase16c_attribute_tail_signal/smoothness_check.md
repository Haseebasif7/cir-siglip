# Phase 16c, Step 4: Adjacent vs Distant Alpha Overlap

Same method as phase 16's `06_smoothness_check.py` (phase 12d's method), full pairwise top-10 overlap matrix across all 11 alpha values, all 1872 phase 1b queries.

## Mean overlap as a function of |delta alpha|

| |delta alpha| | Mean top-10 overlap |
|---|---|
| 0.1 | 0.9823 |
| 0.2 | 0.9660 |
| 0.3 | 0.9509 |
| 0.4 | 0.9362 |
| 0.5 | 0.9220 |
| 0.6 | 0.9083 |
| 0.7 | 0.8946 |
| 0.8 | 0.8811 |
| 0.9 | 0.8680 |
| 1.0 | 0.8562 |

**Adjacent steps (delta=0.1): mean overlap = 0.9823**
**Most distant (delta=1.0): mean overlap = 0.8562**
**Gap (adjacent - distant): 0.1261**

## Direct comparison to phase 16 and this phase's own success bar

- Phase 16's gap: 0.0534
- Phase 16c's gap: 0.1261 (2.36x phase 16's)
- Success bar (this phase's own, set in advance): at least 2x phase 16's gap = 0.1068
- **CLEARS the bar**

## Verdict

**Smoothness bar CLEARED**: gap (0.1261) is at least double phase 16's (0.0534), and decay is monotonic -- a genuinely stronger, more usable dial than phase 16 produced.

## Full 11x11 matrix (rows/cols = alpha, values = mean top-10 overlap)

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.983 | 0.966 | 0.951 | 0.938 | 0.924 | 0.910 | 0.896 | 0.883 | 0.868 | 0.856 |
| 0.1 | 0.983 | 1.000 | 0.981 | 0.965 | 0.951 | 0.937 | 0.922 | 0.908 | 0.895 | 0.880 | 0.868 |
| 0.2 | 0.966 | 0.981 | 1.000 | 0.982 | 0.967 | 0.952 | 0.938 | 0.923 | 0.909 | 0.893 | 0.881 |
| 0.3 | 0.951 | 0.965 | 0.982 | 1.000 | 0.984 | 0.968 | 0.953 | 0.937 | 0.923 | 0.907 | 0.894 |
| 0.4 | 0.938 | 0.951 | 0.967 | 0.984 | 1.000 | 0.983 | 0.967 | 0.951 | 0.936 | 0.920 | 0.908 |
| 0.5 | 0.924 | 0.937 | 0.952 | 0.968 | 0.983 | 1.000 | 0.983 | 0.965 | 0.950 | 0.933 | 0.920 |
| 0.6 | 0.910 | 0.922 | 0.938 | 0.953 | 0.967 | 0.983 | 1.000 | 0.981 | 0.965 | 0.948 | 0.934 |
| 0.7 | 0.896 | 0.908 | 0.923 | 0.937 | 0.951 | 0.965 | 0.981 | 1.000 | 0.982 | 0.964 | 0.950 |
| 0.8 | 0.883 | 0.895 | 0.909 | 0.923 | 0.936 | 0.950 | 0.965 | 0.982 | 1.000 | 0.980 | 0.966 |
| 0.9 | 0.868 | 0.880 | 0.893 | 0.907 | 0.920 | 0.933 | 0.948 | 0.964 | 0.980 | 1.000 | 0.983 |
| 1.0 | 0.856 | 0.868 | 0.881 | 0.894 | 0.908 | 0.920 | 0.934 | 0.950 | 0.966 | 0.983 | 1.000 |
