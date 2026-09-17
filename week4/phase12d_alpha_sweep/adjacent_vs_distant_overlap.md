# Phase 12d, Step 3: Adjacent vs Distant Alpha Overlap

Full pairwise top-10 overlap matrix across all 11 alpha values, 500-query sample (same 500-query overlap sample as step 1, seed=42), computed from step 1's cached per-alpha retrieval lists.

## Mean overlap as a function of |delta alpha| (the real test)

| |delta alpha| | Mean top-10 overlap |
|---|---|
| 0.1 | 0.8773 |
| 0.2 | 0.7694 |
| 0.3 | 0.6694 |
| 0.4 | 0.5821 |
| 0.5 | 0.5087 |
| 0.6 | 0.4528 |
| 0.7 | 0.4154 |
| 0.8 | 0.3969 |
| 0.9 | 0.3937 |
| 1.0 | 0.4022 |

**Adjacent steps (delta=0.1): mean overlap = 0.8773**
**Most distant (delta=1.0, alpha=0.0 vs alpha=1.0): mean overlap = 0.4022**

## Verdict

**Real, smooth relationship confirmed**: overlap decays consistently as |delta alpha| grows (adjacent=0.8773 down to distant=0.4022, a gap of 0.4751), not just a binary near-vs-far difference. This is the operational definition of a usable dial -- small turns change results a little, large turns change results a lot.

## Full 11x11 matrix (rows/cols = alpha, values = mean top-10 overlap)

| alpha | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 0.869 | 0.774 | 0.695 | 0.637 | 0.575 | 0.534 | 0.489 | 0.457 | 0.426 | 0.402 |
| 0.1 | 0.869 | 1.000 | 0.884 | 0.788 | 0.699 | 0.607 | 0.538 | 0.472 | 0.429 | 0.390 | 0.361 |
| 0.2 | 0.774 | 0.884 | 1.000 | 0.880 | 0.763 | 0.645 | 0.556 | 0.477 | 0.423 | 0.377 | 0.344 |
| 0.3 | 0.695 | 0.788 | 0.880 | 1.000 | 0.859 | 0.724 | 0.615 | 0.520 | 0.459 | 0.405 | 0.367 |
| 0.4 | 0.637 | 0.699 | 0.763 | 0.859 | 1.000 | 0.844 | 0.718 | 0.610 | 0.538 | 0.476 | 0.431 |
| 0.5 | 0.575 | 0.607 | 0.645 | 0.724 | 0.844 | 1.000 | 0.854 | 0.734 | 0.651 | 0.582 | 0.529 |
| 0.6 | 0.534 | 0.538 | 0.556 | 0.615 | 0.718 | 0.854 | 1.000 | 0.863 | 0.770 | 0.693 | 0.635 |
| 0.7 | 0.489 | 0.472 | 0.477 | 0.520 | 0.610 | 0.734 | 0.863 | 1.000 | 0.892 | 0.812 | 0.746 |
| 0.8 | 0.457 | 0.429 | 0.423 | 0.459 | 0.538 | 0.651 | 0.770 | 0.892 | 1.000 | 0.906 | 0.842 |
| 0.9 | 0.426 | 0.390 | 0.377 | 0.405 | 0.476 | 0.582 | 0.693 | 0.812 | 0.906 | 1.000 | 0.923 |
| 1.0 | 0.402 | 0.361 | 0.344 | 0.367 | 0.431 | 0.529 | 0.635 | 0.746 | 0.842 | 0.923 | 1.000 |

