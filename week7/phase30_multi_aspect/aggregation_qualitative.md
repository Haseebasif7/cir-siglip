# Phase 30, Step 6: Aggregation Weight and Per-Aspect Contribution Inspection

Sample of 8 validation queries (context length >= 2, seed=7), single-seed gate checkpoint (`models/multiaspect_seed42.pt`). For each query: the learned aggregation weights (computed from the query's own mean-pooled-per-aspect context, never from the candidate) and the per-aspect similarity contribution to the true target vs. 3 random same-category negatives.

## Query 173126574 (category=all-body, context length=5)

Aggregation weights: visual=0.338, form=0.153, semantic=0.194, general=0.314  (entropy raw=1.336, normalized=0.964)

| Candidate | sim_visual | sim_form | sim_semantic | sim_general | weighted score |
|---|---|---|---|---|---|
| TRUE TARGET | 0.721 | 0.824 | 0.561 | 0.846 | 0.745 |
| random negative 1 | -0.163 | 0.590 | 0.416 | 0.728 | 0.345 |
| random negative 2 | 0.061 | -0.340 | -0.443 | 0.065 | -0.097 |
| random negative 3 | -0.359 | -0.322 | -0.098 | 0.363 | -0.076 |

## Query 212856802 (category=bottoms, context length=6)

Aggregation weights: visual=0.294, form=0.327, semantic=0.217, general=0.161  (entropy raw=1.351, normalized=0.975)

| Candidate | sim_visual | sim_form | sim_semantic | sim_general | weighted score |
|---|---|---|---|---|---|
| TRUE TARGET | -0.065 | 0.346 | 0.340 | 0.817 | 0.300 |
| random negative 1 | 0.491 | 0.633 | 0.475 | 0.818 | 0.587 |
| random negative 2 | 0.116 | 0.110 | 0.057 | 0.684 | 0.193 |
| random negative 3 | -0.086 | 0.390 | 0.154 | 0.744 | 0.256 |

## Query 195488623 (category=bottoms, context length=3)

Aggregation weights: visual=0.308, form=0.329, semantic=0.270, general=0.093  (entropy raw=1.302, normalized=0.940)

| Candidate | sim_visual | sim_form | sim_semantic | sim_general | weighted score |
|---|---|---|---|---|---|
| TRUE TARGET | -0.171 | 0.508 | 0.761 | 0.704 | 0.385 |
| random negative 1 | 0.030 | 0.603 | 0.287 | 0.866 | 0.365 |
| random negative 2 | -0.224 | -0.082 | 0.020 | 0.573 | -0.038 |
| random negative 3 | -0.134 | -0.075 | -0.122 | 0.439 | -0.058 |

## Query 197695224 (category=jewellery, context length=4)

Aggregation weights: visual=0.278, form=0.244, semantic=0.245, general=0.233  (entropy raw=1.384, normalized=0.998)

| Candidate | sim_visual | sim_form | sim_semantic | sim_general | weighted score |
|---|---|---|---|---|---|
| TRUE TARGET | 0.376 | 0.425 | 0.397 | 0.832 | 0.499 |
| random negative 1 | -0.160 | -0.121 | -0.123 | 0.560 | 0.026 |
| random negative 2 | -0.453 | -0.405 | 0.210 | 0.574 | -0.040 |
| random negative 3 | -0.038 | 0.271 | 0.087 | 0.634 | 0.224 |

## Query 63437805 (category=sunglasses, context length=3)

Aggregation weights: visual=0.334, form=0.155, semantic=0.118, general=0.393  (entropy raw=1.275, normalized=0.919)

| Candidate | sim_visual | sim_form | sim_semantic | sim_general | weighted score |
|---|---|---|---|---|---|
| TRUE TARGET | 0.591 | 0.806 | 0.821 | 0.824 | 0.743 |
| random negative 1 | -0.274 | -0.324 | -0.465 | -0.054 | -0.218 |
| random negative 2 | -0.206 | -0.095 | -0.337 | 0.212 | -0.040 |
| random negative 3 | -0.449 | -0.141 | -0.413 | 0.106 | -0.179 |

## Query 119170154 (category=bags, context length=3)

Aggregation weights: visual=0.364, form=0.412, semantic=0.183, general=0.041  (entropy raw=1.175, normalized=0.847)

| Candidate | sim_visual | sim_form | sim_semantic | sim_general | weighted score |
|---|---|---|---|---|---|
| TRUE TARGET | 0.773 | 0.692 | 0.680 | 0.810 | 0.724 |
| random negative 1 | -0.178 | -0.238 | -0.063 | 0.195 | -0.166 |
| random negative 2 | -0.417 | -0.057 | -0.093 | 0.434 | -0.175 |
| random negative 3 | 0.041 | 0.175 | 0.087 | 0.631 | 0.129 |

## Query 117663124 (category=bags, context length=4)

Aggregation weights: visual=0.230, form=0.113, semantic=0.091, general=0.567  (entropy raw=1.124, normalized=0.811)

| Candidate | sim_visual | sim_form | sim_semantic | sim_general | weighted score |
|---|---|---|---|---|---|
| TRUE TARGET | 0.567 | 0.677 | 0.719 | 0.743 | 0.693 |
| random negative 1 | -0.135 | 0.283 | 0.118 | 0.486 | 0.287 |
| random negative 2 | 0.472 | 0.239 | 0.518 | 0.646 | 0.549 |
| random negative 3 | 0.121 | 0.217 | 0.023 | 0.540 | 0.360 |

## Query 46867568 (category=outerwear, context length=7)

Aggregation weights: visual=0.485, form=0.270, semantic=0.197, general=0.047  (entropy raw=1.169, normalized=0.843)

| Candidate | sim_visual | sim_form | sim_semantic | sim_general | weighted score |
|---|---|---|---|---|---|
| TRUE TARGET | 0.687 | 0.572 | 0.762 | 0.835 | 0.678 |
| random negative 1 | -0.347 | 0.017 | 0.235 | 0.556 | -0.091 |
| random negative 2 | 0.041 | 0.319 | 0.479 | 0.803 | 0.239 |
| random negative 3 | -0.325 | -0.126 | -0.137 | 0.591 | -0.191 |

## Summary across the sample

| Aspect | Mean weight | Std across queries |
|---|---|---|
| visual | 0.329 | 0.071 |
| form | 0.250 | 0.098 |
| semantic | 0.190 | 0.056 |
| general | 0.231 | 0.173 |

Mean normalized entropy across the sample: 0.912 (1.0 = fully uniform/aspect-bias-disengaged, 0.0 = one-hot collapse onto a single aspect).

## Aspect head output similarity check (2000-item sample)

Mean cosine similarity between each pair of aspect heads' outputs on the same items -- near 1.0 would mean two heads collapsed onto the same function.

| | visual | form | semantic | general |
|---|---|---|---|---|
| visual | 1.000 | 0.017 | 0.040 | -0.055 |
| form | 0.017 | 1.000 | 0.021 | 0.019 |
| semantic | 0.040 | 0.021 | 1.000 | -0.025 |
| general | -0.055 | 0.019 | -0.025 | 1.000 |

