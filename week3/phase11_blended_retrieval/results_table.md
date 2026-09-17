# Phase 11: Blended Similarity and Diversity-Forcing Retrieval -- Results

Eval sample: same untouched 1,872-product Amazon sample used since phase 1b. Cross-type-only ground truth: 1691 edges across 1732 queries (140 excluded, unknown own type at categories-breadcrumb index 3 -- identical to phase 8/9/10's view). Category diversity metric excludes the same 140 unknown-own-type queries from its average, for the same reason.

Diversity-forcing construction: 1732 queries got a genuine forced cross-type slot; 140 fell back to plain raw-SigLIP ranking because the query's own type is unknown; 0 fell back because no cross-type candidate existed anywhere in the sample for that query (neither case was silently filled with a same-type item).

Blend formula: `blended = alpha * raw_visual_similarity + (1 - alpha) * compatibility_similarity`. alpha=0.7 means mostly raw visual similarity; alpha=0.3 means mostly the compatibility signal.

## 1. Retrieval accuracy -- full also_buy ground truth (N=1,872 queries)

| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| Raw SigLIP (alone) | 0.502 | 0.578 | 0.191 | 0.144 |
| Phase 8 alone (Amazon-trained) | 0.441 | 0.523 | 0.160 | 0.120 |
| Phase 9 alone (Polyvore-trained) | 0.315 | 0.384 | 0.098 | 0.073 |
| Blend: alpha=0.3 raw + Phase 8 (0.3 raw / 0.7 compat) | 0.477 | 0.553 | 0.175 | 0.132 |
| Blend: alpha=0.5 raw + Phase 8 (0.5 raw / 0.5 compat) | 0.503 | 0.573 | 0.187 | 0.139 |
| Blend: alpha=0.7 raw + Phase 8 (0.7 raw / 0.3 compat) | 0.510 | 0.577 | 0.194 | 0.145 |
| Blend: alpha=0.3 raw + Phase 9 (0.3 raw / 0.7 compat) | 0.446 | 0.515 | 0.159 | 0.116 |
| Blend: alpha=0.5 raw + Phase 9 (0.5 raw / 0.5 compat) | 0.478 | 0.554 | 0.175 | 0.133 |
| Blend: alpha=0.7 raw + Phase 9 (0.7 raw / 0.3 compat) | 0.499 | 0.566 | 0.188 | 0.139 |
| Diversity-forcing (top-4 raw + 1 forced cross-type slot) | 0.488 | 0.574 | 0.172 | 0.140 |

## 2. Retrieval accuracy -- cross-type-only ground truth (N=1732 queries, 1691 edges)

| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| Raw SigLIP (alone) | 0.076 | 0.112 | 0.019 | 0.016 |
| Phase 8 alone (Amazon-trained) | 0.078 | 0.107 | 0.020 | 0.016 |
| Phase 9 alone (Polyvore-trained) | 0.046 | 0.074 | 0.012 | 0.011 |
| Blend: alpha=0.3 raw + Phase 8 (0.3 raw / 0.7 compat) | 0.081 | 0.118 | 0.021 | 0.018 |
| Blend: alpha=0.5 raw + Phase 8 (0.5 raw / 0.5 compat) | 0.086 | 0.122 | 0.022 | 0.018 |
| Blend: alpha=0.7 raw + Phase 8 (0.7 raw / 0.3 compat) | 0.085 | 0.122 | 0.022 | 0.018 |
| Blend: alpha=0.3 raw + Phase 9 (0.3 raw / 0.7 compat) | 0.069 | 0.102 | 0.017 | 0.015 |
| Blend: alpha=0.5 raw + Phase 9 (0.5 raw / 0.5 compat) | 0.077 | 0.109 | 0.020 | 0.016 |
| Blend: alpha=0.7 raw + Phase 9 (0.7 raw / 0.3 compat) | 0.080 | 0.109 | 0.020 | 0.016 |
| Diversity-forcing (top-4 raw + 1 forced cross-type slot) | 0.087 | 0.116 | 0.022 | 0.017 |

## 3. Category diversity -- mean fraction of top-K that is a different fine-grained type than the query (N=1732 queries with known own type)

| Configuration | Diversity@5 | Diversity@10 |
|---|---|---|
| Raw SigLIP (alone) | 0.330 | 0.373 |
| Phase 8 alone (Amazon-trained) | 0.415 | 0.453 |
| Phase 9 alone (Polyvore-trained) | 0.579 | 0.626 |
| Blend: alpha=0.3 raw + Phase 8 (0.3 raw / 0.7 compat) | 0.382 | 0.423 |
| Blend: alpha=0.5 raw + Phase 8 (0.5 raw / 0.5 compat) | 0.362 | 0.400 |
| Blend: alpha=0.7 raw + Phase 8 (0.7 raw / 0.3 compat) | 0.338 | 0.381 |
| Blend: alpha=0.3 raw + Phase 9 (0.3 raw / 0.7 compat) | 0.403 | 0.451 |
| Blend: alpha=0.5 raw + Phase 9 (0.5 raw / 0.5 compat) | 0.361 | 0.404 |
| Blend: alpha=0.7 raw + Phase 9 (0.7 raw / 0.3 compat) | 0.335 | 0.382 |
| Diversity-forcing (top-4 raw + 1 forced cross-type slot) | 0.454 | 0.410 |

## 4. Popularity behavior (ARP = mean catalog-wide reference count; coverage = fraction of the 1,872-product sample recommended at least once)

| Configuration | ARP@5 | ARP@10 | Coverage@5 | Coverage@10 |
|---|---|---|---|---|
| Raw SigLIP (alone) | 42.78 | 41.87 | 0.918 | 0.968 |
| Phase 8 alone (Amazon-trained) | 41.41 | 40.79 | 0.947 | 0.990 |
| Phase 9 alone (Polyvore-trained) | 39.52 | 39.44 | 0.950 | 0.991 |
| Blend: alpha=0.3 raw + Phase 8 (0.3 raw / 0.7 compat) | 42.07 | 41.16 | 0.935 | 0.984 |
| Blend: alpha=0.5 raw + Phase 8 (0.5 raw / 0.5 compat) | 42.09 | 41.79 | 0.935 | 0.975 |
| Blend: alpha=0.7 raw + Phase 8 (0.7 raw / 0.3 compat) | 43.04 | 41.84 | 0.924 | 0.971 |
| Blend: alpha=0.3 raw + Phase 9 (0.3 raw / 0.7 compat) | 39.44 | 39.33 | 0.931 | 0.978 |
| Blend: alpha=0.5 raw + Phase 9 (0.5 raw / 0.5 compat) | 41.00 | 40.34 | 0.926 | 0.976 |
| Blend: alpha=0.7 raw + Phase 9 (0.7 raw / 0.3 compat) | 42.17 | 41.00 | 0.924 | 0.975 |
| Diversity-forcing (top-4 raw + 1 forced cross-type slot) | 41.49 | 41.35 | 0.920 | 0.970 |

