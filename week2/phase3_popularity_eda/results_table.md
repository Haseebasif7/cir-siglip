# Phase 3 Results: ARP and Catalog Coverage by Technique

**Average Recommendation Popularity (ARP)**: mean catalog-wide reference count (from `popularity_lookup.csv`) of items appearing in recommendations, averaged across all 1,872 queries. Higher = systematically recommending more popular items.

**Catalog coverage**: fraction of the 1,872-product sample that appears at least once, anywhere, across every query's top-K recommendations pooled together. Low coverage = a small subset of products dominates recommendations regardless of query -- the direct signature of the overspecialization/repetition problem this phase was asked to check for.

| Technique | ARP@5 | ARP@10 | Unique products recommended @5 | Coverage@5 | Unique products recommended @10 | Coverage@10 |
|---|---|---|---|---|---|---|
| ResNet50 | 42.62 | 41.87 | 1757 | 0.939 | 1832 | 0.979 |
| CLIP ViT-B/32 | 41.49 | 42.48 | 1665 | 0.889 | 1764 | 0.942 |
| FashionCLIP | 38.50 | 38.51 | 1727 | 0.923 | 1823 | 0.974 |
| SigLIP | 42.78 | 41.87 | 1718 | 0.918 | 1813 | 0.968 |
