# Phase 4 Results: DeepFashion In-shop Retrieval (Query vs. Gallery, Official Split)

Recall@K = fraction of queries where at least one of the top-K gallery images shares the query's item_id. Computed strictly query-vs-gallery (never query-vs-query, never involving train images) per the dataset's official protocol.

| Technique | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|
| ResNet50 | 0.290 | 0.459 | 0.531 | 0.602 |
| CLIP ViT-B/32 | 0.455 | 0.673 | 0.743 | 0.807 |
| FashionCLIP | 0.666 | 0.851 | 0.897 | 0.930 |
| SigLIP | 0.737 | 0.894 | 0.929 | 0.953 |
