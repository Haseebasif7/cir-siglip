# Phase 1b: Report-Ready Summary Table

One row per category: sample size, ground-truth density (overlap%),
and Hit Rate@5 for all four techniques side by side, plus which
technique won. Meant to be dropped directly into the technical report
-- results_table.md and category_breakdown.md have the same numbers
split across files; this puts them in one place.

| Category | N queries | Overlap refs | Overlap % | ResNet50 HR@5 | CLIP ViT-B/32 HR@5 | FashionCLIP HR@5 | SigLIP HR@5 | Best technique |
|---|---|---|---|---|---|---|---|---|
| Girls | 175 | 746 | 7.9% | 0.423 | 0.434 | 0.491 | 0.509 | SigLIP |
| Luggage & Travel Gear | 175 | 992 | 9.3% | 0.417 | 0.411 | 0.474 | 0.560 | SigLIP |
| Shoe, Jewelry & Watch Accessories | 175 | 1214 | 12.0% | 0.451 | 0.469 | 0.537 | 0.600 | SigLIP |
| Baby | 174 | 638 | 5.1% | 0.345 | 0.339 | 0.402 | 0.420 | SigLIP |
| Traditional & Cultural Wear | 174 | 228 | 3.5% | 0.172 | 0.195 | 0.264 | 0.276 | SigLIP |
| Costumes & Accessories | 174 | 805 | 6.3% | 0.236 | 0.299 | 0.379 | 0.437 | SigLIP |
| Boys | 174 | 556 | 6.1% | 0.408 | 0.477 | 0.534 | 0.529 | FashionCLIP |
| Men | 173 | 2639 | 19.2% | 0.688 | 0.647 | 0.717 | 0.723 | SigLIP |
| Novelty & More | 171 | 1052 | 11.3% | 0.503 | 0.608 | 0.632 | 0.684 | SigLIP |
| Uniforms, Work & Safety | 165 | 145 | 2.0% | 0.170 | 0.139 | 0.200 | 0.212 | SigLIP |
| Women | 142 | 435 | 3.8% | 0.310 | 0.437 | 0.521 | 0.570 | SigLIP |
