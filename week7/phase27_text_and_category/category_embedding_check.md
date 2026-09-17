# Category Embedding Collapse Check

Requested addition: after training the `text_category_learned` variant, verify the learned 11-category embedding table actually contains meaningful, differentiated vectors rather than having collapsed toward a single direction. A collapsed table would mean category conditioning never actually engaged the network, which would make any "category doesn't help" finding an architectural failure, not a real result about whether category conditioning helps retrieval -- these must not be conflated.

**Verdict: NOT collapsed** (mean off-diagonal pairwise cosine = 0.0815, threshold = 0.95).

## Pairwise cosine similarity matrix

| | accessories | all-body | bags | bottoms | hats | jewellery | outerwear | scarves | shoes | sunglasses | tops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **accessories** | 1.000 | 0.075 | 0.027 | -0.053 | 0.159 | 0.104 | 0.007 | 0.215 | -0.026 | 0.130 | 0.031 |
| **all-body** | 0.075 | 1.000 | -0.119 | 0.127 | -0.031 | 0.191 | -0.022 | 0.050 | 0.045 | 0.047 | 0.122 |
| **bags** | 0.027 | -0.119 | 1.000 | 0.045 | 0.142 | 0.145 | 0.111 | 0.050 | 0.246 | 0.186 | 0.138 |
| **bottoms** | -0.053 | 0.127 | 0.045 | 1.000 | 0.180 | -0.057 | 0.105 | 0.116 | 0.201 | 0.014 | 0.177 |
| **hats** | 0.159 | -0.031 | 0.142 | 0.180 | 1.000 | -0.189 | 0.159 | 0.163 | 0.150 | 0.267 | -0.037 |
| **jewellery** | 0.104 | 0.191 | 0.145 | -0.057 | -0.189 | 1.000 | -0.015 | 0.085 | 0.163 | -0.014 | 0.256 |
| **outerwear** | 0.007 | -0.022 | 0.111 | 0.105 | 0.159 | -0.015 | 1.000 | 0.198 | 0.100 | 0.249 | -0.073 |
| **scarves** | 0.215 | 0.050 | 0.050 | 0.116 | 0.163 | 0.085 | 0.198 | 1.000 | 0.198 | 0.180 | -0.058 |
| **shoes** | -0.026 | 0.045 | 0.246 | 0.201 | 0.150 | 0.163 | 0.100 | 0.198 | 1.000 | 0.195 | -0.194 |
| **sunglasses** | 0.130 | 0.047 | 0.186 | 0.014 | 0.267 | -0.014 | 0.249 | 0.180 | 0.195 | 1.000 | -0.183 |
| **tops** | 0.031 | 0.122 | 0.138 | 0.177 | -0.037 | 0.256 | -0.073 | -0.058 | -0.194 | -0.183 | 1.000 |

Mean off-diagonal cosine: **0.0815**  (range [-0.1936, 0.2668])
Most-confusable pair (highest cosine): **hats** / **sunglasses** (cosine=0.2668)
Most-differentiated pair (lowest cosine): **shoes** / **tops** (cosine=-0.1936)

