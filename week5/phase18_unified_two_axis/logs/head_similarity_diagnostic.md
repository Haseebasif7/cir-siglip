# Phase 18: Head-Similarity Diagnostic (Bonus, Mechanistic)

Mean per-item cosine similarity between each pair of the 4 heads' own normalized outputs, same 5,000-item
sample (phase 7 pool), same method phases 16d/17 used for their 2-head checks.

| Pair | Mean cosine |
|---|---|
| sub_rel vs sub_tail | -0.0167 |
| sub_rel vs comp_rel | -0.0026 |
| sub_rel vs comp_tail | -0.0084 |
| sub_tail vs comp_rel | 0.0235 |
| sub_tail vs comp_tail | -0.0050 |
| comp_rel vs comp_tail | 0.0108 |

All 6 pairs are near-orthogonal (|cosine| <= 0.0235), consistent with phase 16d/17's dedicated-capacity
finding that independent heads learn independent, near-uncorrelated transformations even without an
explicit uniformity/repulsion term. This holds across all 4 heads simultaneously here, not just pairwise
within a single 2-head mechanism.

**This representational independence does NOT by itself guarantee retrieval-level axis independence** --
step 7's axis independence check found a real cross-axis interaction (axis 2's direction reverses at high
alpha1) despite every head pair being near-orthogonal in representation space. The interaction likely comes
from the bilinear BLENDING dynamics themselves (mixing normalized vectors from up to 4 heads, then
renormalizing, is not a purely additive operation on retrieval rankings even when the underlying heads are
orthogonal), not from the heads sharing representational structure.
