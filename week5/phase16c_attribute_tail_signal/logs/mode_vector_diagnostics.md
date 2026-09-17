# Phase 16c: Mode Vector Mechanistic Diagnostics

Same check phase 16 used, direct comparison:

| | Phase 16 | Phase 16c |
|---|---|---|
| mode_relevance norm | 0.1914 | 0.2077 |
| mode_tail norm | 0.1660 | 0.2268 |
| cosine(mode_relevance, mode_tail) | 0.5525 | 0.0590 |
| base projection mean norm | 1.2334 | 1.0725 (std 0.2623) |

Mode vector magnitude relative to base projection: mode_relevance is 19.4% of the base norm, mode_tail is 21.1% (phase 16: 15.5% / 13.5%).

**The two mode vectors point in more DIVERGENT directions than phase 16's did** (cosine 0.0590 vs phase 16's 0.5525) -- direct mechanistic evidence that training on genuinely different edge populations pushed the modes further apart than reweighting the same population did.
