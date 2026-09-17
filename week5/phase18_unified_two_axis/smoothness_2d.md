# Phase 18, Step 8: Smoothness Extended to Two Dimensions

Same adjacent-vs-distant top-10 overlap logic as phases 12d/16/16c/16d/17, applied along each axis (other alpha held fixed, averaged across all 5 values of it) and diagonally across the grid.

## Axis 1 (substitute <-> complement), alpha2 held fixed

- Adjacent (|delta alpha1|=0.25): mean overlap = 0.7519
- Distant (|delta alpha1|=1.0, full range): mean overlap = 0.4043
- **Gap = 0.3476**

## Axis 2 (relevance <-> tail-exposure), alpha1 held fixed

- Adjacent (|delta alpha2|=0.25): mean overlap = 0.7684
- Distant (|delta alpha2|=1.0, full range): mean overlap = 0.4859
- **Gap = 0.2825**

## Diagonal (both alpha1 and alpha2 move together)

- Adjacent (one 0.25 step along either diagonal): mean overlap = 0.6804
- Distant (full corner-to-corner, either diagonal): mean overlap = 0.3909
- **Gap = 0.2895**

## Comparison to this project's prior 1D smoothness gaps

| Mechanism | Gap |
|---|---|
| Phase 12d (substitute/complement, shared trunk) | 0.4751 |
| Phase 16d (relevance/tail, dedicated capacity) | 0.6110 |
| Phase 17 (substitute/complement, dedicated capacity, Polyvore) | 0.6646 |
| **Phase 18, axis 1 (substitute/complement, 4-head, Amazon)** | **0.3476** |
| **Phase 18, axis 2 (relevance/tail, 4-head, Amazon)** | **0.2825** |
| **Phase 18, diagonal (both axes together)** | **0.2895** |

## Verdict

Gaps are computed from a coarser 5-point grid per axis (vs. the usual 11-point 1D sweep), so these numbers are directly comparable in spirit but not on an identical point density -- flagged here rather than presented as an apples-to-apples match.

**Both individual axes, and the diagonal combining them, show real, substantial smoothness gaps -- this is a genuinely, smoothly navigable 2D control space, not just two working 1D dials bolted together with a sharp seam between them.**
