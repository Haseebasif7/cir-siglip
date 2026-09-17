# Phase 18: Smoke Test Report

5000 relevance-axis / 3000 tail-axis edges, 3 epochs, uncalibrated equal weights (1.0 each) -- this test only checks for embedding collapse in any of the 4 heads before committing to full training with fresh calibration.

## Final-epoch mean pairwise cosine per head (256-item sample)

| Corner | Mean pairwise cosine |
|---|---|
| sub_rel | 0.6423 |
| sub_tail | 0.7044 |
| comp_rel | 0.1311 |
| comp_tail | 0.0433 |

**Collapse warning (>0.98 in any head): NO**

