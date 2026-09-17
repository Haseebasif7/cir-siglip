# Phase 18b: Smoke Test Report

5000 REL / 3000 CT / 5000 ST edges, 3 epochs, uncalibrated equal weights (1.0 each) -- checks for collapse in any of the 4 heads, especially the newly-reformulated sub_tail (now MNRL with head-biased negatives instead of ranking distillation), before committing to full training with fresh calibration.

## Final-epoch mean pairwise cosine per head (256-item sample)

| Corner | Mean pairwise cosine |
|---|---|
| sub_rel | 0.5991 |
| sub_tail | 0.0182 |
| comp_rel | 0.1641 |
| comp_tail | 0.0657 |

**Collapse warning (>0.98 in any head): NO**

