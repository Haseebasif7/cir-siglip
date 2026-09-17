# Phase 7, Step 6: Evaluation on the Untouched 1,872-Product Eval Sample

Sanity check: recomputed raw SigLIP metrics matched the existing reported baseline within tolerance (see phase7_notes.md for the exact values) -- confirms the untouched eval sample and eval methodology are unchanged from prior phases before trusting the learned/blended rows below.

| Configuration | Hit Rate@5 | Hit Rate@10 | Precision@5 | Precision@10 |
|---|---|---|---|---|
| Raw SigLIP (baseline, carried forward) | 0.502 | 0.578 | 0.191 | 0.144 |
| Model A (random negatives only) | 0.446 | 0.532 | 0.160 | 0.126 |
| Model B (random + hard negatives) | 0.346 | 0.431 | 0.118 | 0.094 |
| Blended alpha=0.7 (raw-heavy) | 0.497 | 0.572 | 0.188 | 0.144 |
| Blended alpha=0.5 (even) | 0.466 | 0.534 | 0.173 | 0.133 |
| Blended alpha=0.3 (Model-B-heavy) | 0.424 | 0.496 | 0.150 | 0.117 |

