# Phase 12c, Step 4.2: Control-Effectiveness Diagnostic, Three-Way Comparison

Same procedure as phases 12/12b (sample size 1000, seed=42, top-10), on the ranking-distillation checkpoint.

## Axis 1: visual similarity to the query (raw SigLIP cosine, top-10 average)

- Substitute mode: phase 12 = 0.7193, phase 12b = 0.7233, **phase 12c = 0.7498**
- Complement mode: phase 12 = 0.7120, phase 12b = 0.7228, **phase 12c = 0.7217**
- Gap (substitute - complement): phase 12 = 0.0073, phase 12b = 0.0005, **phase 12c = 0.0281**

## Axis 2: match rate against real outfit co-occurrence (hit@10 on the true target)

- Substitute mode: phase 12 = 0.1280, phase 12b = 0.1250, **phase 12c = 0.0760**
- Complement mode: phase 12 = 0.1280, phase 12b = 0.1180, **phase 12c = 0.1020**

## Overlap and per-item cosine between the two modes

- Top-10 overlap between modes (n=500): phase 12 = 0.7636, phase 12b = 0.5102, **phase 12c = 0.4022**
- Per-item cosine(z_sub, z_comp), 5,000 items: phase 12 = 0.8288, phase 12b = 0.5844, **phase 12c = 0.3000**

## THE DECISIVE CHECK: does substitute mode's retrieval now resemble raw SigLIP's more than complement mode's does?

- Substitute mode vs raw SigLIP top-10 overlap: phase 12 = --, phase 12b = 0.1612, **phase 12c = 0.3428**
- Complement mode vs raw SigLIP top-10 overlap: phase 12 = --, phase 12b = 0.1754, **phase 12c = 0.1708**
- **Gap (substitute vs raw) - (complement vs raw) = 0.1720** (phase 12b's gap was -0.0142, in the WRONG direction -- substitute was slightly LOWER).
- Stopping-condition threshold (stated in advance, see script docstring): a gap below 0.03 counts as noise-level, not a real separation.
- **GAP CLEARS THE THRESHOLD**: gap = 0.1720 >= 0.03.

