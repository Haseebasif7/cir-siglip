# Phase 17: CIR Benchmark Results, Four-Way Comparison

Same benchmark as phases 12/12b/12c (`week4/phase12_controllable_modes/data/cir_benchmark.json`, unchanged): 26494 pool slots, 29681 queries.

| Configuration | Recall@10 | Recall@30 | Recall@50 | N queries |
|---|---|---|---|---|
| Raw SigLIP (alone) [reference] | 0.0553 | 0.1067 | 0.1437 | 29681 |
| Phase 9 Model A (alone) [reference] | 0.1317 | 0.2464 | 0.3216 | 29681 |
| | | | | |
| Phase 12 (shared trunk, batch-local pairwise): Substitute mode | 0.1329 | 0.2467 | 0.3215 | 29681 |
| Phase 12b (shared trunk, PCA-128 target): Substitute mode | 0.1212 | 0.2254 | 0.2957 | 29681 |
| Phase 12c (shared trunk, ranking distillation): Substitute mode | 0.0667 | 0.1307 | 0.1734 | 29681 |
| **Phase 17 (dedicated capacity, ranking distillation): Substitute mode** | 0.0573 | 0.1157 | 0.1572 | 29681 |
| | | | | |
| Phase 12 (shared trunk, batch-local pairwise): Complement mode | 0.1335 | 0.2507 | 0.3250 | 29681 |
| Phase 12b (shared trunk, PCA-128 target): Complement mode | 0.1200 | 0.2262 | 0.2963 | 29681 |
| Phase 12c (shared trunk, ranking distillation): Complement mode | 0.0971 | 0.1875 | 0.2471 | 29681 |
| **Phase 17 (dedicated capacity, ranking distillation): Complement mode** | 0.1202 | 0.2283 | 0.3000 | 29681 |
| | | | | |
| Phase 12 (shared trunk, batch-local pairwise): Blend (0.5) | 0.1366 | 0.2524 | 0.3289 | 29681 |
| Phase 12b (shared trunk, PCA-128 target): Blend (0.5) | 0.1207 | 0.2261 | 0.2957 | 29681 |
| Phase 12c (shared trunk, ranking distillation): Blend (0.5) | 0.0893 | 0.1695 | 0.2255 | 29681 |
| **Phase 17 (dedicated capacity, ranking distillation): Blend (0.5)** | 0.0937 | 0.1833 | 0.2469 | 29681 |
| | | | | |
| OutfitTransformer (literature anchor, not independently reproduced) | 0.0958 | 0.1796 | 0.2198 | -- |

**Reading this table**: as established since phase 12, Recall@K alone is not sensitive enough to tell whether the two modes are behaviorally distinct on its own -- see the alpha-sweep table below (axis checks, mode overlap, overlap with raw SigLIP) for the decisive comparison, and `smoothness_check.md` for whether this is a genuine, continuously usable dial.


---

## Step 5.1: Official Polyvore Test-Split Evaluation (Compatibility AUC + FITB)

Same protocol as `week3/phase9_polyvore_compatibility/scripts/06_evaluate_official.py`, reused directly. **Scoping note**: phase 12c never ran this benchmark (phases 12/12b/12c/12d all used this project's own CIR/Recall@K harness exclusively) -- so there is no pre-existing phase 12c AUC/FITB number to cite. The phase 12c row below is computed FRESH, in this same script, from phase 12c's own saved checkpoint (`week4/phase12c_ranking_distillation/models/ranking_distillation.pt`), making this a genuine same-protocol, same-day comparison rather than a mismatched citation.

Literature anchor (Vasileva et al. ECCV'18, full type-aware trained network, NOT directly comparable to either frozen-SigLIP+small-head setup here): AUC=0.88, FITB accuracy=0.576.

| Configuration | Compatibility AUC | FITB Accuracy |
|---|---|---|
| Raw SigLIP (reference) | 0.7172 | 0.4843 |
| Phase 9 Model A (single-mode projection, reference) | 0.9469 | 0.7031 |
| | | |
| Phase 12c (shared trunk): Substitute mode | 0.7607 | 0.5333 |
| Phase 12c (shared trunk): Complement mode | 0.9061 | 0.6319 |
| Phase 12c (shared trunk): Blend (0.5) | 0.8493 | 0.6040 |
| Phase 17 (dedicated capacity): Substitute mode | 0.7281 | 0.5086 |
| Phase 17 (dedicated capacity): Complement mode | 0.9402 | 0.6917 |
| Phase 17 (dedicated capacity): Blend (0.5) | 0.8951 | 0.6402 |

(Compatibility test: n=20000 lines scored; FITB test: n=10000 questions scored, same test-split sizes as phase 9's own run.)


---

## Step 5.2: Alpha Interpolation Sweep

This phase's `dedicated_capacity_substitute_complement.pt` checkpoint, evaluated at 11 alpha values (0.0 to 1.0, step 0.1) -- no retraining. Full CIR benchmark: all 26494 pool slots, 29681 queries. Diagnostic metrics a/b on the same 1000-query sample (seed=42) used throughout phases 12/12b/12c/12d; metrics c/d on the same 500-query overlap sample (seed=42).

| alpha | Recall@10 | Recall@30 | Recall@50 | Visual sim (a) | Co-occur hit rate (b) | Overlap w/ raw SigLIP (c) | Overlap w/ complement (d) |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.1202 | 0.2283 | 0.3000 | 0.7142 | 0.1260 | 0.1132 | 1.0000 |
| 0.1 | 0.1199 | 0.2268 | 0.2994 | 0.7149 | 0.1260 | 0.1160 | 0.9016 |
| 0.2 | 0.1178 | 0.2244 | 0.2955 | 0.7171 | 0.1240 | 0.1222 | 0.7958 |
| 0.3 | 0.1130 | 0.2180 | 0.2890 | 0.7213 | 0.1200 | 0.1324 | 0.6742 |
| 0.4 | 0.1055 | 0.2049 | 0.2730 | 0.7267 | 0.1230 | 0.1536 | 0.5346 |
| 0.5 | 0.0937 | 0.1833 | 0.2469 | 0.7328 | 0.1160 | 0.1844 | 0.3992 |
| 0.6 | 0.0803 | 0.1599 | 0.2146 | 0.7385 | 0.0970 | 0.2178 | 0.2876 |
| 0.7 | 0.0690 | 0.1381 | 0.1881 | 0.7431 | 0.0840 | 0.2594 | 0.2148 |
| 0.8 | 0.0614 | 0.1233 | 0.1664 | 0.7462 | 0.0740 | 0.3002 | 0.1810 |
| 0.9 | 0.0577 | 0.1167 | 0.1574 | 0.7483 | 0.0700 | 0.3236 | 0.1616 |
| 1.0 | 0.0573 | 0.1157 | 0.1572 | 0.7492 | 0.0710 | 0.3310 | 0.1580 |

Expected directions per the brief: (a) increase as alpha rises toward 1; (b) increase as alpha falls toward 0; (c) increase as alpha rises toward 1; (d) decrease as alpha rises toward 1 (trivially 1.0 at alpha=0.0 itself, since that IS the complement-mode reference).

### Endpoint comparison against phase 12d's own sweep (same checkpoint family, shared-trunk architecture)

| Metric | Phase 12d alpha=0.0 (complement) | Phase 17 alpha=0.0 (complement) | Phase 12d alpha=1.0 (substitute) | Phase 17 alpha=1.0 (substitute) |
|---|---|---|---|---|
| Recall@10 | 0.0971 | 0.1202 | 0.0667 | 0.0573 |
| Visual sim (a) | 0.7217 | 0.7142 | 0.7498 | 0.7492 |
| Co-occur hit rate (b) | 0.1020 | 0.1260 | 0.0760 | 0.0710 |
| Overlap w/ raw SigLIP (c) | 0.1708 | 0.1132 | 0.3428 | 0.3310 |

- Axis 1 gap (substitute - complement, visual sim): phase 12d = 0.0281, **phase 17 = 0.0350**
- Axis 2 gap (complement - substitute, hit rate): phase 12d = 0.0260, **phase 17 = 0.0550**
- Decisive gap (substitute-vs-raw minus complement-vs-raw overlap): phase 12d = 0.1720, **phase 17 = 0.2178** (phase 12c's own stopping-condition threshold for a 'real' gap was 0.03)

