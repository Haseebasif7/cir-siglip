# Phase 12b: Controllable Mode Embeddings, Retest With the Diagnosed Fix

## Context

Phase 12 tested a shared projection + two additive "mode vectors" (substitute, complement)
for steerable retrieval, and found it didn't work: the two modes retrieved almost
identical results (76.4% top-10 overlap, 0.829 average per-item cosine). Root cause
pinned down precisely: the two training losses differed in raw magnitude by ~400x
(complement's MNRL ~4.0-4.3, substitute's batch-local pairwise-MSE distillation ~0.011),
summed unweighted into the same shared network, so the much larger complement loss
dominated and left the mode vectors too weak to matter.

This phase is the isolated retest with that diagnosed fix applied, same architecture and
same evaluation as phase 12, so any change in outcome traces specifically to the fix.
**Two changes only**, everything else held fixed:

1. **Substitute loss reformulated** (step 1): batch-local pairwise-similarity matching
   replaced with a direct, global, per-item target -- `1 - cosine(z_sub_i, pca_target_i)`,
   where `pca_target_i` is a FIXED PCA-128 projection of item i's own raw SigLIP embedding
   (computed once via eigendecomposition, no sklearn, 73.7% of total variance retained --
   see `data/pca_variance_check.md`). A literal per-item cosine against the full 768-d raw
   embedding isn't dimensionally defined against z_sub's 128-d output, so this fixed
   projection makes "match the item's own raw embedding directly" well-defined while
   staying global and query-independent, unlike phase 12's batch-local objective. Full
   rationale in `scripts/01_compute_raw_targets.py`'s docstring.
2. **Loss balancing measured and verified, not assumed** (step 2): initial magnitudes
   remeasured under the new formulation (not assumed to match phase 12's old ratio) --
   found complement ~4.90, substitute ~1.00, a ~4.9x ratio (much closer than phase 12's
   ~400x, on its own evidence the reformulation already helped). Applied
   `weight_sub = initial_comp / initial_sub = 4.89`. **Explicitly verified** via a
   gradient-norm check into the shared base projection, isolating each loss term: before
   weighting, complement=0.478 vs substitute=0.307 (already only 1.6x apart); after
   weighting, complement=0.478 vs substitute=1.498 (0.3x, i.e. substitute now larger) --
   both comfortably within the "reasonable order of magnitude" the brief asked to confirm.
   Full numbers: `loss_balancing_check.md`.

## Step 3: retraining

Same architecture, same complement loss/positive edges, same hyperparameters and
early-stopping procedure as phase 12 -- only the two items above changed. Converged
cleanly (best checkpoint epoch 0, early-stopped epoch 5, `models/training_curves.json`,
same "best epoch is epoch 0" pattern as phase 9/12 at this edge-count scale). One
striking early signal, visible in the training curves before any evaluation was run:
**substitute mode's collapse-guard metric (mean pairwise cosine across a 256-item sample)
dropped to ~0.001, essentially the value expected for random unit vectors spread evenly
across a 128-d hypersphere, while complement mode's stayed at ~0.60-0.62** (phase 12 had
both in the same 0.55-0.65 range). This was the first hint that the fix had genuinely
changed substitute mode's geometry, not just its loss curve.

## Step 4: re-running phase 12's exact evaluation

### 4.1 -- CIR benchmark Recall@K (`results_table.md`)

| Configuration | Recall@10 | Phase 12 | Recall@30 | Phase 12 | Recall@50 | Phase 12 |
|---|---|---|---|---|---|---|
| Raw SigLIP (reference) | -- | 0.0553 | -- | 0.1067 | -- | 0.1437 |
| Substitute mode | **0.1212** | 0.1329 | **0.2254** | 0.2467 | **0.2957** | 0.3215 |
| Complement mode | **0.1200** | 0.1335 | **0.2262** | 0.2507 | **0.2963** | 0.3250 |
| Blend (0.5) | **0.1207** | 0.1366 | **0.2261** | 0.2524 | **0.2957** | 0.3289 |

Both modes' Recall@K dropped slightly from phase 12 (consistent with complement mode's
own validation loss also rising a bit more this run -- see "a real cost" below) but
**neither moved anywhere close to raw SigLIP's much lower range**, and the two modes
still track each other almost exactly. Recall@K alone -- as phase 12 already showed --
is not sensitive enough to tell the real story here; the diagnostic below is.

### 4.2 -- control-effectiveness diagnostic, with follow-ups (`control_effectiveness.md`)

| Check | Phase 12 | Phase 12b | Read |
|---|---|---|---|
| Axis 1 gap (substitute − complement, visual sim) | +0.0073 | **+0.0005** | confirmed, but gap shrank |
| Axis 2 (complement hit rate − substitute hit rate) | 0 (exact tie) | **−0.0070** | tie broken, wrong direction |
| Top-10 overlap between modes | 0.7636 | **0.5102** | real, substantial drop |
| Per-item cosine(z_sub, z_comp) | 0.8288 | **0.5844** | real, substantial drop |
| **New check**: substitute-vs-raw-SigLIP top-10 overlap | -- | **0.1612** | |
| **New check**: complement-vs-raw-SigLIP top-10 overlap | -- | **0.1754** | substitute is NOT closer |

The follow-up checks (overlap, per-item cosine) that actually exposed phase 12's failure
show a **real, large improvement** -- the two modes are now genuinely different
embeddings, not a weakly-offset copy of one shared representation. Qualitative grids
confirm this visually: the shoes example (`qualitative_examples/..._shoes_fixed.png`)
shows substitute mode converging on pink floral strappy sandals while complement mode
returns a stylistically varied mix (a cat-print ankle boot, a bow pump, ballet flats) --
a real character difference, unlike phase 12's version of the same query where all three
rows showed the identical five shoes just reordered.

But the two axis-level premise checks did **not** clearly improve, and the added check
settles why. **Substitute mode's retrieval is not measurably closer to raw SigLIP's own
retrieval than complement mode's is (16.1% vs 17.5% overlap with raw SigLIP -- if
anything, slightly lower).** Substitute mode moved away from complement mode, but not
*toward* genuine visual similarity -- it moved toward something else. The likely
mechanism: the PCA-128 target retains 73.7% of raw SigLIP's total *variance*, a global,
aggregate property, but that does not guarantee it retains the *local nearest-neighbor
structure* that actually determines which specific items rank in a top-10 for a given
query -- fine-grained texture/color/shape relationships that drive real visual-similarity
retrieval may live partly in the ~26% of variance the projection discards. A representation
can diverge substantially from a shared baseline while still not resembling the intended
target.

### A real cost worth naming plainly

Complement mode's own validation loss got measurably worse this run (best val_comp 6.17
vs phase 12's 5.34, and complement's Recall@K also dropped slightly, 0.1335 -> 0.1200
@10). The most likely explanation: once the substitute objective's gradient into the
shared `net` became comparable in magnitude to complement's (by design, that was the
fix), the two objectives now genuinely compete for the same shared representation's
capacity, rather than complement effectively having the shared net to itself as it did
in phase 12. Fixing the imbalance in one direction introduced a new, smaller cost in the
other -- not a fatal one, but a real trade-off worth tracking if this mechanism is
iterated on further.

## Step 5: go/no-go verdict

The brief posed this as two clean outcomes: either the modes separate clearly (confirming
the additive architecture works, worth building on) or they still barely separate
(evidence the architecture itself is too weak, motivating a structurally different
mechanism). **The actual result is neither of these cleanly -- it splits the question in
a way worth stating precisely rather than forcing it into one bucket:**

- **The additive architecture is NOT too structurally weak to separate two objectives.**
  This is now disconfirmed directly: given a properly-balanced, direct, global training
  signal, the same shared-projection-plus-additive-mode-vector architecture produced a
  25-percentage-point drop in top-10 overlap and a real, visible qualitative character
  difference between modes. Phase 12's failure was a tuning/objective-design problem, not
  an architectural ceiling -- the fix targeted the right thing.
- **But this specific fix's substitute-mode target was not good enough to produce the
  actual intended behavior.** Substitute mode is now genuinely different from complement
  mode, but not genuinely *visual-similarity-like* -- it does not track raw SigLIP's own
  retrieval any better than complement mode does. The PCA-128 compression was likely too
  lossy in exactly the dimension that matters for retrieval (local neighbor structure,
  not global variance).

**Recommendation: do not escalate to the fuller CSA-Net-style conditioning architecture
yet, and do not conclude the additive mechanism is a dead end -- iterate once more on the
substitute-mode target specifically**, since the architecture itself just cleared the bar
the brief set for it. Concrete next try, cheap and reusing everything built in this
phase: replace the PCA-128 fixed target with a target that directly preserves per-item
*nearest-neighbor ranking* rather than *global variance* -- for example, a listwise/
ranking-distillation loss that pulls z_sub's similarity ordering toward each anchor's
actual top-K raw-SigLIP neighbors (already computable from the existing embeddings, no
new data needed), instead of regressing toward a single lossy fixed-dimension summary
vector. If that also fails to make substitute mode's retrieval resemble raw SigLIP's,
*that* would be the point to conclude the problem is more fundamental than the target
formulation and to consider the bigger architectural investment.

## Required-output checklist

- `loss_balancing_check.md` -- done, verification passed cleanly (gradient norms within
  1.6x/0.3x, not hundreds of times off).
- `results_table.md` -- done, phase 12 numbers included for direct comparison.
- `control_effectiveness.md` -- done, both axes, both follow-ups, plus the new
  substitute-vs-raw-SigLIP check that resolves the ambiguity the other checks left open.
- `qualitative_examples/` -- done, same 6 outfits as phase 12 (deterministic reuse of the
  same picking logic against the same benchmark file, confirmed identical picks).
- This file -- explicit go/no-go verdict above.
