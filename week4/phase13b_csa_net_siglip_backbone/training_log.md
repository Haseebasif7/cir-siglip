# Phase 13b: Training Log

## Step 2: verifying phase 13's known failure modes don't resurface

Checked directly, not assumed, before trusting any results (same evidence
standard phase 13 used):

- **Dead gradients**: the same near-identity mask init was verified with a
  direct forward/backward on random inputs before any real training --
  `masks grad norm = 0.250`, `proj grad norm = 15.77` (healthy, non-trivial
  gradients from the very first step; see the sanity-check output preserved
  in this phase's build process). No dead-gradient issue.
- **Magnitude collapse**: L2-normalization is unconditionally applied in
  `embed_from_feature` (copied unchanged from phase 13) -- structurally
  impossible for embeddings to shrink toward the origin regardless of
  training dynamics.
- **Direction collapse**: the uniformity regularizer is also copied
  unchanged and active from the first training step. Confirmed empirically
  on real data: an 8-epoch, 1,000-outfit local smoke test showed D_pos/D_neg
  climbing from ~0.48/0.41 to ~1.78/1.67 (embeddings actively spreading
  apart, the opposite of collapse), matching the healthy pattern phase 13
  eventually reached only after all four of its fixes. On the FULL run,
  D_pos/D_neg stayed in the 1.85-1.93 / 1.75-1.83 range for the entire
  40-epoch run -- stable, never trending toward 0.
- **OOM**: not applicable and not checked for -- there is no CNN forward
  pass in this phase at all (see `architecture_notes.md`), so the specific
  memory pattern that caused phase 13's OOM (thousands of images through
  ResNet18 with gradients) cannot occur here. A full 96-outfit batch's
  worth of SigLIP-vector lookups is a `(~1500, 768) -> (~1500, 64)` matrix
  operation, negligible memory regardless of device.

**None of phase 13's four failure modes resurfaced.** Training was stable
from the first step.

## Step 3: training run

Ran locally (MPS), no GPU/Modal needed. Full config:

| Setting | Value |
|---|---|
| Optimizer | Adam, initial LR 5e-5, linear decay to 0 (matches phase 13/the paper) |
| Batch size | 96 outfits (paper's own value -- no OOM risk here, so no need for phase 13's gradient-accumulation workaround) |
| Negatives per sample | 10, same mined candidates phase 13 used (see `architecture_notes.md`) |
| Max epochs / patience | 40 / 5 |
| Margin / aggregation | 0.3 / min (unchanged from phase 13) |

**Ran the full 40-epoch schedule to completion -- never triggered early
stopping.** Total wall time: 24,872s (~414.5 min, ~6.9 hours) on a local M4
Air. The LR schedule reached exactly 0 at epoch 39 as designed. Early
stopping (patience=5) never fired because the validation loss kept finding
tiny new best values (deltas as small as 0.0001-0.0005) throughout training
-- best checkpoint is from **epoch 38** (`val_loss = -3.4863`).

### Full training curve (selected epochs; full data in `models/training_curves.json`)

| Epoch | val_loss | val_D_pos | val_D_neg | D_pos - D_neg gap |
|---|---|---|---|---|
| 0 | -3.4296 | 1.892 | 1.783 | 0.109 |
| 5 | -3.4673 | 1.888 | 1.792 | 0.096 |
| 10 | -3.4739 | 1.880 | 1.790 | 0.090 |
| 15 | -3.4782 | 1.874 | 1.788 | 0.086 |
| 20 | -3.4804 | 1.871 | 1.787 | 0.085 |
| 25 | -3.4835 | 1.870 | 1.786 | 0.083 |
| 30 | -3.4847 | 1.868 | 1.786 | 0.082 |
| 35 | -3.4858 | 1.868 | 1.787 | 0.082 |
| 38 (best) | -3.4863 | 1.868 | 1.787 | 0.082 |
| 39 (final) | -3.4863 | 1.868 | 1.787 | 0.082 |

**Honest read of this curve** (see `phase13b_notes.md` for the full
interpretation): val_loss did genuinely keep improving through all 40
epochs -- this is a real, converged result, not a budget-truncated one like
phase 13. But almost all of that improvement happened in the first ~15
epochs; the D_pos-D_neg ranking gap dropped from 0.109 to 0.086 by epoch 15,
then barely moved for the remaining 25 epochs (0.086 -> 0.082). The model
reached a stable equilibrium where the true positive item is, on average,
still farther from the outfit context than the hardest of its 10 mined
negatives -- training converged, but not to a point where D_pos < D_neg.
