# Phase 33, Step 1: Checkpoint Selection Check

## What phase 13b actually used

`week4/phase13b_csa_net_siglip_backbone/scripts/train_core.py`, lines 221-230 -- functionally identical
code to phase 14b's own bug:

```python
if val_loss < best_val_loss - 1e-4:
    best_val_loss = val_loss
    epochs_since_improve = 0
    torch.save(model.state_dict(), out_dir / "csa_net_siglip_best.pt")
else:
    epochs_since_improve += 1
    if epochs_since_improve >= patience:
        break
```

Checkpoint selection and early stopping are keyed to validation LOSS, never Recall@K -- Recall@K was
never computed during training at all, exactly the same class of problem phase 31 found and fixed for
OutfitTransformer.

## But the underlying dynamics are genuinely different, and the practical consequence is much smaller

The unresolved-margin signature IS present here too: at both epoch 0 (val_D_pos=1.888 > val_D_neg=1.779)
and epoch 39 (val_D_pos=1.871 > val_D_neg=1.789), the triplet margin (0.3) is never fully satisfied --
`relu(D_pos - D_neg + margin)` stays positive throughout, the same mechanistic pattern phase 31 diagnosed
for OutfitTransformer. **But unlike OutfitTransformer, this does not translate into a broken checkpoint
selection.**

Direct A1 (val_loss selection, phase 13b's original patience=5) vs. A2 (Recall@10 selection, SAME
patience=5, no recalibration) comparison, both at phase 13b's exact original hyperparameters
(lr=5e-5, batch_size=96, max_epochs=40):

| Run | Selection metric | Patience | Best epoch | Epochs run | Val Recall@10 |
|---|---|---|---|---|---|
| A0 | -- (phase 13b's existing checkpoint, no retrain) | -- | -- | -- | 0.0786 |
| A1 | val_loss | 5 | 39 | 40 (never early-stopped) | 0.0777 |
| **A2** | **recall10** | **5 (unchanged)** | 30 | 36 (clean early stop) | **0.0779** |

**A2 and A1 land within 0.3% relative of each other.** The naive fix -- just swap the selection metric,
keep phase 13b's original patience=5 unchanged -- works fine here. **This is a genuine, notable
architecture-specific difference from OutfitTransformer**, where the identical naive fix caused a
catastrophic regression (val Recall@10 0.017 vs. 0.064, phase 31's `checkpoint_selection_check.md`) because
Recall@10 had a real ~17-epoch noisy plateau/dip early in training that a val_loss-tuned patience=8
couldn't survive.

## Why the difference: CSA-Net's Recall@10 trajectory is smooth, not volatile

A1's own tracked (not selected-on) Recall@10 curve climbs almost monotonically across all 40 epochs
(0.0568 at epoch 0 -> 0.0777 at epoch 39, `data/checkpoint_selection_results.json`'s full curve) -- no
dramatic early dip or multi-epoch noisy plateau of the kind that broke OutfitTransformer's naive fix. Since
Recall@10 is already close to monotonically improving on this architecture, even an imperfect proxy metric
(val_loss, dominated by the same unresolved-margin/uniformity dynamic present in both architectures) still
lands on a near-optimal epoch by coincidence of the underlying trajectory's shape, not because the proxy
itself is a good one in principle.

## Verdict and what this phase adopts going forward

**No patience recalibration needed for CSA-Net.** `selection_metric="recall10", patience=5` (unchanged from
phase 13b) is adopted as the standing configuration for every subsequent step in this phase (text
integration, tuning, ensembling) -- confirmed sufficient by A2's own clean, non-premature early stop. This
is reported as a genuine, real difference from OutfitTransformer's investment story, not smoothed over: the
mechanism that made the fix matter a great deal for one architecture (patience/plateau mismatch) simply
isn't present for this one, even though the underlying loss-shape issue (unresolved margin) is shared by
both.

The selection FIX itself (metric swap) is still adopted -- A2's 0.0779 is used going forward, not A1's
0.0777 -- both because it is this phase's stated methodology (Recall@10 selection throughout, per every
prior phase's convention) and because, even though the two numbers are close here, Recall@10 is still the
metric this project actually cares about optimizing for.
