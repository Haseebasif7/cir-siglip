# Phase 14: Training Log

## Step 3 (done first, before the real run): failure-mode safeguards, checked directly

### Smoke test 1: dead-gradient / OOM check, and D_pos/D_neg baseline

Ran 5 epochs on a 2,000-outfit subset (batch=96, `use_uniformity=False`
first). No OOM (expected, frozen backbone -- confirmed anyway). Timing:
8.95s for 5 epochs on this small subset, no crash.

**Bug found and fixed before this**: the first attempt crashed with
`ValueError: Tried to step 31 times. The specified number of total steps is
30` from `OneCycleLR`. Cause: `steps_per_epoch` was computed via floor
division (`len(train_outfits) // batch_size`), but the training loop's
`range(0, N, batch_size)` actually produces `ceil(N / batch_size)` batches
per epoch -- an off-by-one that would have crashed the real 100-epoch run
partway through, not just the smoke test. Fixed by switching to
`math.ceil` plus a small safety buffer in `train_core.py`.

**D_pos/D_neg WITHOUT the uniformity regularizer** (5 epochs, 2,000
outfits): gap (D_pos - D_neg, wrong sign is D_pos > D_neg) went 0.180 ->
0.089 -> 0.061 -> 0.053 -> 0.052 -- both D_pos and D_neg were ALSO shrinking
together every epoch (D_pos 0.60 -> 0.23, D_neg 0.42 -> 0.18), which is the
same "distances shrinking toward each other" signature phase 13's original
attempts flagged as a possible collapse precursor. Checked directly rather
than assumed:

```
mean pairwise cosine similarity, candidate embeddings, WITHOUT uniformity reg: 0.974
mean pairwise cosine similarity, candidate embeddings, untrained (random init): 0.806
```

**Real direction collapse, confirmed** -- 5 epochs on a small subset was
enough to push embeddings to near-identical directions (0.974 average
cosine similarity between unrelated items). This is exactly phase 13's
generic (backbone-independent) failure mode #4, resurfacing here as the
brief anticipated it might.

### Smoke test 2: same setup, `use_uniformity=True`

```
mean pairwise cosine similarity, candidate embeddings, WITH uniformity reg: 0.026
```

Collapse fixed. Distances also stopped shrinking toward zero (D_pos ~1.29,
D_neg ~0.98 by epoch 4, vs. ~0.23/0.18 without the regularizer) -- the
model is now using the embedding space's actual volume instead of
collapsing into a shrinking point. **The uniformity regularizer (weight
1.0, identical formula to phases 13/13b) was therefore enabled from the
start of the real training run below, not added after discovering a
problem partway through a full run.**

### Gradient sanity check

One `loss.backward()` on a real 96-outfit batch, `use_uniformity=True`,
untrained model: every parameter -- `proj.weight/bias`, all 4 transformer
layers' self-attention/FFN/LayerNorm weights, `embed_ffn.weight`, and the
`outfit_token` itself -- had a nonzero gradient norm (ranging ~0.03 to ~16
across parameters, no zeros). No dead-gradient trap; standard PyTorch
initialization for `nn.Linear`/`nn.TransformerEncoderLayer` was sufficient
here (unlike CSA-Net's multiplicative masks, there's no near-zero-init
multiplicative structure in this architecture to cause one).

## Step 4: local vs. Modal decision

1-epoch timing test, full 53,306 training outfits, batch=96,
`use_uniformity=True`: **24.5 seconds**. Projected a 40-epoch run (phase
13b's own budget) at ~16 minutes. **No Modal GPU spending was triggered or
needed** -- local was clearly practical from the first measurement, so a
larger epoch budget (100, with early-stopping patience=8) was used instead
of matching phase 13b's 40 exactly, since compute was not the constraint.

## The real training run

Launched locally (MPS) under `caffeinate -is` (the lesson from phase 13c's
laptop-sleep interruption applied proactively here, before it could
recur). Ran uninterrupted.

- **Total wall time: 1,852.4s (~30.9 min)** for 100 epochs, well within the
  smoke test's projection.
- **Early-stopping patience (8 epochs) was reached right at the scheduled
  end** (epoch 99 = last epoch anyway, since `max_epochs=100`) -- this is a
  completed run, not a truncated one; LR had already decayed to
  ~8e-11 (OneCycleLR's cosine anneal) by the final epoch, i.e. the model
  had stopped moving well before the run ended regardless of the patience
  counter.
- **Best checkpoint: epoch 91** (`val_loss = -3.5050`), per the same
  `>1e-4`-improvement save rule used in phases 13/13b/13c.

### D_pos/D_neg ranking diagnostic -- monitored continuously, not just at the end

| Epoch | train_loss | val_loss | val_D_pos | val_D_neg | gap (D_pos - D_neg) |
|---|---|---|---|---|---|
| 0 | -2.2253 | -2.4542 | 1.2906 | 0.9217 | 0.3689 |
| 10 | -3.3877 | -3.1074 | 1.3916 | 1.3452 | 0.0464 |
| 20 | -3.4813 | -3.1812 | 1.4090 | 1.3914 | 0.0176 |
| 30 | -3.5144 | -3.3589 | 1.4163 | 1.4035 | 0.0127 |
| 40 | -3.5298 | -3.4263 | 1.4143 | 1.4036 | 0.0107 |
| 50 | -3.5386 | -3.4789 | 1.4134 | 1.4048 | 0.0086 |
| 60 | -3.5429 | -3.4857 | 1.4135 | 1.4066 | 0.0070 |
| 70 | -3.5461 | -3.4958 | 1.4134 | 1.4073 | 0.0061 |
| 80 | -3.5483 | -3.5014 | 1.4138 | 1.4078 | 0.0059 |
| 90 | -3.5495 | -3.5036 | 1.4136 | 1.4080 | 0.0057 |
| **91 (best)** | -3.5491 | -3.5050 | 1.4137 | 1.4080 | **0.0056** |
| 99 (final) | -3.5502 | -3.5040 | 1.4137 | 1.4081 | 0.0056 |

**The gap closes steadily and monotonically from 0.369 at epoch 0 down to
~0.006 by epoch ~80, then plateaus there** -- dramatically closer to zero
than CSA-Net-on-SigLIP ever got in either phase 13b (plateaued at 0.082)
or phase 13c (plateaued at 0.120, worse). By this internal diagnostic
alone, this configuration looks like the best-converged mechanism this
project has trained on this backbone. **This impression turned out to be
misleading -- see `phase14_notes.md` for why the CIR Recall@K result tells
a very different story, and the specific reason this diagnostic doesn't
transfer here.**

### Final-model collapse check (not just the smoke-test checkpoint)

The uniformity check above was run on the small-subset smoke-test
checkpoint. Re-checked directly on the actual final trained model's
precomputed candidate embeddings (all 251,008 items, see
`02_extract_candidate_features.py`'s output) before trusting the Recall@K
result in `results_table.md`:

```
mean pairwise cosine similarity, candidate embeddings, FINAL model, N=2000 sample: 0.004
std: 0.143, range: [-0.559, 0.985]
all embedding norms: 1.0 (L2-normalization confirmed working)
```

No collapse in the final model either. The low Recall@K result reported in
`results_table.md` is real model behavior, not a normalization or collapse
bug.
