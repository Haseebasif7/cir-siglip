# Phase 29: Training Log

## Step 2: learning rate check

Short runs (6 epochs, patience 3) at three learning rates, compared on the validation benchmark before committing to the full single-seed run.

| LR | Tag | Best epoch | Val Recall@10 |
|---|---|---|---|
| 0.0005 | half | 2 | 0.1228 |
| 0.001 | current | 2 | 0.1048 |
| 0.002 | double | 3 | 0.0877 |

**Chosen LR: 0.0005 (half)** -- beat phase 28's original lr=0.001 on this short check, so the full single-seed run in step 3 uses this value instead of a blind carryover. Note all three learning rates land well below phase 28's mean-pool result (0.1819) even at their best -- the LR check already hints the architecture itself is the limiting factor, not the learning rate, confirmed by the full single-seed run below.

## Step 3: single-seed run (seed=42, lr=0.0005), full training curve

| Epoch | Train loss | Val Recall@10 | Val Recall@30 | Val Recall@50 | Mean attn entropy | Mean attn entropy (normalized) |
|---|---|---|---|---|---|---|
| 0 | 3.392 | 0.1091 | 0.2098 | 0.2761 | 1.288 | 0.845 |
| 1 | 2.843 | 0.1210 | 0.2288 | 0.2964 | 1.204 | 0.792 |
| 2 (best) | 2.699 | 0.1228 | 0.2331 | 0.3002 | 1.181 | 0.778 |
| 3 | 2.595 | 0.1193 | 0.2297 | 0.2992 | 1.183 | 0.780 |
| 4 | 2.506 | 0.1194 | 0.2269 | 0.2955 | 1.186 | 0.782 |
| 5 | 2.431 | 0.1152 | 0.2217 | 0.2921 | 1.187 | 0.782 |
| 6 | 2.368 | 0.1152 | 0.2263 | 0.2946 | 1.188 | 0.784 |

Early stopping fired cleanly at epoch 6 (patience=4 past the epoch-2 best). Training loss keeps falling smoothly through epoch 6 while validation Recall@10 peaks at epoch 2 and then degrades -- the model keeps improving at the training-time task (predicting the true positive given true-positive-conditioned attention) while getting worse at the actual evaluation task (ranking the true positive among a pool of unconditioned-on candidates). This is exactly the signature of the train/eval conditioning mismatch diagnosed in `architecture_notes.md` and confirmed directly in `attention_qualitative.md`: the more the model specializes `candidate_key`/`context_query` to the one conditioning candidate it ever sees in training (the true positive), the less useful its attention becomes when conditioned on the negative-heavy candidate pool it faces at evaluation time.

Attention entropy never collapses (normalized entropy stays in 0.78-0.85 throughout, well above 0 and below the uniform ceiling of 1.0) -- the mechanism is genuinely engaged, not degenerated into either failure mode the brief warned about (uniform-collapse or single-item-collapse). The problem is not that attention isn't doing anything; it's that what it's doing doesn't generalize past the one conditioning candidate it was trained against.
