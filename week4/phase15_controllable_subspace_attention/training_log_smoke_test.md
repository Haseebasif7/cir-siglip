# Phase 15: Smoke Test Report

## Check 1: embedding collapse (with vs. without uniformity regularizer)

5 epochs, 2000 train outfits, 400 val outfits, weight_sub=9.5801 (from loss_balancing_check.md), alpha ~ Uniform(0,1) each step.

- WITHOUT uniformity regularizer: mean pairwise cosine similarity = 0.7565
- WITH uniformity regularizer (weight=1.0): mean pairwise cosine similarity = 0.4039

**Real collapse confirmed without the regularizer** (0.7565 is far above the untrained baseline) -- matches phase 13/13b/14's generic direction-collapse failure mode resurfacing here as expected. Uniformity regularizer (weight=1.0) is enabled from the start of both real training runs.

## Check 2: conditioning-consistency (design decision 5, the alpha double-duty confound)

Re-run the WITH-uniformity smoke config, but with `decoupled_alpha_frac=0.2` -- 20% of steps feed attn_net a forward-pass alpha sampled independently from the loss-weighting alpha, so the conditioning pathway stays live across the full alpha range even near the envelope's own alpha=0/1 endpoints.

Ran 5 epochs with 20% decoupled-alpha steps, no crash, final val_loss=-1.5934 -- training remains stable under decoupled alpha, so this safeguard is cheap to keep on for the smoke test (NOT proposed for the real runs, which use alpha_forward == alpha_weight throughout, matching the plan's design decision 3).

## Check 3: attention-weight-shift probe (does attn_net actually use alpha?)

For a fixed sample of all 121 (cat_s, cat_t) category pairs, sweep the FORWARD-PASS alpha alone from 0 to 1 (no loss involved) and measure the mean L1 distance between the alpha=0 and alpha=1 softmax attention-weight vectors (max possible L1 distance for a 5-way softmax pair is 2.0).


| Model | Attention-weight L1 shift (alpha 0 vs 1) |
|---|---|
| Untrained (random init) | 0.0405 |
| Smoke, no uniformity, alpha coupled | 0.0242 |
| Smoke, with uniformity, alpha coupled | 0.0269 |
| Smoke, with uniformity, 20% decoupled alpha | 0.0282 |

**Still near the untrained baseline (0.0405) after only 5 smoke-test epochs on 2000 outfits** -- this is expected at this scale (the smoke test is deliberately tiny) and is NOT yet evidence the confound is real or fixed; re-run this exact probe on both real trained checkpoints (01_train_full.py, 01b_train_discrete.py) before drawing any conclusion about whether attn_net learns to condition on alpha -- see phase15_notes.md for that final measurement.

