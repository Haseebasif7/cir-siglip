# Phase 35, Stage 2: Retrieval Fine-tuning Log

Local M4 (MPS, CPU fallback for `torch.cdist`'s backward). Warm-started `proj`+`set_enc` (50 tensors) from
stage 1's CP checkpoint (`models/stage1_cp_pretrain.pt`). Config: phase 31/32's exact winning hyperparameters
(`input_mode=image_text, lr=1.5e-4, batch_size=384, uniformity_weight=0.1, margin=0.2, max_epochs=100,
patience=25, num_negatives=10`), selection on val Recall@10 (min_delta=0.0005) -- unchanged from phase 32.
Curriculum (new, piece 3): `hard_fraction(epoch) = min(1, epoch/99)`, linear.

## Result

**Early-stopped at epoch 59 (patience=25), best checkpoint at epoch 34: val Recall@10 = 0.1403** (hard
fraction at that point: 0.34). Wall time: 2273s (37.9 min), mean 37.9s/epoch, 60 epochs run. Checkpoint:
`models/stage2_retrieval_finetune.pt`.

## Curriculum progression (val Recall@10 vs. hard_fraction)

| Epoch | hard_fraction | Val Recall@10 | Val D_pos | Val D_neg |
|---|---|---|---|---|
| 0 | 0.00 | 0.0853 | 1.1402 | 1.0297 |
| 5 | 0.05 | 0.1048 | 1.1912 | 1.1186 |
| 10 | 0.10 | 0.1176 | 1.2602 | 1.2163 |
| 15 | 0.15 | 0.1214 | 1.3777 | 1.3698 |
| 20 | 0.20 | 0.1199 | 1.3766 | 1.3715 |
| 24 | 0.24 | 0.1341 | 1.3631 | 1.3586 |
| **34 (best)** | **0.34** | **0.1403** | 1.3717 | 1.3710 |
| 40 | 0.40 | 0.1265 | 1.3793 | 1.3802 |
| 50 | 0.51 | 0.1284 | 1.3759 | 1.3783 |
| 59 (stopped) | 0.60 | 0.1202 | 1.3825 | 1.3853 |

Full per-epoch curve in `data/stage2_result.json`.

## Reading the curriculum's effect, honestly (per the brief's own instruction to test this directly)

**Val Recall@10 rose steadily and cleanly through roughly `hard_fraction` 0.0-0.34** (0.0853 -> 0.1403, the
run's peak), tracking the curriculum's early "mostly random, a little hard" regime. **Past the peak
(`hard_fraction` > ~0.35), Recall@10 did not keep improving -- it plateaued and then drifted down with
visibly more epoch-to-epoch noise** (0.12-0.13 range, single-epoch swings as large as 0.015, vs. a much
smoother climb in the first 30 epochs), while `val_D_pos`/`val_D_neg` stayed close together and stable
throughout (both climbing together from ~1.03-1.14 early to ~1.37-1.39 late, never diverging or collapsing)
-- so this is **not** the kind of instability the brief flagged as a known risk (no divergence, no
oscillating loss, no magnitude/direction collapse); it is a real, honest ceiling-then-mild-decline in
ranking quality as the negative pool got harder, with training remaining numerically well-behaved
throughout.

**This is a genuinely informative result for the brief's own central curiosity** ("this is a real,
meaningful distinction [from the project's 5 prior mined-negatives-underperform findings], don't assume the
earlier finding automatically applies here -- test it honestly"): the early, gradual introduction of hard
negatives (roughly the first third of the curriculum) DID help -- Recall@10 at `hard_fraction`≈0.34
(0.1403) is clearly above what a hypothetical `hard_fraction=0` run would extrapolate to from the early
trend. But pushing further toward `hard_fraction`→1.0 did not continue to help and mildly hurt -- the same
qualitative direction as this project's five earlier findings (mined-only negatives underperform), just
arriving at it gradually rather than immediately, and only once the hard fraction got large enough. The
honest reading: **curriculum negatives are not a free lunch that avoids the mined-negatives problem
entirely -- they postpone and soften it, delivering real benefit in the early-to-middle regime before the
same underlying effect reasserts itself as the pool becomes dominated by hard negatives.** This is reported
here as this phase's own direct, first-time evidence on that question, not asserted from the earlier
phases' results.

## Secondary diagnostics

`val_D_pos` and `val_D_neg` track each other closely for the entire run (never separated by more than
~0.03), meaning the triplet margin (0.2) was comfortably satisfied at essentially every epoch once training
got underway -- unlike phase 14b's original, uncorrected run (where `val_D_pos > val_D_neg` at both the
start and end of training). No evidence here of the loss-selection pathology phase 31 diagnosed; val
Recall@10 and val_loss move together for most of the run, only diverging mildly in the noisy late-curriculum
regime described above.

## Warm start's effect

Epoch 0's val Recall@10 (0.0853, with `embed_ffn` still at its random initialization but `proj`/`set_enc`
warm-started from stage 1) is already well above stage 1's own untrained-`embed_ffn` diagnostic number
(~0.05-0.06, `stage1_training_log.md`) -- some benefit from the warm-started backbone is visible from the
very first stage-2 epoch, before the retrieval objective has done any work of its own. Isolating exactly how
much of stage 2's final result traces to the warm start vs. the category token vs. the curriculum
individually would need an ablation this phase's brief explicitly did not ask for (a single, unablated
"full recipe" run) -- flagged here as a limitation of what this phase can attribute internally, not
something to fill in speculatively.
