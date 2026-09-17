# Phase 35, Stage 1: CP Pre-training Log

Local M4 (MPS, CPU fallback for `torch.cdist`'s backward -- see implementation_notes.md). Config: batch=192
outfits (384 real+fake sequences/step), lr=1.5e-4 (AdamW+OneCycleLR, same schedule shape as stage 2),
max_epochs=40, patience=8, selection on val CP accuracy (min_delta=0.0005).

## Result

**Early-stopped at epoch 31 (patience=8), best checkpoint at epoch 23: val CP accuracy = 0.9327.**
Wall time: 1602s (26.7 min), mean 50.1s/epoch, 32 epochs run. Checkpoint:
`models/stage1_cp_pretrain.pt`. n_params=998,273.

| Epoch | Train loss | Train acc | Val loss | Val acc | Val R@10 (secondary) |
|---|---|---|---|---|---|
| 0 | 0.0890 | 0.5043 | 0.0736 | 0.5001 | 0.0100 |
| 5 | -- | -- | -- | ~0.86 | ~0.06 |
| 12 | 0.0163 | 0.9330 | 0.0175 | 0.9223 | 0.0589 |
| **23 (best)** | 0.0115 | 0.9564 | 0.0184 | **0.9327** | 0.0585 |
| 31 (stopped) | 0.0093 | 0.9665 | 0.0276 | 0.9156 | 0.0546 |

Full per-epoch curve in `data/stage1_result.json`.

## Reading the result

**The CP task learned cleanly and well**, going from chance (0.50) to 0.9327 val accuracy distinguishing
real Polyvore outfits from shuffled-item fakes -- no collapse, no divergence, a clean plateau-then-mild-
overfit pattern after epoch 23 (train acc kept climbing to 0.9665 by epoch 31 while val acc fell back to
0.9156, textbook overfitting past the selection point, which is exactly what patience=8 is for). This
confirms the transformer set-encoder + outfit token architecture is a straightforward fit for whole-set
binary classification on frozen SigLIP features -- no L2-normalization or uniformity-regularizer safeguard
was needed here (unlike this project's own earlier compatibility-model collapse episodes), likely because
CP's own MLP head + focal loss doesn't share the embedding-uniformity failure mode a contrastive/triplet
objective can hit.

**The secondary val Recall@10 diagnostic stayed low throughout (0.05-0.06) and did not track val CP
accuracy at all** -- exactly as anticipated in `implementation_notes.md`: `embed_ffn` (the head that
actually produces retrieval query/candidate embeddings) is never touched by the CP objective, which reads
out through the entirely separate `cp_head`. A ~0.05-0.06 Recall@10 with an untrained, randomly-initialized
`embed_ffn` is itself an unsurprising near-floor number, not evidence the recipe is failing -- it simply
confirms CP pre-training and retrieval quality are, as the brief itself warned, not expected to be
correlated at this stage. The real test of whether CP pre-training helped happens in stage 2, once
`embed_ffn` is actually trained on the retrieval objective (warm-started from stage 1's `proj`+`set_enc`
only).

## One operational note

A transient warning appeared around epoch 29-30 ("the volume ... is out of space") from an internal MPS
temp-file write (unrelated to this script's own checkpoint/JSON saves, which use ordinary paths and
succeeded without error both here and at run end). Disk had ~2.7GB free at the time this was checked after
the run finished; training completed and saved cleanly regardless. Flagged here for visibility going into
stage 2's longer run, not acted on (not this phase's scope to manage disk space, and the run was
unaffected).

## Warm start into stage 2

Only `proj` and `set_enc` (the shared item-encoding transformer backbone) are carried into stage 2, per the
paper's own stated procedure -- `embed_ffn` (new, retrieval-only) and `cp_head`/`outfit_token`'s CP-specific
role are not reused. See `02_stage2_retrieval_finetune.py`'s warm-start log line for confirmation of exactly
which tensors were copied.
