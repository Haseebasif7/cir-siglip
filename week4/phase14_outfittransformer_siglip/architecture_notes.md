# Phase 14: Architecture Notes

## Step 1: what was actually checked (repo + paper), not assumed

### The repo (`github.com/bigohofone/outfit-transformer`)

Confirmed this is a real, non-fork repository (`git clone`, inspected via
`api.github.com/repos/bigohofone/outfit-transformer`: `"fork": false`,
description "Implementation of 'Outfit Transformer: Outfit Representations
for Fashion Recommendation (CVPR 2023)' with improvements in performance and
usability"). It is a *different* implementation from the similarly-named
`owj0421/outfit-transformer` that turned up in the same search, not the same
codebase under two URLs -- cloned and read the actual one named in the brief.

Phase 12 already established (`cir_protocol_notes.md`) that this repo's
*evaluation* code (`compute_cir_scores`) is mislabeled 4-way FITB accuracy,
not real candidate-pool Recall@K -- that finding was not re-litigated here.
This phase only needed the *architecture* code, which is a separate part of
the repo and was read fresh: `src/models/outfit_transformer.py`,
`src/utils/loss.py`, `src/data/datasets/polyvore.py`,
`src/run/3_train_complementary.py`.

**Confirmed architecture (`OutfitTransformer` class):**

- An item encoder (`ItemEncoder`) produces a per-item embedding from CLIP
  image + text features, concatenated (`aggregation_method='concat'`,
  `item_enc_dim_per_modality=128` each, so 256-dim combined).
- A learnable **task token** is prepended to a sequence of item embeddings,
  and the whole sequence is passed through `nn.TransformerEncoder`
  (`transformer_n_head=16`, `transformer_d_ffn=2024`,
  `transformer_n_layers=6`, `dropout=0.3`, `batch_first=True,
  norm_first=True, activation=F.mish`). The token's own output position
  (`last_hidden_states[:, 0, :]`) is read out as the outfit's representation,
  then passed through a final linear (`embed_ffn`) to `d_embed=128`.
- Three different call paths reuse the *same* `style_enc` transformer stack:
  - `predict_score` (compatibility prediction): task token = `[task_emb,
    predict_emb]` concatenated, prepended to ALL outfit items, sigmoid head
    on the output.
  - `embed_query` (complementary item retrieval, the task this phase
    reproduces): task token = `[task_emb, embed_emb]`, prepended to the
    outfit's CONTEXT items (one item held out), `embed_ffn` on the output.
  - `embed_item` (a candidate/target item's own embedding): **no task token
    at all** -- the item is run through the same transformer as its own
    length-1 sequence, and `embed_ffn` is applied to that single position's
    output. This is what makes candidate embeddings context-independent and
    precomputable once per catalog (same trick phase 13b used for CSA-Net's
    SigLIP variant).
  - Confirmed directly by reading the code: `embed_query` does **not** take
    the target item's category or description as input anywhere -- the paper
    describes a "target item token" conditioned on a target
    category/description, but this specific reference implementation does
    not implement that conditioning. This is a real, confirmed
    simplification already present in the actual reference code, not one
    introduced by this phase.
- **Training objective for retrieval** (`3_train_complementary.py`,
  `InBatchTripletMarginLoss`, `src/utils/loss.py`): for each outfit, one
  item is randomly chosen as the held-out target (`PolyvoreTripletDataset`),
  the rest is the query context. `query_emb = model.embed_query(context)`,
  `answer_emb = model.embed_item(target)`. Loss is an in-batch
  hardest-negative triplet margin loss on **Euclidean distance** between
  query and answer embeddings (`margin=2.0`, `torch.cdist`, hardest negative
  = any OTHER outfit's target item in the same batch, not a mined pool, and
  **not restricted to the same category** -- confirmed directly from the
  dataset code, no category filter present). Optimizer: `AdamW`, LR
  schedule: `OneCycleLR` (`max_lr=2e-5`, `pct_start=0.3`, `div_factor=25`,
  `final_div_factor=1e4`).
- `transformer_norm_out: bool = False` by default -- the repo does **not**
  L2-normalize `embed_query`/`embed_item`'s output before computing distance.

### The paper (via search of the primary WACV 2023 source and its CVPR 2022
workshop precursor, since a full PDF re-read line-by-line was not
practical in this session)

Confirms the same high-level mechanism the repo implements: a transformer
processes items as a *set*, self-attention lets items attend to each other,
an "outfit token" produces a holistic outfit representation for compatibility
prediction, and for retrieval specifically the paper additionally describes
a "target item token" that conditions on the target's category/description
(the piece the actual repo implementation above does not implement). The
paper's own stated training objective for retrieval is a "set-wise outfit
ranking loss" -- consistent in spirit with the repo's in-batch triplet
margin loss (both are ranking/margin-style losses over a query-vs-candidate
embedding space), though the repo's specific loss formula was taken as the
authoritative implementation detail here, per this project's established
practice (phase 13 also worked from the paper's OWN loss description where
no usable code existed; here, usable architecture code exists, so it takes
priority per the same "read the actual source" discipline).

## Step 2: how the frozen-SigLIP adaptation was actually implemented

`scripts/model.py`'s `OutfitTransformerSigLIP` mirrors the repo's mechanism
directly, with the deviations below. See the module's own docstring for the
condensed version; full reasoning here.

| Aspect | Repo (`bigohofone/outfit-transformer`) | This phase |
|---|---|---|
| Item features | CLIP image + text, concat, `d_embed=256` | Frozen SigLIP image only (768-dim), L2-normalized, projected to `d_model=128` |
| Why single-modality | -- | This project has never extracted a text encoder for these items in any phase (phases 9/12/13/13b all use SigLIP image embeddings only) -- reusing that established practice, not a new limitation introduced here |
| Transformer | 6 layers, 16 heads, `d_ffn=2024`, dropout 0.3 | 4 layers, 8 heads, `d_ffn=512`, dropout 0.1 -- the brief's own step 2 asks for "a small transformer encoder"; there is no full end-to-end CLIP fine-tuning signal here to justify the repo's larger capacity, and a smoke test (below) confirmed this size trains cleanly |
| Outfit/task token | Two-purpose token (`task_emb` + `predict_emb` or `embed_emb`, shared half) supporting both compatibility prediction and retrieval | A single-purpose learnable "outfit token" (only the retrieval task is needed here, matching the brief's own phrasing) |
| Target-category conditioning | None (confirmed above -- paper describes it, repo doesn't implement it) | None -- faithful to what the repo's `embed_query` actually does, not a simplification added here |
| Final embedding normalization | `transformer_norm_out=False` by default | **Forced to `True`** -- L2-normalize before any distance computation. Required by phase 13's own lesson: an unnormalized margin/triplet loss has a trivial "shrink everything toward the origin" solution. See step 3. |
| Triplet margin | 2.0 (sized for an unbounded/unnormalized embedding space) | 0.3 -- rescaled because L2-normalized embeddings live on a unit hypersphere where max Euclidean distance is 2, not unbounded; 0.3 matches this project's own CSA-Net margin (phases 13/13b/13c), already a known-working value on a unit hypersphere in this exact codebase |
| Negative sampling | In-batch hardest negative, no category restriction | Same -- faithful reproduction, confirmed from the dataset code (see step 1) |
| Optimizer/schedule | AdamW, OneCycleLR, `max_lr=2e-5` | Same values, same schedule shape |
| Training data | Official Polyvore Outfits splits via `owj0421/polyvore-outfits` on HuggingFace | Phase 13's already-built `training_data.json` (same official Polyvore Outfits train/val outfit lists, item-id lists per outfit, reused directly -- no negative-mining fields needed here since this phase uses in-batch negatives, not a mined pool) |
| Negative candidates file (`negative_candidates.json`) | N/A | Deliberately **not used** -- this phase's loss needs no mined negatives, unlike CSA-Net's phases 13/13b/13c |

## Step 3: known failure-mode safeguards, checked directly (not assumed)

Per the brief's explicit instruction, phase 13's three backbone-independent
failure modes (dead gradients, magnitude collapse, direction collapse) were
checked directly and early, via a smoke test on a 2,000-outfit subset --
full numbers and the resulting fix are in `training_log.md`. Short version:

- **OOM**: not applicable here for the same reason it wasn't in phase 13b --
  the backbone is frozen, so a full 96-outfit batch is a handful of small
  matrix operations through a 4-layer transformer, not a CNN
  forward+backward over ~1,500 images.
- **Dead gradients**: checked directly (all parameters' gradient norms
  printed after one `loss.backward()` on a real batch) -- every parameter,
  including the outfit token itself, had a healthy nonzero gradient. No
  special initialization was needed beyond PyTorch's own defaults for
  `nn.Linear`/`nn.TransformerEncoderLayer` (unlike CSA-Net's multiplicative
  masks, there is no near-zero-init multiplicative structure here to cause
  a dead-gradient trap).
- **Magnitude collapse**: guarded by construction -- every embedding used in
  the loss (`embed_query`'s and `embed_item_alone`'s outputs) is
  L2-normalized before `torch.cdist`, so a trivial "shrink toward the
  origin" solution isn't available to the optimizer at all.
- **Direction collapse**: found for real, then fixed for real -- see
  `training_log.md` for the exact numbers (mean pairwise candidate-embedding
  cosine similarity climbed to 0.974 after 5 epochs on a small subset
  without the uniformity regularizer; enabling it dropped that to 0.026).
  The regularizer (Wang & Isola 2020, identical formula to phases 13/13b) is
  therefore **on from the start of the real training run**, not something
  discovered partway through the way phase 13's first attempts found it.

## Step 4: local vs. Modal decision

A 1-full-epoch timing smoke test (all 53,306 training outfits, batch size
96, `use_uniformity=True`) measured **~24.5 seconds per epoch** on the
local M4 Air (MPS). A 40-epoch run (phase 13b's own budget) would take
roughly 16 minutes; even a much larger epoch budget stays well within
minutes, not hours. **No Modal GPU spending was needed or used for this
phase** -- local training was clearly practical from the very first
measurement, so escalation was never triggered. See `training_log.md` for
the actual full-run numbers.
