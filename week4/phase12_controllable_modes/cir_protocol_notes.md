# Phase 12, Step 1: Building the CIR Evaluation Harness

## What the brief asked for

Look at `github.com/bigohofone/outfit-transformer` to understand how it implements the
"refined" Complementary Item Retrieval (CIR) protocol (the one that fixes a data-leakage
issue present in the original CSA-Net/OutfitTransformer setup), reuse that same protocol
for this project's own evaluation, and if the linked repo's pretrained checkpoint can be
run under this same evaluation, do so for a directly-measured comparison number.

## What was actually found investigating the repo

The repo exists and is real (confirmed reachable at both `bigohofone/outfit-transformer`
and `owj0421/outfit-transformer` -- these resolve to the same content; the docstring
inside `src/data/datasets/polyvore.py` credits "Wonjun Oh, owj0421@naver.com", and the
repo's own Hugging Face dataset dependencies are all under the `owj0421/` namespace, so
`owj0421` appears to be the actual author and `bigohofone` a fork/rename of the same
repo). Its README confirms it is a non-official OutfitTransformer (CVPR 2023)
reimplementation, reports Compatibility-Prediction AUC and FITB accuracy numbers, and
provides a downloadable pretrained checkpoint via `gdown`.

Three concrete things did **not** match what the brief described, checked directly
against the repo's source rather than assumed:

1. **No candidate-pool retrieval evaluation exists in the repo's accessible source.**
   The file that sounds like the CIR eval (`src/run/3_test_complementary.py`) calls
   `compute_cir_scores()` from `src/evaluation/metrics.py`, but that function is just
   4-way fill-in-the-blank accuracy (`torch.mean((predictions == labels).float())`) --
   the same task phase 9 already evaluates as "FITB", not a large-candidate-pool
   Recall@10/30/50 retrieval task. No file under `src/data`, `src/evaluation`, or
   `src/run` builds a per-category candidate pool of ~3,000 items or computes Recall@K
   against one. `compute_cir_scores` appears to be a naming choice (FITB accuracy
   labelled "cir" internally), not evidence the actual CSA-Net-style retrieval protocol
   is implemented here.
2. **The repo's own "eliminates data leakage" claim is about model inputs, not
   evaluation methodology.** Its README describes this as replacing item
   descriptions/categories fed into the *model* with learnable embeddings, so the model
   can't shortcut on text metadata -- an architecture change, not a fix to how the
   candidate pool or query set is constructed for evaluation.
3. **Its checkpoint is not a drop-in comparison for this project's harness.** It uses
   CLIP embeddings (not SigLIP), the official OutfitTransformer transformer-based scorer
   (not this project's frozen-embedding + small-MLP setup), and its dataset pipeline
   pulls `owj0421/polyvore-outfits` from the Hugging Face Hub in a different record
   format from the parquet-packed images this project already downloaded and decoded in
   phase 9 (`dataset_structure_check.md`). Running it under this project's own harness
   would mean reimplementing a large fraction of their inference pipeline for a repo that
   doesn't itself implement the eval we need.

**Given the brief's own fallback instruction** ("if the linked repo's checkpoint or code
doesn't run cleanly, report exactly what happened rather than spending excessive time
debugging someone else's repository") -- this phase did not attempt to clone, install, and
run the repo's checkpoint. The literature numbers quoted in the brief (Recall@10=9.58%,
Recall@30=17.96%, Recall@50=21.98%) are carried forward as-given, attributed to the
OutfitTransformer paper itself (matching the brief's own framing, "OutfitTransformer's
reported numbers on this task"), the same way phase 9 carried forward Vasileva et al.'s
published AUC/FITB numbers as a literature anchor without re-deriving them --
**this number is not independently re-verified by this phase**, and the repo investigated
here does not provide a way to reproduce it directly.

## The protocol actually implemented here

Since no usable reference implementation of the candidate-pool retrieval task was found,
this phase implements the standard CSA-Net/OutfitTransformer-style CIR protocol directly,
built to be leakage-safe by construction rather than by copying an unverified fix:

1. **Queries**: leave-one-out over every item slot in every Polyvore `nondisjoint` test
   outfit (`test.json`, 10,000 outfits, 53,506 item slots). For a given outfit, each item
   with a downloaded image and known `semantic_category` becomes a target exactly once;
   the query representation is the mean embedding of the outfit's *other* items (also
   filtered to known-category, image-present items). Outfits where fewer than 2 items
   qualify are skipped (no query set would exist) -- checked directly: all 10,000 test
   outfits have every item image-downloaded and categorized (see counts below), so this
   filter removes only single-item outfits, if any.
2. **Candidate pools, built once per category, independent of any specific query.** For
   each of Polyvore's 11 `semantic_category` values, a background pool is sampled once
   (seeded, `random.Random(42)`) from all *test-split* items of that category: the full
   set if the category has &le;3,000 test items, otherwise a random 3,000. This pool does
   not depend on outfit membership or on which item is being queried -- nothing about a
   specific query's ground truth informs which distractors are in its pool.
3. **Queries whose target missed the fixed sample are dropped, not force-added back
   in.** An earlier version of this script force-added a query's true target into its
   category's pool whenever it was missing, matching what the brief's "3,000 images per
   category" framing suggests. That approach turned out to defeat the cap entirely:
   almost every item in a popular category (e.g. shoes) is *someone's* leave-one-out
   target somewhere in the test set, so force-inclusion re-inflated the "capped" pool
   back to near the full category size (shoes: 3,000 -> 8,551, confirmed directly by
   running it before switching approach). The fix kept here: sample the fixed pool
   first, then only keep queries whose target happens to already be inside that fixed
   sample; queries whose target missed the sample are excluded from the benchmark
   rather than growing the pool. This keeps every category's pool genuinely capped and
   keeps the pool identical across every query of that category (a real practical
   advantage: one similarity matrix per category, not a custom pool per query), at the
   cost of evaluating roughly `3000/N_category` of a popular category's possible queries
   -- reported per-category in the coverage table below, and not hidden.
4. **No leakage across splits**: both the queries and every candidate pool are built
   exclusively from `test.json` and test-split item ids -- nothing from
   `train.json`/`valid.json` enters either side. This mirrors this project's own standing
   rule (phase 7's explicit train/eval disjointness check) rather than depending on an
   external, unverified claim about what "the" refined protocol fixes.
5. **Scoring and metric**: rank each query's candidate pool by cosine similarity between
   the query representation and each candidate's embedding (raw SigLIP or a trained
   projection, whichever configuration is being evaluated); Recall@K = fraction of
   queries where the true target lands in the top K.

## Test-split coverage check (done before building anything, per this project's
convention of verifying rather than assuming)

- 10,000/10,000 test outfits have &ge;2 items with both a downloaded image and a known
  `semantic_category` -- no outfit was excluded for missing data.
- 47,854 unique items appear across the test split; all 47,854 have both an image and a
  known `semantic_category` (0 unknown-type test items) -- unlike Amazon's `categories`
  breadcrumb (phase 8), Polyvore's `semantic_category` field has no missing/unreliable
  cases in this split.
- 23,825 of the 53,506 possible leave-one-out item-slots were dropped because their
  target item fell outside their category's fixed 3,000-item sample (point 3 above) --
  this only affects the 7 categories with more than 3,000 unique test items. **29,681
  leave-one-out queries survive into the final benchmark.**
- Per-category counts (full numbers in `data/cir_benchmark_coverage.md`, generated by
  the harness script itself, not hand-maintained here):

| `semantic_category` | test items | queries kept | pool size used |
|---|---|---|---|
| bags | 7,783 | 3,382 | 3,000 (capped) |
| shoes | 8,551 | 3,315 | 3,000 (capped) |
| jewellery | 7,937 | 3,352 | 3,000 (capped) |
| all-body | 3,254 | 3,182 | 3,000 (capped) |
| tops | 6,143 | 3,263 | 3,000 (capped) |
| bottoms | 5,480 | 3,361 | 3,000 (capped) |
| outerwear | 3,212 | 3,244 | 3,000 (capped) |
| sunglasses | 2,202 | 2,906 | 2,202 (all) |
| accessories | 1,254 | 1,361 | 1,254 (all) |
| hats | 1,194 | 1,401 | 1,194 (all) |
| scarves | 844 | 914 | 844 (all) |

(Queries kept can exceed test-item count for uncapped categories because every
qualifying occurrence of an item across multiple outfits is a separate query, while
"test items" counts unique items once.)

Implementation: `scripts/01_build_cir_benchmark.py` (builds and saves
`data/cir_benchmark.json` -- pools, queries, target categories) and a shared
`recall_at_k()` evaluator reused by every subsequent step's evaluation script, so every
configuration in this phase (raw SigLIP, phase 9's model, and this phase's own
substitute/complement/blend modes) is scored under the exact same queries and candidate
pools.
