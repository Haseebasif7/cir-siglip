# Phase 4 Notes: DeepFashion In-shop Retrieval Benchmark

## What this phase was for

The professor asked for findings on both DeepFashion and Amazon so the four frozen
encoders could be compared across both. Phase 3 had shown Amazon's `also_buy`/
`also_viewed` ground truth is heavily popularity-skewed and structurally can't evaluate
tail items. DeepFashion's In-shop Clothes Retrieval benchmark gives the opposite:
multiple clean photos of the exact same physical item, no behavioral noise at all. This
phase benchmarks the same four techniques on it and checks whether the Amazon ranking
holds.

## Step 0-1: dataset access and split parsing

Dataset was already present at `week2/phase4_deepfashion_data/` (manually placed, per the
phase brief) -- no MMLab-access blocker. `list_eval_partition.txt` parsed cleanly: 52,712
rows, no malformed rows, all three status values (train/query/gallery) as expected.

- train: 25,882 images, 3,997 unique item_ids
- query: 14,218 images, 3,985 unique item_ids
- gallery: 12,612 images, 3,985 unique item_ids
- **Train item_ids are completely disjoint from query/gallery item_ids** (0 overlap) --
  confirms the official split has no train/test leakage, exactly as expected for a proper
  retrieval benchmark.
- **Query and gallery share the identical set of 3,985 item_ids** (0 query item_ids
  missing a gallery counterpart) -- Recall@K is well-defined for every single query, no
  queries had to be dropped or excluded.
- Extracted only the 26,830 needed query+gallery images from `img.zip` (not the full
  60,820-file archive) -- all found and extracted cleanly, 0 missing from the zip.

No surprises in this step -- the official DeepFashion split is exactly as clean as its
reputation suggests, a real contrast to Amazon's metadata (dead image URLs, popularity
skew, category imbalance all needed handling in earlier phases).

## Step 2: embedding extraction

Reused phase 1b's extraction script structure verbatim (same four techniques, same
batching/device/normalization logic), only changing the id column from `asin` to
`image_name` since DeepFashion needs per-image embeddings (multiple images share an
item_id, unlike Amazon where one product = one image). Ran locally on MPS, all four
techniques in one pass over all 26,830 query+gallery images together:

- ResNet50: (26830, 2048), ~8-9 min
- CLIP ViT-B/32: (26830, 512)
- FashionCLIP: (26830, 512)
- SigLIP: (26830, 768), ~14.5 min (slowest of the four)
- Total: ~35-40 min, no memory issues, no MPS backend errors -- unlike phase 2's
  Grounding DINO, none of these four techniques hit the kind of MPS gaps documented in
  the standing rules.

## Step 3: retrieval evaluation

**Results** (`results_table.md`):

| Technique | Recall@1 | Recall@5 | Recall@10 | Recall@20 |
|---|---|---|---|---|
| ResNet50 | 0.290 | 0.459 | 0.531 | 0.602 |
| CLIP ViT-B/32 | 0.455 | 0.673 | 0.743 | 0.807 |
| FashionCLIP | 0.666 | 0.851 | 0.897 | 0.930 |
| SigLIP | 0.737 | 0.894 | 0.929 | 0.953 |

Recall@K is monotonically non-decreasing in K for every technique (checked
programmatically in the eval script, no warnings triggered) -- confirms the top-K slicing
logic has no off-by-one or ordering bug.

**Ranking: SigLIP > FashionCLIP > CLIP ViT-B/32 > ResNet50 -- identical to Amazon.** Full
comparison and discussion in `comparison_to_amazon.md`.

## Anything unexpected

The ranking holding exactly wasn't a surprise (it was the expected/hoped-for outcome, and
consistent with 3 prior Amazon sample constructions). What *was* a bit unexpected: the
gap between DeepFashion and Amazon scores is much smaller for ResNet50 (+22% relative)
than for the other three techniques (+66% to +82%). Naively one might expect a "cleaner
dataset" to just uniformly inflate every technique's score by a similar margin -- instead,
ResNet50's generic ImageNet features stay weak on both datasets, while SigLIP/
FashionCLIP's real advantage seems to have been partly hidden by Amazon's noisier ground
truth. Worth keeping in mind for the final report: DeepFashion isn't just "the same
comparison but easier," it changes how large the encoder-choice effect looks.

## Required methodology note: Recall@K (DeepFashion) vs. Hit Rate@K (Amazon) are NOT the same measurement

This needs to be explicit for the eventual technical report, since the two numbers look
superficially identical (both are "fraction of queries with at least one same-label item
in the top-K") but were computed under different retrieval setups:

- **Amazon (phases 1, 1b, 2, 3)**: every sampled product in the 1,872-product sample was
  used as a query against *every other sampled product in the same sample* (the diagonal
  -- self-similarity -- was masked out, but every other in-sample product was a valid
  candidate). There was no query/gallery split; the candidate pool for every query was
  "the rest of the sample."
- **DeepFashion (this phase)**: queries and gallery images are two disjoint sets defined
  by the dataset's official split. A query image is compared *only* against the 12,612
  gallery images, never against other query images, never against train images (which
  aren't even embedded). No diagonal masking was needed or performed, since there's no
  overlap between the two sets to begin with -- a query image can never retrieve itself.
- **Practical consequence**: Amazon's candidate pool per query was ~1,871 items (whatever
  the sample size was, minus the query itself); DeepFashion's is fixed at 12,612 gallery
  images. Per the standing rule from phase 1b (`PROJECT_MEMORY.md`), larger candidate
  pools make Hit Rate/Recall@K harder to score well on for a matched difficulty, all else
  equal -- so the fact that DeepFashion scores are *higher* than Amazon's despite a larger
  gallery is entirely attributable to how much cleaner DeepFashion's ground truth is (an
  even larger effect than the candidate-pool-size effect would predict working in the
  other direction).
- **Bottom line for the report**: treat the ranking comparison as solid (same relative
  ordering under two structurally different retrieval setups is a strong result), but
  never present the absolute Amazon HR@5 and DeepFashion Recall@5 numbers as if they were
  measured the same way -- they weren't, and the gap between them reflects both ground
  truth cleanliness AND retrieval-setup differences, not cleanliness alone.
