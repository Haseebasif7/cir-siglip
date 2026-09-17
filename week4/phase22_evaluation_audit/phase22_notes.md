# Phase 22: Evaluation Audit -- Verdict

## What this phase was for

Phase 9's plain projection has come out as the strongest configuration in every comparison this project has run, by a wide margin (roughly 2x the next-best "mine" configuration, CSA-Net's reproduction, and 1.4-1.5x OutfitTransformer's own published numbers -- see `week4/phase20_full_consolidation/final_comparison_table.md`). Before trusting that result enough to build further work on top of it, this phase checked whether it's a genuinely fair result or an artifact of how the benchmark was built or how different methods were scored on it. Five checks were run, each with its own output file; this is the honest verdict across all five.

## Step 1: train/test item overlap -- `overlap_check.md`

**Real overlap exists, but it is exactly symmetric across all three models.** 34.8% of the CIR benchmark's item universe (16,434 of 47,220 unique items) was also seen during training by phase 9, CSA-Net (phase 13b), and OutfitTransformer (phase 14b run 2) alike -- this is a direct, structural consequence of Polyvore's official "nondisjoint" train/test split, which explicitly permits the same physical item to appear in different outfit groupings across splits. This was quantified directly, not assumed: phase 9's own item extraction and phase 13's independent re-derivation of the training item universe from the same two source files (`train.json`/`valid.json`) agree exactly (Jaccard = 1.0000, both produce the identical 220,455-item universe), and the specific set of benchmark items that overlaps each model's training data is identical between the two pipelines (16,434 items in both, 0 items unique to either side). All three models were also trained with early stopping on the same underlying validation split, ruling out one model exploiting the shared overlap more aggressively via more epochs or extra sampling.

**Interpretation**: this overlap means the absolute Recall@K numbers reported across this project are not strictly "zero-shot on unseen items" numbers -- a real caveat on how those absolute numbers should be read. But since the overlap is identical in size and identical in which items it covers across all three models being compared, it cannot explain why phase 9 outperforms the other two -- whatever advantage this overlap grants, it grants equally to every configuration in the comparison table.

## Step 2: benchmark construction consistency -- `benchmark_consistency_check.md`

**Confirmed identical, not assumed.** The actual `BENCHMARK_JSON` path each evaluation script resolves to (phase 9's verification via phase 20, CSA-Net's phase 13b evaluator, OutfitTransformer's phase 14b run-2 evaluator) was extracted directly from each script's own code and hashed. All three resolve to the exact same file on disk, byte-for-byte identical SHA-256 hash. There is no possibility of a quietly-diverged reconstructed copy. The rank/hit-counting convention (count of candidates at least as good as the target, `>=` on similarity or `<=` on distance) was also confirmed identical across all three evaluators by directly reading each script's rank-computation lines.

## Step 3: scoring asymmetry between context-free and context-aware models -- `scoring_asymmetry_check.md`

**No asymmetry found.** Phase 9's own query representation is itself a mean-pool over context items (not a truly "single fixed" query, contrary to how the concern is sometimes framed) -- so context-length dependence exists for every model here, not uniquely for CSA-Net/OutfitTransformer; the real question was whether the context-aware models' own aggregation introduces an *additional*, asymmetric penalty. Checked directly:

- **OutfitTransformer** uses a real self-attention set-encoder with a learned "outfit token" readout and a proper padding mask (`embed_query` in `model.py`) -- not a naive average, and not diluted by padding. After that, scoring against the pool is the identical single dot-product ranking phase 9 uses.
- **CSA-Net** implements the paper's own eq. 5 (average per-context-item pairwise distance), explicitly divided by the query's own actual context length -- a scale-normalizing step, not a length-dependent penalty, and documented in the code as a deliberate, necessary adaptation of the paper's own mechanism, not a loosening of the comparison.
- A numeric follow-up was actually run (not just argued from code): Recall@10 was bucketed by query context length (1-3, 4-5, 6-7, 8-16 items) for phase 9 and OutfitTransformer, re-running both models' real scoring logic per bucket. Phase 9's advantage ratio over OutfitTransformer stayed in a narrow band (2.01x-2.36x) across every bucket, with no monotonic growth as context length increased -- the signature of a genuine capability gap between the two models, not an aggregation artifact. CSA-Net was not included in this specific numeric run (its evaluator is materially more expensive to re-run stratified) but its length-normalized aggregation formula was verified directly in code.

## Step 4: duplicate/near-duplicate images in candidate pools -- `duplicate_check.md`

**Negligible.** Checked directly against the actual image files and raw SigLIP embeddings, not assumed carried over from the (unrelated, Amazon-dataset) dedup work in phases 6-8:

- 0 of 29,681 queries (0.000%) have a target item that is byte-identical to one of that query's own context items -- the most damaging possible leakage form does not occur at all in this benchmark.
- 0 of 29,681 queries have a target item with raw-SigLIP cosine similarity > 0.995 to a context item either (the near-duplicate superset of the byte-identical check).
- 15 groups of byte-identical images exist among the 26,494 unique pool items (30 items total, 0.11% of pool items), and 44 near-duplicate pairs (cosine > 0.995) exist within pools overall -- present, but far too small in absolute terms to explain a roughly 2x recall gap between configurations.

## Step 5: manual, by-hand reproduction -- `manual_reproduction_check.md`

Three randomly chosen queries were traced by hand end-to-end for both phase 9 and OutfitTransformer, computing rank directly from raw similarity scores rather than calling the shared `evaluate_recall()` function at all. Separately, phase 9's own `evaluate_recall()` was re-run against the full benchmark using the same loaded checkpoint and embeddings this manual script used, reproducing the cited `0.1317/0.2464/0.3216` to 4 decimal places (`0.13167/0.24639/0.32155`) -- direct empirical confirmation, not just a code-reading argument, that the batched harness computes exactly what the manual per-query arithmetic computes.

## Step 6: the overall verdict

**Everything checked out clean.** No check in this audit turned up an issue that would call the current comparison into question:

- The one real structural fact found -- 34.8% train/test item overlap -- is a known, documented property of Polyvore's official split, and is exactly symmetric across all three models compared, so it cannot explain phase 9's advantage over the others (it advantages everyone equally, or no one relative to each other).
- Benchmark construction is verified byte-identical across every model's evaluation.
- No scoring-code asymmetry was found between the context-free and context-aware models; OutfitTransformer's learned attention aggregation and CSA-Net's paper-faithful, length-normalized aggregation were checked directly in code and (for OutfitTransformer) confirmed numerically flat across query context lengths.
- Duplicate/near-duplicate leakage inside candidate pools is present at a negligible rate (0.000% target/context leakage, <0.2% near-duplicate pool items) -- far too small to account for the magnitude of the gap between configurations.
- Manual, by-hand reproduction of individual queries matches the harness's own reported numbers exactly.

**This becomes the confirmed, trustworthy foundation for further work.** Phase 9's plain projection is not winning because of a benchmark artifact, a scoring bug, or unequal access to leaked test items -- it is winning on its own merits under a benchmark that was checked, directly, at every step this audit specified. The one caveat worth carrying forward explicitly (not because it changes the comparison, but because it affects how absolute Recall@K numbers should be described in any future writeup) is that these numbers are not strictly cold-start/zero-shot numbers, given the confirmed 34.8% train/test item overlap inherited from Polyvore's own official split design.

## What this means for what comes next

Per this phase's own brief ("do not attempt any model improvements... until this audit is complete and resolved") and per phase 21's own closing note (nothing further is queued in the professor's seven-point redirect sequence), this audit's clean result removes the last blocking concern on trusting phase 9's number. A fresh session should treat phase 9's confirmed `0.1317/0.2464/0.3216` (and the full `final_comparison_table.md` ranking) as settled and ask the user what's next -- the long-deferred paper writeup, or a genuinely new direction -- rather than assume another CIR-baseline variant or another audit pass is wanted.
