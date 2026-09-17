# Phase 31: Architecture and Implementation Notes

## What changed from phase 14b, and what didn't

The transformer architecture itself (`OutfitTransformerSigLIP`: shared trunk projection, multi-head
self-attention set encoder, learnable outfit token, `embed_query`/`embed_item_alone` split) is byte-for-
byte unchanged from `week4/phase14b_outfittransformer_category_negatives/scripts/model.py` -- every
constructor parameter (`siglip_dim`, `d_model`, `d_embed`, `n_heads`, `n_layers`, `d_ffn`, `dropout`) was
already exposed there, so text integration (step 1) needed zero architecture code changes, only a
different `in_dim` at instantiation. The category-restricted-random negative-sampling *scheme* (phase
14b's own adopted fix) is also unchanged -- only how it's computed changed (see below). Everything that
changed is in `modal_app.py`'s `train_one`, not in the model class.

## 1. Text input (`in_dim`)

`self.proj = nn.Linear(siglip_dim, d_model)`, `siglip_dim` set to 768 (image-only) or 1536
(`normalize(concat(image_768, text_768))`, identical construction to phase 27/28/30). See
`text_input_integration.md` for the parameter delta and isolated contribution.

## 2. Checkpoint selection (`selection_metric`)

Added a `selection_metric` config flag (`"recall10"` default, `"val_loss"` for the one A1 fidelity-check
run only) and a real GPU-resident `evaluate_recall_gpu` (adapted from phase 14b's own
`04b_cir_eval_random_negatives.py` `embed_query`/`embed_item_alone` split), evaluated every epoch against
`cir_val_benchmark.json`. See `checkpoint_selection_check.md` for the full diagnosis, including the
discovery that patience also needed recalibrating (8 -> 25) once the selection metric changed, because
Recall@10 has a genuine ~17-epoch noisy plateau early in training that val_loss's much smoother curve
never exposed.

## 3. Vectorized negative sampler

Phase 14b's `TrainState._sample_negatives` (random mode) built `extra_pool = [i for i in pool[category] if
i not in exclude]` per training sample -- an O(pool size) Python list comprehension over a mean ~36,500-
item category pool (measured ~56s/epoch of pure sampling overhead in phase 14b's own
`training_log.md`). Replaced with an index-space rejection sampler: a numpy int64 array is precomputed
once per category (built from the RAW, duplicated per-category item list, since these lists are genuine
multisets -- an item appears once per outfit containing it, e.g. "shoes": 51,132 entries, 36,481 unique --
preserving the original's implicit outfit-occurrence-weighted sampling exactly), and each sample draws a
small batch of candidate positions via `rng.integers`, rejecting only against the tiny (~3-8 item) exclude
set.

Verified statistically equivalent, not just fast (`00a_sampler_equivalence_check.py`): 200,000 draws under
each implementation, zero exclude-set violations in either, and a two-sample chi-square contingency test
(NOT a naive one-sample `chisquare` against a treated-as-fixed reference distribution, which would
understate the true null variance and produce spurious failures -- caught and fixed during this check's own
development) gives stat/dof=1.005, p=0.265 -- statistically indistinguishable. ~103x measured speedup
(1.199ms/draw -> 0.012ms/draw). One minor, deliberate, documented deviation: this sampler suppresses
within-call duplicate canonical items (the original's position-based `random.sample` could, at very low
probability, draw the same item's value twice via two of its duplicate positions); negligible at this pool
scale and arguably more correct for a triplet loss's negative set.

## 4. Seed plumbing

Phase 14b never called `torch.manual_seed`, and its `TrainState` was hardcoded to `seed=0` -- its single
reported run was not reproducible. Fixed: `torch.manual_seed(cfg["seed"])` before model construction,
`TrainState(seed=cfg["seed"])` for data order and negative draws.

## 5. Benchmark discipline

`train_one` never loads `cir_test_benchmark.json` -- structurally enforced (verified by grep, not just
convention). `cir_val_benchmark.json` (selection, every step) and `cir_train_benchmark.json` (overfitting
diagnostic, reused unchanged from `week4/phase25_scale/`, built from the same Polyvore nondisjoint split
`training_data.json` uses) are the only benchmarks `train_one` ever opens. The test benchmark is opened
exactly once, by `07_final_test_eval.py`, at the very end.

## 6. What was NOT changed, and why (scale axes)

Step 4 (architectural scale testing: `n_heads`, `n_layers`, `d_ffn`, `d_model`, `d_embed`) was designed
(`scripts/04_scale_sweep.py`, 30 configs across 5 axes x 2 sizes x 3-point LR recheck) but not run to
completion -- a budget constraint stopped it immediately after launch (see `scale_sweep.md`). The final
reported model therefore uses phase 14b's original architecture shape throughout
(`d_model=128, d_embed=64, n_heads=8, n_layers=4, d_ffn=512`) -- only the training procedure (text input,
checkpoint selection, hyperparameters, loss shape) differs from phase 14b's own configuration.
