# Phase 22, Step 1: Train/Test Item Overlap Against the CIR Benchmark

Polyvore's official 'nondisjoint' split explicitly allows the same physical item to appear in both train and test outfits (as part of different outfit groupings) -- this is a known, documented property of the split, not a leakage bug to be fixed. What matters for a fair comparison is whether this overlap is roughly symmetric across the models being compared, since all three were trained on the identical official train/valid split files.

**CIR benchmark item universe** (from `week4/phase12_controllable_modes/data/cir_benchmark.json`): 47220 unique items appear in the benchmark (queries' target items, queries' own context items, and/or category candidate pools). 26494 unique items sit in some candidate pool; 26494 unique items are ever a query's target.

## Sanity check: do phase 9's own item extraction and phase 13's independent re-derivation agree on the training item universe?

Both pipelines start from the same two files (`polyvore_raw/nondisjoint/train.json` + `valid.json`) and apply the same qualifying filter (downloaded image + known `semantic_category`), but via two separately written scripts (`week3/.../02_build_training_pairs.py` vs `week4/phase13_csa_net_baseline/scripts/01_prepare_training_data.py`).

- Phase 9's train+val item universe: 220455 unique items
- Phase 13's train+val item universe: 220455 unique items
- Overlap: 220455 items shared, 220455 in the union -- Jaccard = 1.0000

**Confirmed: the two independent extractions agree almost exactly.** This rules out a subtle difference in how the two pipelines read the official split (e.g. a different image-validity filter) as a source of asymmetric overlap below -- both models are drawing from essentially the same training item pool.

## Overlap between each model's training items and the CIR benchmark's item universe

| Model | Train items seen | Val items seen | Train ∩ benchmark | % of benchmark items | Val ∩ benchmark | % of benchmark items |
|---|---|---|---|---|---|---|
| Phase 9 (ProjectionHead) | 204679 | 25132 | 16434 | 34.8% | 3664 | 7.8% |
| Phase 13/13b (CSA-Net) & Phase 14/14b (OutfitTransformer) -- shared training_data.json | 204679 | 25132 | 16434 | 34.8% | 3664 | 7.8% |

## Direct symmetry check: is the overlap the same set of items for both training pipelines?

- Benchmark items overlapping phase 9's train set: 16434
- Benchmark items overlapping phase 13's (CSA-Net/OutfitTransformer) train set: 16434
- In both: 16434
- Only in phase 9's overlap: 0
- Only in phase 13's overlap: 0

**The overlapping item sets are nearly identical between the two training pipelines.** Since phase 9, CSA-Net (phase 13b), and OutfitTransformer (phase 14/14b) all draw from the same official train/valid split, and the resulting benchmark-overlap sets match this closely, no model has a structural advantage over the others purely from which items leaked across the split -- the leakage, where it exists, is shared.

## Training-procedure differences beyond raw item overlap

Even with a symmetric item-level overlap, one model could still benefit more from the shared overlap if it trains for materially more epochs/passes over the data, or samples the overlapping items more aggressively than the others. Checked directly against each phase's own training logs/notes:

- **Phase 9**: `05_train_projection.py`, MAX_EPOCHS=100 with early stopping (PATIENCE=5, MIN_DELTA=1e-4) on val loss -- converged well short of 100 in every run this project has logged (see phase 9's own training curves).
- **Phase 13b (CSA-Net)**: `01_train_full.py`, max_epochs=40 with early stopping (patience=5) on the *same* held-out val_outfits split as phase 9's val split source (both derive from `valid.json`).
- **Phase 14b run 2 (OutfitTransformer)**: `02b_train_full_random_negatives.py`, same `training_data.json` train/val split, same early-stopping discipline.

All three use early stopping keyed to validation loss on the same underlying val_outfits pool (from `valid.json`), rather than a fixed epoch budget that could let one model see more repetitions of overlapping items than another. No model in this comparison was trained with an unusually large epoch count, extra data augmentation, or an oversampling scheme that would let it exploit shared train/test overlap more than the others.

