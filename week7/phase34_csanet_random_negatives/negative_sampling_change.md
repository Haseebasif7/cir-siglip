# Phase 34, Step 1: The Negative-Sampling Change, and Confirmation Nothing Else Did

## What changed

`scripts/train_core.py` is phase 33's own `train_core.py`, copied and edited in exactly one place:
`TrainState._sample_negatives`.

**Phase 33 (mined, unchanged since phase 13):**
```python
def _sample_negatives(self, positive_id, category, exclude, num_negatives):
    cands = self.neg_candidates.get(positive_id, [])
    cands = [c for c in cands if c not in exclude]
    if len(cands) >= num_negatives:
        return self.rng.sample(cands, num_negatives)
    extra_pool = [i for i in self.items_by_cat[category] if i not in exclude and i not in cands]
    n_extra = num_negatives - len(cands)
    extra = self.rng.sample(extra_pool, min(n_extra, len(extra_pool))) if extra_pool else []
    return cands + extra
```
Draws from `week4/phase13_csa_net_baseline/data/negative_candidates.json` -- SigLIP-nearest-neighbor,
same-category hard negatives mined once, offline, in phase 13 -- and only falls back to random same-category
items when the mined list runs short.

**Phase 34 (random, this phase's only change):**
```python
def _sample_negatives(self, positive_id, category, exclude, num_negatives):
    pool = [i for i in self.items_by_cat[category] if i not in exclude]
    if not pool:
        return []
    return self.rng.sample(pool, min(num_negatives, len(pool)))
```
Draws `num_negatives` uniform-random items from the positive's own same-category pool, full stop. No file
read, no mined candidates involved at all. This is exactly the category-restricted, uniform-random scheme
phase 14b's run 2 adopted for OutfitTransformer, and what this project's own model has used since phases 7-9.

A consequence of this change (not a separate edit): `load_catalog()` no longer takes a
`negative_candidates_path` argument, and `TrainState.__init__` no longer takes a `neg_candidates` argument --
both simply have nothing left to do with that file. `week4/phase13_csa_net_baseline/data/negative_candidates.json`
is not read anywhere in this phase's code.

Negatives are already redrawn fresh every epoch under this scheme, exactly as the brief requires -- this was
already true in phase 33 too (`make_sample` is called once per outfit per epoch, inside the per-epoch
training loop in `run_training`), not a new property introduced here.

## Confirmation nothing else changed

Everything else in `train_core.py` is a byte-for-byte copy of phase 33's file:
- `model.py`: copied unchanged (diffed below).
- Loss function shape: triplet margin (`MARGIN = 0.3`) with min-aggregation over negatives, plus the
  uniformity regularizer (`UNIFORMITY_WEIGHT = 1.0`) -- the exact values phase 33 actually trained with
  (`phase33/scripts/modal_app.py`'s `DEFAULT_CONFIG` and `04_train_seed.py`'s `WINNING_CONFIG`, neither of
  which override `margin`/`uniformity_weight` from `train_core.py`'s own module-level defaults). The phase
  34 brief's own parenthetical aside ("uniformity weight at 0.1") does not match anything CSA-Net was ever
  tuned to in phase 33 -- 0.1 was phase 31's OutfitTransformer tuning-grid winner (a different architecture,
  different loss surface). This document preserves phase 33's actual CSA-Net value (1.0) exactly, per the
  brief's own primary instruction that "everything else stays identical" to phase 33's winning configuration.
- Text integration: image+text cascaded concat (`base_repr = normalize(cat([image_768, text_768]))`),
  identical construction.
- Checkpoint-selection criterion: validation Recall@10, `patience=5`, `min_delta=0.0005` -- phase 33's step
  1 finding (CSA-Net needs no patience recalibration) is inherited, not re-derived here.
- Hyperparameters for the single-seed gate (step 2): `lr=1e-4, batch_size=48, max_epochs=40` -- phase 33's
  own tuning-grid winner (`phase33/scripts/03_tuning_grid.py`), reused exactly, not re-tuned. Re-tuning
  under a different negative-sampling scheme is explicitly out of scope for this phase (the brief's own
  "single variable test" framing) -- an interesting but separate question for any future phase, none of
  which are planned.
- `compute_batch_loss`, `evaluate_recall`, `evaluate_recall_ensemble`: byte-identical. The category-sharing
  assumption they rely on (negatives share the positive's category) still holds under random sampling,
  since negatives are still drawn from `items_by_cat[category]` where `category` is the positive's own
  category.

```bash
$ diff week7/phase33_investment_parity_csanet/scripts/model.py week7/phase34_csanet_random_negatives/scripts/model.py
# (no output -- byte-identical)
```

## Data files this phase does NOT touch

- `week4/phase13_csa_net_baseline/data/negative_candidates.json` -- never opened.
- `week4/` folder -- untouched (`git diff --stat week4/` empty, confirmed at phase completion).
