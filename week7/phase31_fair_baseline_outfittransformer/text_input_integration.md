# Phase 31, Step 1: Text Input Integration

## The change

`OutfitTransformerSigLIP`'s constructor already exposed `siglip_dim` as a parameter
(`week4/phase14b_outfittransformer_category_negatives/scripts/model.py:32`), so no architecture
code changed at all. The only change is what gets passed as input:

- `input_mode="image"`: `base_repr = normalize(image_768)` (768-d), phase 14b's original input.
- `input_mode="image_text"`: `base_repr = normalize(concat(image_768, text_768))` (1536-d), identical
  construction to phase 27/28/30's own text integration -- text embeddings reused directly from
  `week7/phase27_text_and_category/data/text_embeddings.npz` (cascaded fallback: description, else
  title, else url_name; SigLIP text encoder, `padding="max_length"`, `max_length=64`), no re-extraction.

`base_repr` feeds into `encode_item_tokens` (the single shared entry point for every role: context items,
the positive target, all 10 category-restricted negatives, and every eval-time candidate/query-context
item), so text enters through the exact same pathway for every role, per the brief's requirement.

## Parameter delta

`self.proj = nn.Linear(siglip_dim, d_model)`: `768*128+128 = 98,432` params (image-only) vs.
`1536*128+128 = 196,736` params (image+text) -- a delta of 98,304 params. Everything else in the
architecture is unchanged. Confirmed directly from the trained checkpoints:
`ot31_repro_recall_p25.n_params = 899,840` (image), `ot31_text_recall_p25.n_params = 998,144`
(image+text) -- difference = 98,304, exactly matching the arithmetic above.

## Verification: trains without error (smoke test)

`modal run scripts/modal_app.py::smoke_test` -- 2 epochs each, `image` and `image_text`:

| input_mode | epoch 0 val R@10 | epoch 1 val R@10 | measured s/epoch (incl. Modal overhead) |
|---|---|---|---|
| image | 0.0148 | 0.0152 | 45.7 |
| image_text | 0.0151 | 0.0148 | 31.8 |

No errors in either mode; `text_embeddings.npz`/`siglip_base.npz` `item_ids` alignment was also verified
directly (`00_verify_volume_data.py`, 251,008 items, exact match) before any training run. Cleared to
proceed to the real measurement runs (step 2).

## Text's isolated contribution (measured in step 2, reported here per the brief's step-1 scope)

Once checkpoint selection was also fixed and correctly patience-calibrated (see
`checkpoint_selection_check.md`), text's isolated contribution on top of the selection fix:

| Configuration | Val Recall@10 |
|---|---|
| A2-corrected: image only, recall10 selection, patience=25 | 0.0786 |
| A3-corrected: image+text, recall10 selection, patience=25 | 0.0890 |

+13.2% relative from adding text alone, holding everything else (selection criterion, patience,
hyperparameters) fixed. This is a real, isolated, single-variable gain -- consistent in direction (if not
yet in scale, since the base architecture differs) with phase 27/28's own text finding for the project's
own model.
