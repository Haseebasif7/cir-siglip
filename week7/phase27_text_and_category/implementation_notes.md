# Phase 27: Implementation Notes

How text and category inputs are actually integrated into week 6's winning model, and the decisions made along the way. Four of these were proposed and explicitly approved before building anything; a fifth is a necessary mechanical consequence of decision #4, documented here rather than left implicit.

## Decision #1: cascaded text-source fallback, not the brief's literal single-field reading

The brief says to use each item's Polyvore text description, falling back to a zero vector if missing. Checked directly against `polyvore_item_metadata.json` before writing any training code: the `description` field is filled for only **28.7%** of the full catalog (72,152 / 251,008 items) and **31.2%** of items that actually appear in the CIR benchmark. Following the brief literally would have zero-vectored the large majority of items, which would have made this experiment mostly a test of "zero vector concatenated to image" rather than a real test of whether text helps.

Resolution (approved): cascade the text source per item, `description` → `title` → `url_name`, and only fall back to a true zero vector if all three are empty. `url_name` (a cleaned product title, e.g. *"river island green tropical bardot"*) is filled for **100%** of items and is genuinely descriptive text.

Actual source breakdown across all 251,008 catalog items, measured by `00_extract_text_embeddings.py`:

| Source actually used | Count | % of catalog |
|---|---|---|
| description | 72,152 | 28.7% |
| title | 2,700 | 1.1% |
| url_name | 176,156 | 70.2% |
| none (true zero vector) | 0 | 0.0% |

The zero-vector fallback the brief specifies never actually triggers in this dataset once `url_name` is included as a rung — the code path is kept (defensive, for robustness) but is empirically dead here.

## Decision #2: SigLIP text encoder needs `padding="max_length", max_length=64`

SigLIP's text tower was trained with a fixed 64-token canonical sequence length (confirmed via `SiglipConfig.text_config.max_position_embeddings == 64`), unlike CLIP-style models that handle dynamic padding gracefully. Left at HuggingFace's default processor behavior, this produces degraded, off-distribution embeddings. `00_extract_text_embeddings.py` passes `padding="max_length", max_length=64, truncation=True` explicitly on every text encoding call (both the 251,008 item texts and the 11 category-name phrases), matching how SigLIP was actually trained.

## Decision #3: 11 semantic categories, matching the CIR benchmark's own pool structure exactly

No new taxonomy was introduced. `semantic_category` (100% filled across the catalog) already has exactly the same 11 values the CIR benchmark's own candidate pools are organized by: accessories, all-body, bags, bottoms, hats, jewellery, outerwear, scarves, shoes, sunglasses, tops. This fixed, alphabetical order is the index order for both category-conditioning mechanisms (the learned embedding table and the SigLIP-phrase lookup), defined once in `model.py`'s `CATEGORY_LIST` and reused verbatim everywhere else in this phase.

## Decision #4: zero-padded category slot, one shared projection head

Only the query side of the retrieval task has a target category to condition on; candidate/pool items being ranked don't. Resolution (approved): every item's input to the shared projection head gets an extra 768-d category slot — filled with the real target-category vector for the query, and with zeros for every candidate item — so one shared head keeps working for both roles, and query/candidate embeddings stay directly comparable by a plain dot product. This preserves the "otherwise identical to week 6 winner" property: no second head, no architecture asymmetry beyond this one input slot.

**The asymmetry is deliberate and is being put on the record here, not hidden**: the category slot is trained only by query-side gradient signal (positive/anchor pairs), never by candidate-side signal (candidates always see zeros in that slot). This means the projection head's weight columns feeding from the category slot only ever get updated when the query pathway is active, which is by construction, not an oversight.

## Decision #4, mechanism elaboration: pooling order differs between variant 1 and variants 2/3

This wasn't a separate open question posed to the user, but it's a direct, necessary consequence of decision #4 that needs to be on the record.

Week 6's model (and this phase's `text_only` variant, which is meant to be a minimal, directly-comparable widening of it) constructs a multi-item query by projecting every item individually through the shared head first, then mean-pooling the *already-projected* 128-d context-item embeddings. That's `evaluate_recall_gpu` in phase 23/25/26's own Modal apps, copied here unchanged for `use_category="none"`.

Decision #4 requires category to be concatenated onto the query vector *before* it goes through the (single, shared) projection head. That's only satisfiable by pooling the *raw* (pre-projection) context-item vectors first, concatenating the category vector, and making one head call — a different pooling order from variant 1's.

Consequence, stated plainly: `text_only` (variant 1) and the two category-conditioned variants use different query-construction mechanisms, not just different inputs. This means the category-conditioned variants' comparison against `text_only` isn't a perfectly isolated single-variable ablation — it changes two things at once (category presence, and pooling order). Two things make this an acceptable, well-understood cost rather than a fatal confound:

1. **Training itself is completely unaffected.** Every training example's "anchor" is always exactly one item (phase 9/23/25/26's positive-edge pairs, unchanged data source) — pooling one item is that item regardless of order convention, so this asymmetry only ever bites at evaluation time, when real queries have more than one context item.
2. **`text_only` vs. week 6's own reported baseline stays clean.** `text_only` keeps week 6's exact post-projection-pooling mechanism, so that specific comparison (does adding text help, holding everything else fixed) is uncontaminated. The pooling-order variable only enters when comparing `text_only` against the category-conditioned variants, and it's flagged here explicitly so `phase27_notes.md` doesn't read the category-conditioning delta as 100% attributable to category alone.

## Text extraction and category text embedding, briefly

`00_extract_text_embeddings.py`: for every catalog item, picks text per decision #1's cascade, encodes it through SigLIP's text tower per decision #2, saves `data/text_embeddings.npz` (item-ordered, positionally aligned to `siglip_base.npz`). Also encodes the 11 category-name phrases (the bare category string, e.g. `"shoes"`) into `data/category_text_embeddings.npz`, used by the `text_category_siglip_phrase` variant.

`01_build_category_index.py`: a compact, item-ordered `int8` array mapping every catalog item to its category index (decision #3's fixed order), saved as `data/category_index.npz` — used for the category-of-the-target lookup during training (see `results_table.md` / `phase27_notes.md` for how that's used) without a per-batch string dictionary lookup.

## Architecture summary

- **Shared representation**: `base_repr(item) = normalize(concat(image_embedding, text_embedding))`, 1536-d, computed once for the whole catalog, used identically for every item regardless of role.
- **`text_only`**: every item (context, candidate, training anchor/positive/negative alike) is projected individually through one `ProjectionHeadGeneral(in_dim=1536, hidden_dims=[1024], out_dim=128)`. Query = mean of context items' *projected* embeddings (week 6's own mechanism, unchanged).
- **`text_category_learned` / `text_category_siglip_phrase`**: candidates get `base_repr` zero-padded to 2304-d (decision #4). The query pools context items' *raw* `base_repr` vectors, concatenates a 768-d category vector (from a trainable `nn.Embedding(11, 768)` for the "learned" variant, or a frozen SigLIP-encoded category-name phrase for the "siglip_phrase" variant), renormalizes, and makes one projection call through the same shared `ProjectionHeadGeneral(in_dim=2304, ...)`.
- During training, the target category fed to the query/anchor side is the **positive/target item's own `semantic_category`** — exactly mirroring how the CIR benchmark's real queries carry a `category` field for the item being retrieved, not an arbitrary label.

## Text extraction moved from local MPS to Modal

`00_extract_text_embeddings.py` (local) was measured directly rather than assumed: batch size 256/1024/2048 all projected to roughly 55-65 minutes for the full 251,008-item catalog on the M4 Air's MPS backend (the bottleneck is real compute, not a fixable sync/batching inefficiency — CPU was measured too, at ~85 minutes, i.e. worse). This is squarely the "more compute than the local M4 Air can reasonably provide" case the project's standing Modal permission exists for, so extraction was moved to `00_extract_text_embeddings_modal.py`, a T4-backed Modal function with identical cascade/padding logic — same decisions #1/#2, only the execution target changed. The original local script is left in place for reference/provenance of the cascade logic, but wasn't used for the actual run.

## Modal account switch mid-phase, and credit-safety

Partway through this phase, Modal was re-authenticated to a different account (`m-haseebasif5`, a separate workspace from `muhamasif`, which every prior phase's deployed apps and data volume lived under). Rather than switch back, the decision was to continue on the new account: a fresh volume (`phase27-text-category-data`) was created and the ~1GB of reused shared data (image embeddings, positive edges, val/test benchmarks, item metadata) was re-uploaded to it, and `phase27-text-category`'s app was redeployed under the new workspace.

Since a fresh account's credit budget is unknown, three resilience measures were added to `modal_app.py` before running any real training, not after a failure:

1. **Immediate checkpoint persistence on every improvement**, not batched until the run ends — `torch.save` + `volume.commit()` fire the moment validation Recall@10 improves, so a container killed mid-run mid-epoch never loses more than the epochs since its last improvement.
2. **Idempotent skip**: each run writes a `status_<name>.json` to the volume; a run already marked `"done"` is detected at the very start of `train_one` and returned immediately without spending any more compute, making it safe to simply re-invoke the same config after an interruption.
3. **Warm-start resume**: if a checkpoint exists from an interrupted earlier attempt at the same run name, its weights (and category table, where applicable) are loaded as the starting point instead of a random init. This is a warm start, not a byte-exact resume — optimizer state, epoch counter, and patience counter are not restored — but the improvement-only-saves logic means further training can only match or improve on the warm-started weights, never regress below what was already banked.

`02_train_variants.py` mirrors this on the orchestration side: variants are launched one at a time (not via `.map()`), each variant's result is written to disk and its checkpoint downloaded immediately after that specific run finishes, and re-running the script skips any variant that already has a local result and checkpoint.

## Hyperparameters (unchanged from week 6, per the brief)

`lr=0.001, batch_size=256, weight_decay=0.0, tau=0.15, r_neg=8`, early stopping on validation-benchmark Recall@10 (never validation loss), `hidden_dims=[1024], out_dim=128` matching week 6's winning architecture. Single-seed runs, per the brief — an initial signal check, not a final result.
