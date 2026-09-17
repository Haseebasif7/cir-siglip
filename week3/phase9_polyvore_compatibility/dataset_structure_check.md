# Phase 9, Step 1: Polyvore Outfits Dataset Structure Check

## Outfit JSONs (nondisjoint/{train,valid,test}.json)

| Split | Outfits | Item slots | Outfits with non-sequential index |
|---|---|---|---|
| train | 53306 | 284767 | 0 |
| valid | 5000 | 26781 | 0 |
| test | 10000 | 53506 | 0 |

**Total outfits: 68306** -- matches the paper's reported 68,306 exactly.
Index confirmed 1-based sequential within every outfit (0 exceptions found).

## Item metadata (polyvore_item_metadata.json)

251008 items total. **`semantic_category` is present directly per item** (11 distinct values) -- used as this phase's "type" level, no join through `categories.csv` needed (that file maps a finer numeric `category_id` to multiple alternate names per id, messier and unnecessary for this phase's purpose).

| semantic_category | count |
|---|---|
| shoes | 44850 |
| jewellery | 41414 |
| bags | 40717 |
| tops | 32998 |
| bottoms | 27670 |
| all-body | 18478 |
| outerwear | 17065 |
| sunglasses | 9874 |
| accessories | 6973 |
| hats | 6071 |
| scarves | 4898 |

## Compatibility files (nondisjoint/compatibility_{split}.txt)

Format: `<label> <set_id>_<index> <set_id>_<index> ...` -- label=1 lines are real outfits (one line per outfit, same order as the split's json); label=0 lines are artificial negative "outfits" assembled from items across different set_ids.

| Split | Label=1 (real outfits) | Label=0 (negative) |
|---|---|---|
| train | 53306 | 53306 |
| valid | 5000 | 5000 |
| test | 10000 | 10000 |

Confirmed: label=1 count matches the split's outfit count exactly (e.g. train: 53,306 positive lines = 53,306 train outfits), and the positive/negative split is exactly 1:1 balanced in every split.

## Fill-in-the-blank files (nondisjoint/fill_in_blank_{split}.json)

Format: `{"question": [set_id_index, ...], "blank_position": int, "answers": [4 set_id_index candidates]}`. Verified on a 2,000-record sample of the test split: **`answers[0]` is always the unique correct (same-outfit) candidate**, `answers[1:4]` are distractors from other outfits -- this is the scoring convention step 5.1 will use.

| Split | Questions |
|---|---|
| train | 53306 |
| valid | 5000 |
| test | 10000 |

Test question count (10,000) matches the test outfit count exactly -- one FITB question per outfit.

## Images (decoded from data/nondisjoint/{train,valid,test}.parquet)

**No flat `images/` folder exists in the HF repo** (unlike the original GitHub repo's expected layout) -- images are packed inside the parquet files as `{item_id: string, image: {bytes: binary, path: string}}` rows. Confirmed by direct schema inspection, not assumed.

| Parquet | Rows |
|---|---|
| train | 204679 |
| valid | 25132 |
| test | 47854 |

**251008 unique images decoded to `data/images/<item_id>.jpg`** (deduplicated across splits -- nondisjoint means the same item_id can recur in more than one split's parquet; total parquet rows across all three is 277665, 26657 of that gap is cross-split recurrence). Images are small (300x300 RGB JPEGs, mean 9.6 KB) -- total extracted size 2476.0 MB, close to the source parquets' combined size, so this extraction is close to disk-neutral once the parquets are deleted below.

## typespaces.p

A list of 66 (semantic_category_a, semantic_category_b) pairs -- used by Vasileva's type-conditioned architecture (separate learned projection per category pair). Not used in this phase, which explicitly does not implement that architecture (single-variable test: same architecture as phase 8).

