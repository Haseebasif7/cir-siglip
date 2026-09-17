# Phase 4, Step 1: DeepFashion In-shop Dataset Counts

Parsed from the official `list_eval_partition.txt` split. Train rows are dropped entirely -- this project benchmarks frozen pretrained encoders only, no training happens here.

| Split | Rows (images) | Unique item_ids |
|---|---|---|
| train (dropped, not used) | 25882 | 3997 |
| query | 14218 | 3985 |
| gallery | 12612 | 3985 |
| query + gallery combined | 26830 | 3985 |

**Query and gallery item_ids are identical sets** (every query item_id has at least one gallery counterpart: 0 missing). **Train item_ids are completely disjoint from query/gallery item_ids** (0 overlap) -- the official split has no train/test item leakage.

Extracted 26830 of 26830 query+gallery images from `img.zip` into `data/images/` (0 missing from the archive).
