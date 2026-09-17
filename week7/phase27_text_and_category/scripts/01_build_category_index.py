"""
Phase 27, step 1: build a compact, item-ordered integer array mapping every
catalog item to its semantic_category index (per model.py's fixed
CATEGORY_LIST order) -- keeps the Modal data volume small (an int8 array
vs. re-uploading the full 251k-item metadata JSON) and lets the training
script do a cheap array lookup instead of a dict-of-strings lookup in the
hot loop.
"""
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"

METADATA_JSON = PHASE9_DIR / "data" / "polyvore_raw" / "polyvore_item_metadata.json"
IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
OUT_NPZ = BASE_DIR / "data" / "category_index.npz"

CATEGORY_LIST = [
    "accessories", "all-body", "bags", "bottoms", "hats", "jewellery",
    "outerwear", "scarves", "shoes", "sunglasses", "tops",
]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_LIST)}


def main():
    with open(METADATA_JSON) as f:
        meta = json.load(f)

    img_data = np.load(IMAGE_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img_data["item_ids"]]

    cat_idx = np.full(len(item_ids), -1, dtype=np.int8)
    n_unknown = 0
    for i, item_id in enumerate(item_ids):
        cat = meta.get(item_id, {}).get("semantic_category", "")
        if cat in CATEGORY_TO_IDX:
            cat_idx[i] = CATEGORY_TO_IDX[cat]
        else:
            n_unknown += 1

    print(f"{len(item_ids)} items, {n_unknown} with an unrecognized/missing semantic_category "
          f"(left at -1, excluded from category-conditioned training/eval paths).")
    np.savez(OUT_NPZ, item_ids=np.array(item_ids), category_idx=cat_idx,
             category_list=np.array(CATEGORY_LIST))
    print(f"Saved {OUT_NPZ}")


if __name__ == "__main__":
    main()
