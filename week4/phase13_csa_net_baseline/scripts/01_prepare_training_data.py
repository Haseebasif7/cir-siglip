"""
Phase 13, step 1 (data prep): build outfit-level training/validation samples
for CSA-Net's outfit ranking loss, and the 11-category vocabulary used for
the one-hot category vectors fed into the subspace attention sub-network.

CSA-Net's loss (paper section 3.2) operates on whole outfits: a training
sample is (outfit context O, positive item p, negative items N), not a
pairwise edge -- unlike phase 9's positive_edges.json (pairwise, built for
MNRL/InfoNCE), so this is built fresh from the official train/valid outfit
lists rather than reusing phase 9's edges file.

Sample construction mirrors phase 12's CIR benchmark leave-one-out framing
for consistency across the project: for each outfit, one item is held out
as the positive target, the rest form the outfit context O. Only items with
a downloaded image AND a known semantic_category qualify. Outfits with fewer
than 2 qualifying items (1 context + 1 target minimum) are dropped.

Unlike phase 12's benchmark (which enumerates every qualifying item as a
separate leave-one-out query, for evaluation completeness), training only
needs ONE held-out item per outfit per epoch to keep epoch size close to
the paper's own framing (53,306 training outfits = ~53,306 samples/epoch,
batch size 96 per the paper). The held-out item is re-randomized every
epoch by the training script itself (this file just enumerates ALL valid
leave-one-out options per outfit; the training loop samples one per epoch).
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
RAW_DIR = PHASE9_DIR / "data" / "polyvore_raw"
IMAGES_DIR = PHASE9_DIR / "data" / "images"

OUT_DIR = BASE_DIR / "data"

# Fixed order matters: this exact list is the one-hot index assignment used
# everywhere in phase 13 (training and eval). 11 semantic categories, same
# vocabulary phase 9/12 already established for this dataset.
CATEGORIES = [
    "shoes", "jewellery", "bags", "tops", "bottoms", "all-body",
    "outerwear", "sunglasses", "accessories", "hats", "scarves",
]


def load_item_types():
    with open(RAW_DIR / "polyvore_item_metadata.json") as f:
        meta = json.load(f)
    return {item_id: v.get("semantic_category") for item_id, v in meta.items()}


def load_outfits(split):
    with open(RAW_DIR / "nondisjoint" / f"{split}.json") as f:
        return json.load(f)


def build_outfit_records(outfits, types, valid_images):
    records = []
    n_dropped = 0
    for outfit in outfits:
        item_ids = [it["item_id"] for it in outfit["items"]]
        qualifying = [i for i in item_ids if i in valid_images and types.get(i) in CATEGORIES]
        if len(qualifying) < 2:
            n_dropped += 1
            continue
        records.append({"outfit_id": outfit["set_id"], "items": qualifying})
    return records, n_dropped


def main():
    types = load_item_types()
    valid_images = {p.stem for p in IMAGES_DIR.glob("*.jpg") if not p.stem.startswith("._")}
    print(f"{len(valid_images)} items have a downloaded image.")

    train_outfits = load_outfits("train")
    valid_outfits = load_outfits("valid")

    train_records, n_drop_train = build_outfit_records(train_outfits, types, valid_images)
    val_records, n_drop_val = build_outfit_records(valid_outfits, types, valid_images)

    # Per-category item pools (train + valid items only, split-respecting) --
    # used by the training loop for negative sampling. A negative for a
    # positive of category C is drawn from this same-split, same-category
    # pool, excluding items literally in the current outfit.
    train_items_by_cat = {c: [] for c in CATEGORIES}
    for r in train_records:
        for i in r["items"]:
            train_items_by_cat[types[i]].append(i)
    val_items_by_cat = {c: [] for c in CATEGORIES}
    for r in val_records:
        for i in r["items"]:
            val_items_by_cat[types[i]].append(i)

    out = {
        "categories": CATEGORIES,
        "train_outfits": train_records,
        "val_outfits": val_records,
        "train_items_by_category": train_items_by_cat,
        "val_items_by_category": val_items_by_cat,
    }
    with open(OUT_DIR / "training_data.json", "w") as f:
        json.dump(out, f)

    lines = [
        "# Phase 13, Step 1: Training Data Preparation",
        "",
        f"- {len(train_outfits)} raw train outfits -> {len(train_records)} usable "
        f"({n_drop_train} dropped, fewer than 2 qualifying items).",
        f"- {len(valid_outfits)} raw valid outfits -> {len(val_records)} usable "
        f"({n_drop_val} dropped, fewer than 2 qualifying items).",
        "",
        "## Category vocabulary (fixed one-hot order, used throughout phase 13)",
        "",
        "| Index | Category | Train items | Valid items |",
        "|---|---|---|---|",
    ]
    for idx, c in enumerate(CATEGORIES):
        lines.append(f"| {idx} | {c} | {len(train_items_by_cat[c])} | {len(val_items_by_cat[c])} |")
    lines.append("")
    (BASE_DIR / "data" / "training_data_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved {OUT_DIR / 'training_data.json'}")


if __name__ == "__main__":
    main()
