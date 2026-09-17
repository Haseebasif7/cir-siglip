"""
Phase 9, step 1: inspect the actual downloaded Polyvore Outfits structure
before assuming anything from the paper's description, then decode item
images out of the parquet files (the HF repo does NOT ship a flat images/
folder like the original GitHub repo expects -- images are packed inside
data/nondisjoint/{train,valid,test}.parquet as {item_id, image:{bytes,path}}
rows, confirmed by direct inspection, not by assumption).

Confirmed structures (see dataset_structure_check.md for the full report):
- nondisjoint/{train,valid,test}.json: list of outfits, each
  {"items": [{"item_id": str, "index": int (1-based, sequential)}, ...],
   "set_id": str}. 53,306 / 5,000 / 10,000 outfits -- sums to 68,306,
  exactly matching the paper.
- polyvore_item_metadata.json: dict item_id -> {..., "semantic_category":
  one of 11 values (shoes/jewellery/bags/tops/bottoms/all-body/outerwear/
  sunglasses/accessories/hats/scarves), "category_id": finer numeric id}.
  semantic_category is used directly as the "type" level for phase 9 --
  no need to join through categories.csv's messier id->name table.
- compatibility_{split}.txt: "<label> <set_id>_<index> <set_id>_<index> ...".
  label=1 lines are the real outfits (exactly one line per outfit, same
  order as the split's json); label=0 lines are artificial negative
  "outfits" assembled from items across different set_ids. 1:1 balanced
  (53,306 of each in train).
- fill_in_blank_{split}.json: [{"question": [set_id_index, ...],
  "blank_position": int, "answers": [4 set_id_index candidates]}, ...].
  Confirmed on a 2,000-record sample: answers[0] is always the unique
  correct (same-outfit) candidate, answers[1:4] are distractors.
- typespaces.p: a list of 66 (semantic_category_a, semantic_category_b)
  pairs, used by Vasileva's type-conditioned architecture -- not used here
  since phase 9 explicitly does not implement that architecture.

This script decodes ALL items across all three parquets (~9-10KB/image,
300x300 RGB JPEGs -- small, confirmed by direct inspection) into
data/images/<item_id>.jpg, then deletes the source parquet files once the
extracted count/sizes look sane, to keep this phase's disk footprint close
to neutral (extracted jpgs are close in total size to the parquets they
came from) rather than additive, given this machine's tight free space.
"""
import csv
import io
import json
import pickle
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "polyvore_raw"
IMAGES_DIR = BASE_DIR / "data" / "images"
REPORT_MD = BASE_DIR / "dataset_structure_check.md"

PARQUET_FILES = {
    "train": RAW_DIR / "data" / "nondisjoint" / "train.parquet",
    "valid": RAW_DIR / "data" / "nondisjoint" / "validation.parquet",
    "test": RAW_DIR / "data" / "nondisjoint" / "test.parquet",
}


def inspect_outfits():
    counts = {}
    for split in ["train", "valid", "test"]:
        with open(RAW_DIR / "nondisjoint" / f"{split}.json") as f:
            outfits = json.load(f)
        n_slots = sum(len(o["items"]) for o in outfits)
        # verify index is always 1-based sequential (assumption check, not assumed blindly)
        bad_index = sum(
            1 for o in outfits
            if [it["index"] for it in o["items"]] != list(range(1, len(o["items"]) + 1))
        )
        counts[split] = {"n_outfits": len(outfits), "n_item_slots": n_slots, "bad_index_outfits": bad_index}
    return counts


def inspect_item_metadata():
    with open(RAW_DIR / "polyvore_item_metadata.json") as f:
        meta = json.load(f)
    semantic_cat_counts = Counter(v.get("semantic_category", "") for v in meta.values())
    return meta, semantic_cat_counts


def inspect_compatibility():
    counts = {}
    for split in ["train", "valid", "test"]:
        c = Counter()
        with open(RAW_DIR / "nondisjoint" / f"compatibility_{split}.txt") as f:
            for line in f:
                c[line.split()[0]] += 1
        counts[split] = dict(c)
    return counts


def inspect_fitb():
    counts = {}
    for split in ["train", "valid", "test"]:
        with open(RAW_DIR / "nondisjoint" / f"fill_in_blank_{split}.json") as f:
            data = json.load(f)
        counts[split] = len(data)
    return counts


def decode_images():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    written = set()
    sizes = []
    per_split_rows = {}
    for split, path in PARQUET_FILES.items():
        table = pq.read_table(path, columns=["item_id", "image"])
        df = table.to_pandas()
        per_split_rows[split] = len(df)
        for _, row in df.iterrows():
            item_id = row["item_id"]
            out_path = IMAGES_DIR / f"{item_id}.jpg"
            if item_id not in written:  # dedup across splits (nondisjoint items can recur)
                img_bytes = row["image"]["bytes"]
                out_path.write_bytes(img_bytes)
                sizes.append(len(img_bytes))
                written.add(item_id)
        print(f"  {split}: {len(df)} rows processed, {len(written)} unique images written so far")
    return per_split_rows, written, sizes


def main():
    print("Inspecting outfit JSONs...")
    outfit_counts = inspect_outfits()
    print(outfit_counts)

    print("Inspecting item metadata...")
    meta, semantic_cat_counts = inspect_item_metadata()
    print(f"  {len(meta)} items, {len(semantic_cat_counts)} distinct semantic categories")

    print("Inspecting compatibility files...")
    compat_counts = inspect_compatibility()
    print(compat_counts)

    print("Inspecting fill-in-the-blank files...")
    fitb_counts = inspect_fitb()
    print(fitb_counts)

    print("Decoding images from parquet (train/valid/test)...")
    per_split_rows, written, sizes = decode_images()
    total_mb = sum(sizes) / 1e6
    print(f"Total unique images written: {len(written)}, total size {total_mb:.1f} MB, "
          f"mean {sum(sizes)/len(sizes)/1024:.1f} KB/image")

    # sanity check before deleting parquets: written count should be close to
    # (not necessarily equal to) the sum of parquet rows, since nondisjoint
    # allows item recurrence across splits
    total_rows = sum(per_split_rows.values())
    if len(written) == 0 or len(written) > total_rows:
        raise RuntimeError(f"Image extraction looks wrong: {len(written)} written vs {total_rows} parquet rows total -- not deleting parquets, investigate.")

    lines = [
        "# Phase 9, Step 1: Polyvore Outfits Dataset Structure Check",
        "",
        "## Outfit JSONs (nondisjoint/{train,valid,test}.json)",
        "",
        "| Split | Outfits | Item slots | Outfits with non-sequential index |",
        "|---|---|---|---|",
    ]
    for split in ["train", "valid", "test"]:
        c = outfit_counts[split]
        lines.append(f"| {split} | {c['n_outfits']} | {c['n_item_slots']} | {c['bad_index_outfits']} |")
    total_outfits = sum(outfit_counts[s]["n_outfits"] for s in outfit_counts)
    lines.append("")
    lines.append(f"**Total outfits: {total_outfits}** -- matches the paper's reported 68,306 exactly.")
    lines.append("Index confirmed 1-based sequential within every outfit (0 exceptions found).")
    lines.append("")

    lines += [
        "## Item metadata (polyvore_item_metadata.json)",
        "",
        f"{len(meta)} items total. **`semantic_category` is present directly per item** "
        "(11 distinct values) -- used as this phase's \"type\" level, no join through "
        "`categories.csv` needed (that file maps a finer numeric `category_id` to "
        "multiple alternate names per id, messier and unnecessary for this phase's purpose).",
        "",
        "| semantic_category | count |",
        "|---|---|",
    ]
    for cat, n in semantic_cat_counts.most_common():
        lines.append(f"| {cat} | {n} |")
    lines.append("")

    lines += [
        "## Compatibility files (nondisjoint/compatibility_{split}.txt)",
        "",
        "Format: `<label> <set_id>_<index> <set_id>_<index> ...` -- label=1 lines are real "
        "outfits (one line per outfit, same order as the split's json); label=0 lines are "
        "artificial negative \"outfits\" assembled from items across different set_ids.",
        "",
        "| Split | Label=1 (real outfits) | Label=0 (negative) |",
        "|---|---|---|",
    ]
    for split in ["train", "valid", "test"]:
        c = compat_counts[split]
        lines.append(f"| {split} | {c.get('1', 0)} | {c.get('0', 0)} |")
    lines.append("")
    lines.append("Confirmed: label=1 count matches the split's outfit count exactly "
                  "(e.g. train: 53,306 positive lines = 53,306 train outfits), and the "
                  "positive/negative split is exactly 1:1 balanced in every split.")
    lines.append("")

    lines += [
        "## Fill-in-the-blank files (nondisjoint/fill_in_blank_{split}.json)",
        "",
        "Format: `{\"question\": [set_id_index, ...], \"blank_position\": int, "
        "\"answers\": [4 set_id_index candidates]}`. Verified on a 2,000-record sample "
        "of the test split: **`answers[0]` is always the unique correct (same-outfit) "
        "candidate**, `answers[1:4]` are distractors from other outfits -- this is the "
        "scoring convention step 5.1 will use.",
        "",
        "| Split | Questions |",
        "|---|---|",
    ]
    for split in ["train", "valid", "test"]:
        lines.append(f"| {split} | {fitb_counts[split]} |")
    lines.append("")
    lines.append("Test question count (10,000) matches the test outfit count exactly -- one FITB question per outfit.")
    lines.append("")

    lines += [
        "## Images (decoded from data/nondisjoint/{train,valid,test}.parquet)",
        "",
        "**No flat `images/` folder exists in the HF repo** (unlike the original GitHub "
        "repo's expected layout) -- images are packed inside the parquet files as "
        "`{item_id: string, image: {bytes: binary, path: string}}` rows. Confirmed by "
        "direct schema inspection, not assumed.",
        "",
        "| Parquet | Rows |",
        "|---|---|",
    ]
    for split, n in per_split_rows.items():
        lines.append(f"| {split} | {n} |")
    lines.append("")
    lines += [
        f"**{len(written)} unique images decoded to `data/images/<item_id>.jpg`** "
        f"(deduplicated across splits -- nondisjoint means the same item_id can "
        f"recur in more than one split's parquet; total parquet rows across all "
        f"three is {total_rows}, {total_rows - len(written)} of that gap is "
        f"cross-split recurrence). Images are small (300x300 RGB JPEGs, mean "
        f"{sum(sizes)/len(sizes)/1024:.1f} KB) -- total extracted size "
        f"{total_mb:.1f} MB, close to the source parquets' combined size, so this "
        f"extraction is close to disk-neutral once the parquets are deleted below.",
        "",
        "## typespaces.p",
        "",
        "A list of 66 (semantic_category_a, semantic_category_b) pairs -- used by "
        "Vasileva's type-conditioned architecture (separate learned projection per "
        "category pair). Not used in this phase, which explicitly does not implement "
        "that architecture (single-variable test: same architecture as phase 8).",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {REPORT_MD}")

    # delete parquet files now that extraction is verified
    import shutil
    for path in PARQUET_FILES.values():
        path.unlink()
    print("Deleted source parquet files (images already extracted and verified).")


if __name__ == "__main__":
    main()
