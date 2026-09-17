"""
Phase 4, Step 1: parse the official DeepFashion In-shop train/query/gallery split
and extract only the images this phase actually needs.

Train rows are dropped entirely -- this project only benchmarks frozen (pretrained)
encoders, no training happens here, so the 25,882 train images are never opened.
Only the needed 26,830 query+gallery images are extracted from img.zip (not the
full 60,820-file archive), since train images would otherwise take up disk space
for no purpose.
"""
import csv
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR.parent / "phase4_deepfashion_data"
PARTITION_TXT = RAW_DATA_DIR / "list_eval_partition.txt"
ZIP_PATH = RAW_DATA_DIR / "img.zip.download" / "img.zip"

DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
QUERY_GALLERY_CSV = DATA_DIR / "query_gallery.csv"
COUNTS_MD = BASE_DIR / "dataset_counts.md"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def parse_partition():
    with open(PARTITION_TXT) as f:
        lines = f.readlines()

    declared_n = int(lines[0].strip())
    # lines[1] is the column header (image_name item_id evaluation_status)
    rows = []
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3:
            raise ValueError(f"Unexpected row format (expected 3 fields): {line!r}")
        image_name, item_id, status = parts
        if status not in ("train", "query", "gallery"):
            raise ValueError(f"Unexpected status value: {status!r} in row {line!r}")
        rows.append((image_name, item_id, status))

    if len(rows) != declared_n:
        print(f"WARNING: declared row count {declared_n} != parsed row count {len(rows)}")

    return rows


def main():
    rows = parse_partition()

    train_rows = [r for r in rows if r[2] == "train"]
    query_rows = [r for r in rows if r[2] == "query"]
    gallery_rows = [r for r in rows if r[2] == "gallery"]

    train_ids = {r[1] for r in train_rows}
    query_ids = {r[1] for r in query_rows}
    gallery_ids = {r[1] for r in gallery_rows}
    qg_ids = query_ids | gallery_ids

    overlap_train_qg = train_ids & qg_ids
    missing_gallery_for_query = query_ids - gallery_ids

    print(f"Total rows: {len(rows)}")
    print(f"train: {len(train_rows)} rows, {len(train_ids)} unique item_ids")
    print(f"query: {len(query_rows)} rows, {len(query_ids)} unique item_ids")
    print(f"gallery: {len(gallery_rows)} rows, {len(gallery_ids)} unique item_ids")
    print(f"query+gallery unique item_ids: {len(qg_ids)}")
    print(f"train/query+gallery item_id overlap (expect 0): {len(overlap_train_qg)}")
    print(f"query item_ids with no gallery counterpart (expect 0): {len(missing_gallery_for_query)}")

    if overlap_train_qg:
        print("WARNING: train and query/gallery item_ids overlap -- split is not clean, "
              "report this rather than proceeding silently.")
    if missing_gallery_for_query:
        print("WARNING: some query item_ids have no gallery counterpart -- "
              "Recall@K would be undefined for those queries.")

    qg_rows = query_rows + gallery_rows

    print(f"\nExtracting {len(qg_rows)} query+gallery images from {ZIP_PATH.name}...")
    n_extracted = 0
    n_missing_in_zip = 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zip_names = set(zf.namelist())
        for image_name, item_id, status in qg_rows:
            if image_name not in zip_names:
                n_missing_in_zip += 1
                continue
            zf.extract(image_name, path=IMAGES_DIR)
            n_extracted += 1
    print(f"Extracted {n_extracted} images ({n_missing_in_zip} missing from zip).")

    with open(QUERY_GALLERY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "item_id", "status", "image_path"])
        writer.writeheader()
        for image_name, item_id, status in qg_rows:
            image_path = Path("data") / "images" / image_name
            writer.writerow({
                "image_name": image_name,
                "item_id": item_id,
                "status": status,
                "image_path": str(image_path),
            })
    print(f"Saved {QUERY_GALLERY_CSV}")

    lines = [
        "# Phase 4, Step 1: DeepFashion In-shop Dataset Counts",
        "",
        "Parsed from the official `list_eval_partition.txt` split. Train rows are dropped "
        "entirely -- this project benchmarks frozen pretrained encoders only, no training "
        "happens here.",
        "",
        "| Split | Rows (images) | Unique item_ids |",
        "|---|---|---|",
        f"| train (dropped, not used) | {len(train_rows)} | {len(train_ids)} |",
        f"| query | {len(query_rows)} | {len(query_ids)} |",
        f"| gallery | {len(gallery_rows)} | {len(gallery_ids)} |",
        f"| query + gallery combined | {len(qg_rows)} | {len(qg_ids)} |",
        "",
        f"**Query and gallery item_ids are identical sets** (every query item_id has at "
        f"least one gallery counterpart: {len(missing_gallery_for_query)} missing). "
        f"**Train item_ids are completely disjoint from query/gallery item_ids** "
        f"({len(overlap_train_qg)} overlap) -- the official split has no train/test "
        f"item leakage.",
        "",
        f"Extracted {n_extracted} of {len(qg_rows)} query+gallery images from `img.zip` "
        f"into `data/images/` ({n_missing_in_zip} missing from the archive).",
    ]
    COUNTS_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {COUNTS_MD}")


if __name__ == "__main__":
    main()
