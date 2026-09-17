"""
Phase 10, step 1: generate one color-perturbed twin per Polyvore train/val
image, by shifting the hue channel in HSV space while leaving saturation and
value untouched (shape, texture, and lighting are preserved -- only color
identity changes).

Reuses phase 9's already-decoded local images directly
(week3/phase9_polyvore_compatibility/data/images/<item_id>.jpg) -- no
re-download or re-decode, per the brief. Only the item_ids that actually
appear in phase 9's train/val positive edges need a perturbed twin (test-
split-only items are never used in training and don't need one): 220,455
unique items (204,679 train + 25,132 val, with some overlap between the two
edge sets after set union).

Hue shift: PIL's "HSV" mode stores hue as a single byte (0-255, circular),
not degrees. A shift magnitude is drawn per-image from degrees in
[40, 320] (uniform), deliberately excluding a +/-40 degree window around 0
-- this guarantees a "meaningfully different" hue on every image (the
brief's requirement to avoid tiny shifts that wouldn't change anything)
while still covering nearly the full color wheel. Near-grayscale/low-
saturation images will still get this same hue shift; per the brief this is
fine and needs no special-casing since a near-zero-saturation pixel's hue
barely affects its rendered color regardless of the shift applied.
Deterministic per item_id (seeded from a hash of the id) so the run is
reproducible without needing to store every shift, though the manifest
still records the actual value used for auditability.
"""
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

PHASE9_DIR = Path(__file__).resolve().parent.parent.parent / "phase9_polyvore_compatibility"
SOURCE_IMAGES_DIR = PHASE9_DIR / "data" / "images"
POSITIVE_EDGES_JSON = PHASE9_DIR / "data" / "positive_edges.json"

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_IMAGES_DIR = BASE_DIR / "data" / "color_perturbed"
MANIFEST_JSON = BASE_DIR / "data" / "color_perturb_manifest.json"
REPORT_MD = BASE_DIR / "data" / "color_perturb_summary.md"

MIN_SHIFT_DEG = 40  # exclude +/- this many degrees around 0 -- guarantees a visible hue change
MAX_SHIFT_DEG = 320


def train_val_item_ids():
    with open(POSITIVE_EDGES_JSON) as f:
        edges = json.load(f)
    items = set()
    for e in edges:
        if e["split"] in ("train", "val"):
            items.add(e["source"])
            items.add(e["target"])
    return sorted(items)


def deterministic_shift_degrees(item_id):
    # hash -> [0,1) -> map onto [MIN_SHIFT_DEG, MAX_SHIFT_DEG], deterministic per item_id
    h = hashlib.sha256(item_id.encode()).hexdigest()
    frac = int(h[:12], 16) / float(16 ** 12)
    return MIN_SHIFT_DEG + frac * (MAX_SHIFT_DEG - MIN_SHIFT_DEG)


def hue_shift_image(img, degrees):
    hsv = img.convert("HSV")
    h, s, v = hsv.split()
    h_arr = np.asarray(h, dtype=np.int16)
    shift_units = int(round(degrees / 360.0 * 256.0))
    h_arr = (h_arr + shift_units) % 256
    h_new = Image.fromarray(h_arr.astype(np.uint8), mode="L")
    return Image.merge("HSV", (h_new, s, v)).convert("RGB")


def process_one(item_id):
    src = SOURCE_IMAGES_DIR / f"{item_id}.jpg"
    dst = OUT_IMAGES_DIR / f"{item_id}.jpg"
    if not src.exists():
        return item_id, None, "missing_source"
    try:
        img = Image.open(src).convert("RGB")
        degrees = deterministic_shift_degrees(item_id)
        out = hue_shift_image(img, degrees)
        out.save(dst, "JPEG", quality=90)
        return item_id, degrees, "ok"
    except Exception as exc:
        return item_id, None, f"error:{exc}"


def main():
    OUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    item_ids = train_val_item_ids()
    print(f"{len(item_ids)} train/val items need a color-perturbed twin.")

    manifest = {}
    n_ok, n_missing, n_error = 0, 0, 0
    with Pool(processes=8) as pool:
        for item_id, degrees, status in tqdm(pool.imap_unordered(process_one, item_ids, chunksize=64),
                                              total=len(item_ids)):
            if status == "ok":
                manifest[item_id] = degrees
                n_ok += 1
            elif status == "missing_source":
                n_missing += 1
            else:
                n_error += 1
                print(f"  WARNING: {item_id}: {status}")

    with open(MANIFEST_JSON, "w") as f:
        json.dump(manifest, f)
    print(f"Saved {MANIFEST_JSON} ({len(manifest)} entries).")

    lines = [
        "# Phase 10, Step 1: Color Perturbation Summary",
        "",
        f"- Train/val items requiring a perturbed twin: {len(item_ids)}",
        f"- Successfully perturbed: {n_ok}",
        f"- Missing source image: {n_missing}",
        f"- Errors: {n_error}",
        "",
        f"Hue shift drawn per-image (deterministic, seeded from item_id) from "
        f"[{MIN_SHIFT_DEG}, {MAX_SHIFT_DEG}] degrees -- excludes a +/-{MIN_SHIFT_DEG} degree "
        f"window around 0 to guarantee a visually meaningful color change. Saturation and "
        f"value channels are untouched; shape/texture/lighting preserved.",
        "",
        f"Mean shift applied: {np.mean(list(manifest.values())):.1f} degrees "
        f"(min={min(manifest.values()):.1f}, max={max(manifest.values()):.1f}).",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_MD}")


if __name__ == "__main__":
    main()
