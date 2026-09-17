import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, GroundingDinoForObjectDetection

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE1B_DIR = BASE_DIR.parent / "phase1b_category_balanced"
SAMPLE_CSV = PHASE1B_DIR / "data" / "sample_data.csv"
CROPS_DIR = BASE_DIR / "data" / "crops_b"
LOG_PATH = BASE_DIR / "logs" / "crop_b_log.md"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "product. clothing item."
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PAD_FRACTION = 0.08
MIN_AREA_FRACTION = 0.03
MIN_DIM_PX = 20

def load_sample():
    df = pd.read_csv(SAMPLE_CSV)
    records = []
    for _, row in df.iterrows():
        p = PHASE1B_DIR / row["image_path"]
        if p.exists():
            records.append((row["asin"], p))
    return records

def pad_and_clip_box(box, img_w, img_h):
    x0, y0, x1, y1 = [float(v) for v in box]
    w, h = x1 - x0, y1 - y0
    pad_x, pad_y = w * PAD_FRACTION, h * PAD_FRACTION
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img_w, x1 + pad_x)
    y1 = min(img_h, y1 + pad_y)
    return int(x0), int(y0), int(x1), int(y1)

@torch.no_grad()
def crop_one(model, processor, path):
    img = Image.open(path).convert("RGB")
    img_w, img_h = img.size

    inputs = processor(images=img, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[(img_h, img_w)],
    )[0]

    if len(results["boxes"]) == 0:
        return None, "no_detection"

    best_idx = int(results["scores"].argmax())
    box = results["boxes"][best_idx].tolist()

    x0, y0, x1, y1 = pad_and_clip_box(box, img_w, img_h)
    crop_w, crop_h = x1 - x0, y1 - y0
    if crop_w < MIN_DIM_PX or crop_h < MIN_DIM_PX:
        return None, "degenerate_bbox"
    if (crop_w * crop_h) < MIN_AREA_FRACTION * (img_w * img_h):
        return None, "near_empty_crop"

    return img.crop((x0, y0, x1, y1)), None

def main():
    records = load_sample()
    print(f"Loaded {len(records)} images to crop with Grounding DINO (Method B).")
    print(f"Device: {DEVICE}, prompt: '{TEXT_PROMPT}'")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(MODEL_ID).eval().to(DEVICE)

    failures = []
    n_ok = 0
    t0 = time.time()

    for asin, path in tqdm(records, desc="grounding-dino crop"):
        try:
            cropped, reason = crop_one(model, processor, path)
        except Exception as e:
            cropped, reason = None, f"exception: {e}"

        if cropped is None:
            failures.append((asin, reason))
            continue

        cropped.save(CROPS_DIR / f"{asin}.jpg", quality=95)
        n_ok += 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed/60:.1f} min. Success: {n_ok}, Failures: {len(failures)}")

    lines = [
        "# Phase 2 Method B (Grounding DINO) Crop Log",
        "",
        f"Model: {MODEL_ID}, prompt: '{TEXT_PROMPT}', "
        f"box_threshold={BOX_THRESHOLD}, text_threshold={TEXT_THRESHOLD}",
        f"Total images attempted: {len(records)}",
        f"Successful crops: {n_ok}",
        f"Failures: {len(failures)} ({100*len(failures)/len(records):.1f}%)",
        "",
        "## Failure reasons breakdown",
        "",
    ]
    reason_counts = {}
    for _, reason in failures:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {reason}: {count}")

    lines.append("")
    lines.append("## First 15 example failures")
    lines.append("")
    lines.append("| ASIN | Reason |")
    lines.append("|---|---|")
    for asin, reason in failures[:15]:
        lines.append(f"| {asin} | {reason} |")

    LOG_PATH.write_text("\n".join(lines) + "\n")
    print(f"Saved log to {LOG_PATH}")

if __name__ == "__main__":
    main()
