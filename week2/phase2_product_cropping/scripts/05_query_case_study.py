import json
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rembg import new_session, remove
from transformers import (
    AutoModel, AutoProcessor,
    CLIPModel, CLIPProcessor,
    GroundingDinoForObjectDetection,
)

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE1_DIR = BASE_DIR.parent / "phase1_frozen_embeddings"
PHASE1B_DIR = BASE_DIR.parent / "phase1b_category_balanced"
PHASE2_EMB_DIR = BASE_DIR / "embeddings"
PHASE1B_EMB_DIR = PHASE1B_DIR / "embeddings"
CROPS_A_DIR = BASE_DIR / "data" / "crops_a"
CROPS_B_DIR = BASE_DIR / "data" / "crops_b"

QUERY_ASIN = "B019SPRFPA"
QUERY_IMAGE = PHASE1_DIR / "data" / "images" / f"{QUERY_ASIN}.jpg"

CASE_DIR = BASE_DIR / "qualitative_examples" / "case_study_B019SPRFPA"
CASE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

PAD_FRACTION = 0.08
MIN_AREA_FRACTION = 0.03
MIN_DIM_PX = 20
ALPHA_THRESHOLD = 10
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
GDINO_PROMPT = "product. clothing item."
GDINO_BOX_THRESHOLD = 0.25
GDINO_TEXT_THRESHOLD = 0.2

def pad_and_clip(bbox, img_w, img_h):
    x0, y0, x1, y1 = [float(v) for v in bbox]
    w, h = x1 - x0, y1 - y0
    pad_x, pad_y = w * PAD_FRACTION, h * PAD_FRACTION
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img_w, x1 + pad_x)
    y1 = min(img_h, y1 + pad_y)
    return int(x0), int(y0), int(x1), int(y1)

def crop_rembg(img):
    session = new_session("u2net")
    cutout = np.array(remove(img, session=session))
    alpha = cutout[:, :, 3]
    ys, xs = np.where(alpha > ALPHA_THRESHOLD)
    if len(xs) == 0:
        raise RuntimeError("rembg: empty mask on query image")
    bbox = pad_and_clip((xs.min(), ys.min(), xs.max(), ys.max()), *img.size)
    return img.crop(bbox)

def crop_grounding_dino(img):
    processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(GDINO_MODEL_ID).eval().to(DEVICE)
    inputs = processor(images=img, text=GDINO_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=GDINO_BOX_THRESHOLD, text_threshold=GDINO_TEXT_THRESHOLD,
        target_sizes=[(img.size[1], img.size[0])],
    )[0]
    if len(results["boxes"]) == 0:
        raise RuntimeError("grounding-dino: no detection on query image")
    best_idx = int(results["scores"].argmax())
    bbox = pad_and_clip(results["boxes"][best_idx].tolist(), *img.size)
    return img.crop(bbox)

@torch.no_grad()
def embed_fashionclip(img):
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").eval().to(DEVICE)
    processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    inputs = processor(images=[img], return_tensors="pt").to(DEVICE)
    out = model.get_image_features(**inputs)
    vec = out.pooler_output.cpu().numpy()[0]
    return vec / np.linalg.norm(vec)

@torch.no_grad()
def embed_siglip(img):
    model = AutoModel.from_pretrained("google/siglip-base-patch16-224").eval().to(DEVICE)
    processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    inputs = processor(images=[img], return_tensors="pt").to(DEVICE)
    out = model.get_image_features(**inputs)
    vec = out.pooler_output.cpu().numpy()[0]
    return vec / np.linalg.norm(vec)

def compute_common_evaluable_set():
    crop_a_asins = {p.stem for p in CROPS_A_DIR.glob("*.jpg")}
    crop_b_asins = {p.stem for p in CROPS_B_DIR.glob("*.jpg")}
    return sorted(crop_a_asins & crop_b_asins)

def load_embeddings_dict(path):
    data = np.load(path, allow_pickle=True)
    asins = [str(a) for a in data["asins"]]
    return dict(zip(asins, data["embeddings"]))

def top5(query_vec, pool_asins, pool_matrix):
    sims = pool_matrix @ query_vec
    idx = np.argsort(-sims)[:5]
    return [pool_asins[i] for i in idx], [float(sims[i]) for i in idx]

def load_phase1_original_top5(encoder):
    with open(PHASE1_DIR / "data" / "retrieval_results.json") as f:
        data = json.load(f)
    for q in data[encoder]:
        if q["asin"] == QUERY_ASIN:
            return q["retrieved"][:5], q["hit_flags_at_5"]
    raise RuntimeError(f"{QUERY_ASIN} not found in phase 1 {encoder} results")

def main():
    img = Image.open(QUERY_IMAGE).convert("RGB")
    img.save(CASE_DIR / "query_original.jpg", quality=95)

    print("Cropping query with Method A (rembg)...")
    crop_a_img = crop_rembg(img)
    crop_a_img.save(CASE_DIR / "query_cropA.jpg", quality=95)

    print("Cropping query with Method B (Grounding DINO)...")
    crop_b_img = crop_grounding_dino(img)
    crop_b_img.save(CASE_DIR / "query_cropB.jpg", quality=95)

    common_asins = compute_common_evaluable_set()
    print(f"Phase 2 common evaluable pool size: {len(common_asins)}")

    encoders = {"fashionclip": embed_fashionclip, "siglip_base": embed_siglip}
    all_results = {}

    for encoder, embed_fn in encoders.items():
        print(f"\n=== {encoder} ===")
        q_uncropped = embed_fn(img)
        q_cropA = embed_fn(crop_a_img)
        q_cropB = embed_fn(crop_b_img)

        baseline_dict = load_embeddings_dict(PHASE1B_EMB_DIR / f"{encoder}.npz")
        cropA_dict = load_embeddings_dict(PHASE2_EMB_DIR / f"{encoder}_cropA.npz")
        cropB_dict = load_embeddings_dict(PHASE2_EMB_DIR / f"{encoder}_cropB.npz")

        baseline_matrix = np.stack([baseline_dict[a] for a in common_asins])
        cropA_matrix = np.stack([cropA_dict[a] for a in common_asins])
        cropB_matrix = np.stack([cropB_dict[a] for a in common_asins])

        orig_top5, orig_hits = load_phase1_original_top5(encoder)
        uncropped_top5, uncropped_sims = top5(q_uncropped, common_asins, baseline_matrix)
        cropA_top5, cropA_sims = top5(q_cropA, common_asins, cropA_matrix)
        cropB_top5, cropB_sims = top5(q_cropB, common_asins, cropB_matrix)

        df1 = pd.read_csv(PHASE1_DIR / "data" / "sample_data.csv")
        row = df1[df1["asin"] == QUERY_ASIN].iloc[0]
        import ast
        related = set(ast.literal_eval(row["also_buy"])) | set(ast.literal_eval(row["also_viewed"]))

        def hit_flags(asins):
            return [1 if a in related else 0 for a in asins]

        all_results[encoder] = {
            "phase1_original": {"retrieved": orig_top5, "hit_flags": orig_hits},
            "phase2_uncropped": {"retrieved": uncropped_top5, "hit_flags": hit_flags(uncropped_top5)},
            "phase2_cropA": {"retrieved": cropA_top5, "hit_flags": hit_flags(cropA_top5)},
            "phase2_cropB": {"retrieved": cropB_top5, "hit_flags": hit_flags(cropB_top5)},
        }
        for variant, r in all_results[encoder].items():
            print(f"  {variant}: {r['retrieved']} hits={r['hit_flags']}")

    with open(CASE_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {CASE_DIR / 'results.json'}")

if __name__ == "__main__":
    main()
