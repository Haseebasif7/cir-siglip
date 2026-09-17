"""
Phase 9, step 3: extract SigLIP embeddings for Polyvore items.

extract_siglip() copied unchanged from
week2/phase1b_category_balanced/scripts/02_extract_embeddings.py -- it's
already dataset-agnostic (takes a bare path list). Only load_sample() is
new (reads data/images/ directly rather than a sample_data.csv, since
Polyvore's item identity is the filename stem, no separate manifest needed).

All 251,008 unique items (across train+valid+test) need embeddings: training
uses train+valid outfits, and the official benchmark evaluation (step 5.1)
needs the full test split -- no subset would serve both needs.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "data" / "images"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_sample():
    # macOS tar bundles AppleDouble sidecar files ("._<name>.jpg") alongside
    # real images when archiving -- not real images, filtered out here
    paths = sorted(p for p in IMAGES_DIR.glob("*.jpg") if not p.name.startswith("._"))
    item_ids = [p.stem for p in paths]
    print(f"Loaded {len(item_ids)} images.")
    return item_ids, paths


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def open_rgb(path):
    return Image.open(path).convert("RGB")


@torch.no_grad()
def extract_siglip(paths, model_name="google/siglip-base-patch16-224"):
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_name).eval().to(DEVICE)
    processor = AutoProcessor.from_pretrained(model_name)

    feats = []
    for batch_paths in tqdm(list(batched(paths, BATCH_SIZE)), desc=model_name):
        imgs = [open_rgb(p) for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        out = model.get_image_features(**inputs)
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out
        feats.append(pooled.cpu().numpy())
    return np.concatenate(feats, axis=0)


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def save(item_ids, embeddings):
    embeddings = l2_normalize(embeddings.astype(np.float32))
    out_path = EMBEDDINGS_DIR / "siglip_base.npz"
    np.savez(out_path, item_ids=np.array(item_ids), embeddings=embeddings)
    print(f"Saved: {embeddings.shape} -> {out_path}")


def main():
    print(f"Using device: {DEVICE}")
    item_ids, paths = load_sample()
    embeddings = extract_siglip(paths)
    save(item_ids, embeddings)


if __name__ == "__main__":
    main()
