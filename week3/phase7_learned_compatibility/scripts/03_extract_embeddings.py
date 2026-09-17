"""
Phase 7, step 3: extract SigLIP embeddings for the new (hygiene-filtered)
training pool. SigLIP-only -- this phase learns a compatibility layer on top
of the encoder already established as best (phases 1/1b/4), not
re-benchmarking encoders again.

Trimmed from week2/phase1b_category_balanced/scripts/02_extract_embeddings.py
(same model, batching, device selection, L2-normalize-before-save convention).

Smoke-tested `model.get_image_features(**inputs).pooler_output` against the
venv's transformers==5.14.1 beforehand on 4 local images -- confirmed still
returns a BaseModelOutputWithPooling with the expected (N, 768) pooler_output,
so no API-compatibility shim is needed here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "sample_data_cleaned.csv"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_sample():
    df = pd.read_csv(SAMPLE_CSV)
    asins, paths = [], []
    for _, row in df.iterrows():
        p = BASE_DIR / row["image_path"]
        if p.exists():
            asins.append(row["asin"])
            paths.append(p)
    print(f"Loaded {len(asins)} images with valid local paths (of {len(df)} in cleaned pool).")
    return asins, paths


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
        # defensive: fall back to raw tensor if a future transformers version
        # changes get_image_features to return a plain tensor instead of a
        # BaseModelOutputWithPooling (smoke-tested against 5.14.1: still has pooler_output)
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out
        feats.append(pooled.cpu().numpy())
    return np.concatenate(feats, axis=0)


def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def save(asins, embeddings):
    embeddings = l2_normalize(embeddings.astype(np.float32))
    out_path = EMBEDDINGS_DIR / "siglip_base.npz"
    np.savez(out_path, asins=np.array(asins), embeddings=embeddings)
    print(f"Saved siglip_base: {embeddings.shape} -> {out_path}")


def main():
    print(f"Using device: {DEVICE}")
    asins, paths = load_sample()
    embeddings = extract_siglip(paths)
    save(asins, embeddings)


if __name__ == "__main__":
    main()
