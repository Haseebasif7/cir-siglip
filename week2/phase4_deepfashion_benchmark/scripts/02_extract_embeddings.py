"""
Phase 4, Step 2: extract embeddings for all four frozen techniques
(ResNet50, CLIP ViT-B/32, FashionCLIP, SigLIP) on the DeepFashion query+gallery set.

Structure copied verbatim from phase1b_category_balanced/scripts/02_extract_embeddings.py
(itself byte-identical to phase 1's version) -- same model loading, batching, device
selection, and L2-normalization for every technique. The only change is the id column:
`image_name` (a DeepFashion image path) instead of `asin` (an Amazon product id), since
this dataset has multiple images per item_id and retrieval operates per-image. Saved
under an `image_ids` key (not `asins`) to make that distinction explicit downstream.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
QUERY_GALLERY_CSV = BASE_DIR / "data" / "query_gallery.csv"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def load_sample():
    df = pd.read_csv(QUERY_GALLERY_CSV)
    image_ids, paths = [], []
    for _, row in df.iterrows():
        p = BASE_DIR / row["image_path"]
        if p.exists():
            image_ids.append(row["image_name"])
            paths.append(p)
    print(f"Loaded {len(image_ids)} images with valid local paths.")
    return image_ids, paths

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def open_rgb(path):
    return Image.open(path).convert("RGB")

@torch.no_grad()
def extract_resnet50(paths):
    from torchvision.models import resnet50, ResNet50_Weights

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(DEVICE)
    transform = weights.transforms()

    feats = []
    for batch_paths in tqdm(list(batched(paths, BATCH_SIZE)), desc="ResNet50"):
        imgs = torch.stack([transform(open_rgb(p)) for p in batch_paths]).to(DEVICE)
        out = model(imgs)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0)

@torch.no_grad()
def extract_clip_family(paths, model_name):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).eval().to(DEVICE)
    processor = CLIPProcessor.from_pretrained(model_name)

    feats = []
    for batch_paths in tqdm(list(batched(paths, BATCH_SIZE)), desc=model_name):
        imgs = [open_rgb(p) for p in batch_paths]
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        out = model.get_image_features(**inputs)
        feats.append(out.pooler_output.cpu().numpy())
    return np.concatenate(feats, axis=0)

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
        feats.append(out.pooler_output.cpu().numpy())
    return np.concatenate(feats, axis=0)

def l2_normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm

def save(technique, image_ids, embeddings):
    embeddings = l2_normalize(embeddings.astype(np.float32))
    out_path = EMBEDDINGS_DIR / f"{technique}.npz"
    np.savez(out_path, image_ids=np.array(image_ids), embeddings=embeddings)
    print(f"Saved {technique}: {embeddings.shape} -> {out_path}")

def main():
    print(f"Using device: {DEVICE}")
    image_ids, paths = load_sample()

    techniques = {
        "resnet50": lambda: extract_resnet50(paths),
        "clip_vit_b32": lambda: extract_clip_family(paths, "openai/clip-vit-base-patch32"),
        "fashionclip": lambda: extract_clip_family(paths, "patrickjohncyh/fashion-clip"),
        "siglip_base": lambda: extract_siglip(paths),
    }

    for name, fn in techniques.items():
        out_path = EMBEDDINGS_DIR / f"{name}.npz"
        if out_path.exists():
            print(f"Skipping {name}, already exists at {out_path}")
            continue
        print(f"\n=== Extracting {name} ===")
        embeddings = fn()
        save(name, image_ids, embeddings)

if __name__ == "__main__":
    main()
