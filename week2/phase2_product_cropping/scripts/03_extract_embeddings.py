from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

CROP_DIRS = {
    "cropA": BASE_DIR / "data" / "crops_a",
    "cropB": BASE_DIR / "data" / "crops_b",
}

BATCH_SIZE = 16
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def load_crop_paths(crop_dir):
    paths = sorted(crop_dir.glob("*.jpg"))
    asins = [p.stem for p in paths]
    return asins, paths

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

def open_rgb(path):
    return Image.open(path).convert("RGB")

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

def save(name, asins, embeddings):
    embeddings = l2_normalize(embeddings.astype(np.float32))
    out_path = EMBEDDINGS_DIR / f"{name}.npz"
    np.savez(out_path, asins=np.array(asins), embeddings=embeddings)
    print(f"Saved {name}: {embeddings.shape} -> {out_path}")

def main():
    print(f"Using device: {DEVICE}")

    encoders = {
        "fashionclip": lambda paths: extract_clip_family(paths, "patrickjohncyh/fashion-clip"),
        "siglip_base": lambda paths: extract_siglip(paths),
    }

    for crop_label, crop_dir in CROP_DIRS.items():
        asins, paths = load_crop_paths(crop_dir)
        print(f"\n=== {crop_label}: {len(asins)} cropped images from {crop_dir} ===")
        for encoder_name, fn in encoders.items():
            out_name = f"{encoder_name}_{crop_label}"
            out_path = EMBEDDINGS_DIR / f"{out_name}.npz"
            if out_path.exists():
                print(f"Skipping {out_name}, already exists at {out_path}")
                continue
            print(f"\n--- Extracting {out_name} ---")
            embeddings = fn(paths)
            save(out_name, asins, embeddings)

if __name__ == "__main__":
    main()
