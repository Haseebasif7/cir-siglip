"""
Phase 10, step 1 (continued): extract SigLIP embeddings for the 220,455
color-perturbed train/val images on a Modal GPU. Only the perturbed images
are new work here -- the originals' embeddings already exist in phase 9's
embeddings/siglip_base.npz and are reused directly, unchanged, by the
training script.

Identical extraction logic and Modal-workflow lessons as phase 9's
modal_extract_embeddings.py (extract the uploaded tar to local container
disk, NOT the Volume mount, since per-file network latency on a Volume was
the actual bottleneck there, not GPU compute) -- only the volume name and
source image count change.

Usage (orchestration, run manually / from local shell, NOT invoked
automatically):
  1. Local: tar the perturbed images dir into one file:
       cd data && tar -cf color_perturbed_images.tar -C color_perturbed .
  2. Upload to a Modal Volume:
       modal volume create phase10-color-perturbed-images   # first time only
       modal volume put phase10-color-perturbed-images data/color_perturbed_images.tar color_perturbed_images.tar
  3. Run this script on Modal:
       modal run scripts/modal_extract_perturbed_embeddings.py
  4. Pull the resulting embeddings back down:
       modal volume get phase10-color-perturbed-images embeddings/siglip_perturbed.npz embeddings/siglip_perturbed.npz
"""
import modal

app = modal.App("phase10-polyvore-siglip-perturbed")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "transformers", "pillow", "numpy", "tqdm", "sentencepiece"
)

volume = modal.Volume.from_name("phase10-color-perturbed-images", create_if_missing=True)

BATCH_SIZE = 256
MODEL_NAME = "google/siglip-base-patch16-224"


@app.cls(gpu="L4", image=image, volumes={"/data": volume}, timeout=7200)
class SigLIPEmbedder:
    @modal.enter()
    def setup(self):
        import tarfile
        from pathlib import Path

        from transformers import AutoModel, AutoProcessor

        images_dir = Path("/root/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        print("Extracting color_perturbed_images.tar into local container disk /root/images ...")
        with tarfile.open("/data/color_perturbed_images.tar") as tf:
            tf.extractall(images_dir)
        n_real = sum(1 for p in images_dir.glob("*.jpg") if not p.name.startswith("._"))
        print(f"Extracted {n_real} real images (AppleDouble '._*' sidecars filtered by glob below).")

        self.device = "cuda"
        self.model = AutoModel.from_pretrained(MODEL_NAME).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

    @modal.method()
    def extract_all(self):
        import numpy as np
        import torch
        from PIL import Image
        from pathlib import Path
        from tqdm import tqdm

        images_dir = Path("/root/images")
        paths = sorted(p for p in images_dir.glob("*.jpg") if not p.name.startswith("._"))
        item_ids = [p.stem for p in paths]
        print(f"Extracting embeddings for {len(paths)} images, batch_size={BATCH_SIZE}...")

        feats = []
        with torch.no_grad():
            for start in tqdm(range(0, len(paths), BATCH_SIZE)):
                batch_paths = paths[start:start + BATCH_SIZE]
                imgs = [Image.open(p).convert("RGB") for p in batch_paths]
                inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
                out = self.model.get_image_features(**inputs)
                pooled = out.pooler_output if hasattr(out, "pooler_output") else out
                feats.append(pooled.cpu().numpy())

        embeddings = np.concatenate(feats, axis=0).astype(np.float32)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        embeddings = embeddings / norm

        out_dir = Path("/data/embeddings")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "siglip_perturbed.npz"
        np.savez(out_path, item_ids=np.array(item_ids), embeddings=embeddings)
        volume.commit()
        print(f"Saved {embeddings.shape} -> {out_path} (volume committed).")
        return len(item_ids), embeddings.shape


@app.local_entrypoint()
def main():
    n, shape = SigLIPEmbedder().extract_all.remote()
    print(f"Done: {n} embeddings, shape {shape}. Run `modal volume get` to fetch the result.")
