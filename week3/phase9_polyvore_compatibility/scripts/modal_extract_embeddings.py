"""
Phase 9, step 3 (Modal GPU variant): extract SigLIP embeddings for the
251,008 Polyvore item images on a Modal GPU, instead of locally on MPS
(local run was cancelled at ~2% after estimating ~3-4 hours; a GPU should
finish this in minutes).

Same extract_siglip() logic as the local script (03_extract_embeddings.py) --
same model, same L2-normalize-then-save-to-npz convention -- only the
compute target changes.

Usage (orchestration, run manually / from local shell, NOT invoked
automatically):
  1. Local: tar the already-decoded images dir into one file (done once,
     avoids 251k individual small-file uploads over an unreliable
     connection):
       cd data && tar -cf polyvore_images.tar -C images .
  2. Upload the tar to a Modal Volume (one resumable-ish large transfer):
       modal volume create phase9-polyvore-images   # first time only
       modal volume put phase9-polyvore-images data/polyvore_images.tar polyvore_images.tar
  3. Run this script on Modal (extracts the tar server-side, then runs
     batched GPU inference over all images):
       modal run scripts/modal_extract_embeddings.py
  4. Pull the resulting embeddings back down (one small ~770MB file):
       modal volume get phase9-polyvore-images embeddings/siglip_base.npz embeddings/siglip_base.npz
"""
import modal

app = modal.App("phase9-polyvore-siglip")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "transformers", "pillow", "numpy", "tqdm", "sentencepiece"
)

volume = modal.Volume.from_name("phase9-polyvore-images", create_if_missing=True)

BATCH_SIZE = 256  # much larger than local's 16 -- GPU memory/throughput allows it
MODEL_NAME = "google/siglip-base-patch16-224"


@app.cls(gpu="L4", image=image, volumes={"/data": volume}, timeout=7200)
class SigLIPEmbedder:
    @modal.enter()
    def setup(self):
        import tarfile
        from pathlib import Path

        import torch
        from transformers import AutoModel, AutoProcessor

        # Extract to LOCAL container disk (/root/images), NOT the volume mount
        # (/data) -- reading 251k individual small files back off a Modal
        # Volume (networked filesystem) turned out to be the actual
        # bottleneck (~115s/batch observed, ~27hr projected), not GPU compute.
        # Local container disk is fast NVMe; the volume is only used to move
        # the single tar file in and the final embeddings.npz out.
        images_dir = Path("/root/images")
        images_dir.mkdir(parents=True, exist_ok=True)
        print("Extracting polyvore_images.tar into local container disk /root/images ...")
        with tarfile.open("/data/polyvore_images.tar") as tf:
            tf.extractall(images_dir)
        n_real = sum(1 for p in images_dir.glob("*.jpg") if not p.name.startswith("._"))
        print(f"Extracted {n_real} real images (macOS tar AppleDouble '._*' sidecar "
              f"files present too, filtered out by the glob below, not real images).")

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
        out_path = out_dir / "siglip_base.npz"
        np.savez(out_path, item_ids=np.array(item_ids), embeddings=embeddings)
        volume.commit()
        print(f"Saved {embeddings.shape} -> {out_path} (volume committed).")
        return len(item_ids), embeddings.shape


@app.local_entrypoint()
def main():
    n, shape = SigLIPEmbedder().extract_all.remote()
    print(f"Done: {n} embeddings, shape {shape}. Run `modal volume get` to fetch the result.")
