"""
Phase 13, step 4 prep (Modal GPU run): extract CSA-Net's base CNN feature
(x, the 64-d feature BEFORE any category-conditioned subspace masking) for
all 251,008 Polyvore items, using the trained checkpoint from
modal_train_csa_net.py.

Same tar-extraction pattern as phase 9's modal_extract_embeddings.py. The
base feature x is category-independent (only the subspace attention step
that happens on top of it needs category vectors -- see model.py), so one
CNN pass per item is enough; the 11-category-conditioned variants needed at
eval time are derived cheaply from this cached x in 04_csa_cir_eval.py
(no GPU needed for that step).

Usage:
  modal run scripts/modal_extract_csa_features.py
  modal volume get phase13-csa-net-outputs csa_base_features.npz ../embeddings/csa_base_features.npz
"""
import modal

app = modal.App("phase13-csa-net-feature-extraction")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "pillow", "numpy", "tqdm")
    .add_local_dir(".", remote_path="/root/code")
)

images_volume = modal.Volume.from_name("phase9-polyvore-images", create_if_missing=False)
outputs_volume = modal.Volume.from_name("phase13-csa-net-outputs", create_if_missing=True)

BATCH_SIZE = 256


@app.function(gpu="L4", image=image, volumes={"/images_vol": images_volume, "/outputs": outputs_volume}, timeout=3600)
def extract():
    import sys
    import tarfile
    from pathlib import Path

    import numpy as np
    import torch
    from PIL import Image
    from tqdm import tqdm

    sys.path.insert(0, "/root/code")
    from model import CSANet
    from train_core import EVAL_TRANSFORM

    images_dir = Path("/root/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting polyvore_images.tar...")
    with tarfile.open("/images_vol/polyvore_images.tar") as tf:
        tf.extractall(images_dir)
    paths = sorted(p for p in images_dir.glob("*.jpg") if not p.name.startswith("._"))
    item_ids = [p.stem for p in paths]
    print(f"{len(paths)} images to extract features for.")

    device = "cuda"
    model = CSANet(pretrained=False).to(device).eval()
    state_dict = torch.load("/outputs/csa_net_best.pt", map_location=device)
    model.load_state_dict(state_dict)

    all_feats = []
    with torch.no_grad():
        for start in tqdm(range(0, len(paths), BATCH_SIZE)):
            batch_paths = paths[start:start + BATCH_SIZE]
            imgs = torch.stack([EVAL_TRANSFORM(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
            x = model.encode_image(imgs)
            all_feats.append(x.cpu().numpy())

    features = np.concatenate(all_feats, axis=0).astype(np.float32)
    np.savez("/outputs/csa_base_features.npz", item_ids=np.array(item_ids), embeddings=features)
    outputs_volume.commit()
    print(f"Saved {features.shape} features.")


@app.local_entrypoint()
def main():
    extract.remote()
