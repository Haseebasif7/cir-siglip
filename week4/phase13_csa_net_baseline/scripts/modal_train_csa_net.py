"""
Phase 13, step 3 (Modal GPU run): full-scale CSA-Net training.

Reuses the already-populated `phase9-polyvore-images` Modal Volume (same tar
of 251,008 item images used for phase 9/10's SigLIP extraction and phase
13's own negative-candidate mining input) -- no re-upload needed. Runs the
exact same train_core.run_training() used by train_smoke_test.py locally,
just pointed at the full dataset with real epoch/batch settings and a GPU.

Usage:
  modal run scripts/modal_train_csa_net.py
Then pull results back down:
  modal volume get phase13-csa-net-outputs csa_net_best.pt ../models/csa_net_best.pt
  modal volume get phase13-csa-net-outputs training_curves.json ../models/training_curves.json
"""
import modal

app = modal.App("phase13-csa-net-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "pillow", "numpy", "tqdm")
    .add_local_dir(".", remote_path="/root/code")
)

images_volume = modal.Volume.from_name("phase9-polyvore-images", create_if_missing=False)
outputs_volume = modal.Volume.from_name("phase13-csa-net-outputs", create_if_missing=True)
data_volume = modal.Volume.from_name("phase13-csa-net-data", create_if_missing=True)

MAX_EPOCHS = 20
BATCH_SIZE = 96  # paper's own batch size
LR = 5e-5  # paper's own initial LR
PATIENCE = 3  # matches this project's own early-stopping convention (phases 7-12)
NUM_NEGATIVES = 10
MICRO_BATCH_SIZE = 12  # gradient-accumulation chunk size -- see train_core.py's
                        # run_training docstring: batch_size=96's ~1,500 unique
                        # images OOM'd a 24GB A10G in initial testing
FREEZE_BACKBONE_EPOCHS = 3  # see train_core.py's run_training docstring:
                             # direction-collapse fix found after two failed
                             # full-scale attempts (embeddings converging to a
                             # single point on the unit sphere)


@app.function(
    gpu="A10G",
    cpu=8,
    image=image,
    volumes={"/images_vol": images_volume, "/outputs": outputs_volume, "/data": data_volume},
    timeout=6 * 3600,
)
def train():
    import sys
    import tarfile
    from pathlib import Path

    import torch

    sys.path.insert(0, "/root/code")
    from train_core import run_training

    # Extract images tar to local container disk (fast NVMe), same pattern
    # as phase 9's modal_extract_embeddings.py -- reading 251k small files
    # off the networked Volume mount directly was that script's bottleneck.
    images_dir = Path("/root/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting polyvore_images.tar to local container disk...")
    with tarfile.open("/images_vol/polyvore_images.tar") as tf:
        tf.extractall(images_dir)
    n_real = sum(1 for p in images_dir.glob("*.jpg") if not p.name.startswith("._"))
    print(f"Extracted {n_real} images.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device={device}")

    history = run_training(
        images_dir=images_dir,
        training_data_path="/data/training_data.json",
        negative_candidates_path="/data/negative_candidates.json",
        out_dir="/outputs",
        device=device,
        max_epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        patience=PATIENCE,
        num_negatives=NUM_NEGATIVES,
        micro_batch_size=MICRO_BATCH_SIZE,
        freeze_backbone_epochs=FREEZE_BACKBONE_EPOCHS,
        log_every=50,
    )
    outputs_volume.commit()
    print("Done. History:", history)
    return history


@app.local_entrypoint()
def main():
    result = train.remote()
    print(result)
