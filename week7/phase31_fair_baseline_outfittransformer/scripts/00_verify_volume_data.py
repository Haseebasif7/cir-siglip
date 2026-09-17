"""
Phase 31, step 0: verify the shared Modal volume ("phase27-text-category-data")
has everything train_one needs, uploading the two files this phase adds
(training_data.json, cir_train_benchmark.json), and settling -- by hash, not
assumption -- whether the volume's "cir_test_benchmark.json" really is
week4/phase12_controllable_modes/data/cir_benchmark.json's content.
"""
import hashlib
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VOLUME_NAME = "phase27-text-category-data"
OUT_JSON = Path(__file__).resolve().parent.parent / "data" / "volume_hash_check.json"

LOCAL_FILES = {
    "cir_val_benchmark.json": REPO_ROOT / "week4/phase23_hyperparameter_tuning/data/cir_val_benchmark.json",
    "cir_test_benchmark.json": REPO_ROOT / "week4/phase12_controllable_modes/data/cir_benchmark.json",
    "training_data.json": REPO_ROOT / "week4/phase13_csa_net_baseline/data/training_data.json",
    "cir_train_benchmark.json": REPO_ROOT / "week4/phase25_scale/data/cir_train_benchmark.json",
}
TO_UPLOAD = ["training_data.json", "cir_train_benchmark.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def volume_ls():
    out = subprocess.run(["modal", "volume", "ls", VOLUME_NAME], capture_output=True, text=True, check=True)
    return out.stdout


def volume_has(name, listing):
    return name in listing


def main():
    print("Computing local hashes...")
    local_hashes = {}
    for name, path in LOCAL_FILES.items():
        assert path.exists(), f"missing local file: {path}"
        local_hashes[name] = sha256(path)
        print(f"  {name}: {local_hashes[name][:16]}... ({path.stat().st_size:,} bytes)")

    listing = volume_ls()
    present = {name: volume_has(name, listing) for name in LOCAL_FILES}
    print("\nVolume presence:")
    for name, ok in present.items():
        print(f"  {name}: {'present' if ok else 'MISSING'}")

    # Upload the two new files this phase needs.
    for name in TO_UPLOAD:
        print(f"\nUploading {name}...")
        subprocess.run(
            ["modal", "volume", "put", VOLUME_NAME, str(LOCAL_FILES[name]), name, "--force"],
            check=True,
        )

    # Download the pre-existing files back to a scratch location and hash them,
    # to settle (by evidence) whether cir_test_benchmark.json really is
    # cir_benchmark.json's content, and that cir_val_benchmark.json matches
    # what phase 23 built.
    scratch = Path(__file__).resolve().parent.parent / "data" / "_volume_scratch"
    scratch.mkdir(exist_ok=True)
    remote_hashes = {}
    for name in ["cir_val_benchmark.json", "cir_test_benchmark.json"]:
        local_copy = scratch / name
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, name, str(local_copy), "--force"], check=True)
        remote_hashes[name] = sha256(local_copy)

    print("\nHash comparison (local vs. volume, for files not being re-uploaded):")
    match = {}
    for name in ["cir_val_benchmark.json", "cir_test_benchmark.json"]:
        m = remote_hashes[name] == local_hashes[name]
        match[name] = m
        print(f"  {name}: {'MATCH' if m else 'MISMATCH'} "
              f"(local={local_hashes[name][:16]}..., volume={remote_hashes[name][:16]}...)")

    assert match["cir_test_benchmark.json"], (
        "volume's cir_test_benchmark.json does NOT match local cir_benchmark.json -- "
        "do not assume identity, investigate before any training run"
    )
    assert match["cir_val_benchmark.json"], (
        "volume's cir_val_benchmark.json does NOT match phase 23's local file -- investigate"
    )

    # text_embeddings.npz / siglip_base.npz alignment check (item_ids match) --
    # a cheap header-only check via numpy, no need to hash 1.5GB.
    import numpy as np
    img = np.load(REPO_ROOT / "week3/phase9_polyvore_compatibility/embeddings/siglip_base.npz", allow_pickle=True)
    txt = np.load(REPO_ROOT / "week7/phase27_text_and_category/data/text_embeddings.npz", allow_pickle=True)
    img_ids = [str(a) for a in img["item_ids"]]
    txt_ids = [str(a) for a in txt["item_ids"]]
    aligned = img_ids == txt_ids
    print(f"\nsiglip_base.npz / text_embeddings.npz item_id alignment: {'OK' if aligned else 'MISMATCH'} "
          f"({len(img_ids)} items)")
    assert aligned, "text_embeddings.npz is not positionally aligned to siglip_base.npz"

    result = {
        "local_hashes": local_hashes, "remote_hashes": remote_hashes, "match": match,
        "presence_before_upload": present, "uploaded": TO_UPLOAD,
        "text_embedding_alignment": aligned, "n_items": len(img_ids),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {OUT_JSON}")
    print("\nAll checks PASSED. Volume is ready for phase 31 training runs.")


if __name__ == "__main__":
    main()
