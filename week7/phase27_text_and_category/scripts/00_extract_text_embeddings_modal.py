"""
Phase 27, step 0 (Modal version): local MPS text encoding was measured
directly at ~55-65 min projected for the full 251,008-item catalog (batch
size didn't change this materially -- see implementation_notes.md), squarely
the "more compute than the M4 Air can reasonably provide" case the
project's standing Modal permission exists for. Moved to a T4 here instead.

Same cascaded fallback (description -> title -> url_name, decision #1) and
same padding="max_length", max_length=64 (decision #2) as the abandoned
local script -- logic is identical, only the execution target changed.

Idempotent: if data/text_embeddings.npz and data/category_text_embeddings.npz
already exist on the volume, this skips re-running (credits-safety, per the
account switch to m-haseebasif5 -- see implementation_notes.md's
"Resilience to interruption / account switch" section).
"""
import modal

app = modal.App("phase27-text-extraction")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "numpy", "transformers", "sentencepiece", "protobuf", "pillow"
)
volume = modal.Volume.from_name("phase27-text-category-data", create_if_missing=False)

DATA_DIR = "/data"
MODEL_NAME = "google/siglip-base-patch16-224"
MAX_LEN = 64
BATCH_SIZE = 512

CATEGORY_LIST = [
    "accessories", "all-body", "bags", "bottoms", "hats", "jewellery",
    "outerwear", "scarves", "shoes", "sunglasses", "tops",
]


@app.function(image=image, gpu="T4", volumes={DATA_DIR: volume}, timeout=3600)
def extract():
    import json
    from collections import Counter
    from pathlib import Path

    import numpy as np
    import torch
    from transformers import AutoModel, AutoProcessor

    data_dir = Path(DATA_DIR)
    out_text = data_dir / "text_embeddings.npz"
    out_cat = data_dir / "category_text_embeddings.npz"

    if out_text.exists() and out_cat.exists():
        return {"status": "already_done", "skipped": True}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(device)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with open(data_dir / "polyvore_item_metadata.json") as f:
        meta = json.load(f)
    img_npz = np.load(data_dir / "siglip_base.npz", allow_pickle=True)
    item_ids = [str(a) for a in img_npz["item_ids"]]

    def pick_text(entry):
        desc = str(entry.get("description", "")).strip()
        if desc:
            return desc, "description"
        title = str(entry.get("title", "")).strip()
        if title:
            return title, "title"
        url_name = str(entry.get("url_name", "")).strip()
        if url_name:
            return url_name, "url_name"
        return "", "none"

    source_counts = Counter()
    texts, sources = [], []
    for item_id in item_ids:
        text, source = pick_text(meta.get(item_id, {}))
        texts.append(text)
        sources.append(source)
        source_counts[source] += 1
    print(f"Text source breakdown: {dict(source_counts)}")

    @torch.no_grad()
    def encode_texts(batch_texts):
        inputs = processor(text=batch_texts, padding="max_length", max_length=MAX_LEN,
                            truncation=True, return_tensors="pt").to(device)
        out = model.get_text_features(**inputs)
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = pooled.cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return feats / norms

    all_feats = np.zeros((len(item_ids), 768), dtype=np.float32)
    for start in range(0, len(item_ids), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        nonempty_idx = [i for i, t in enumerate(batch_texts) if t]
        if nonempty_idx:
            feats = encode_texts([batch_texts[i] for i in nonempty_idx])
            for local_i, global_i in enumerate(nonempty_idx):
                all_feats[start + global_i] = feats[local_i]
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  {start}/{len(item_ids)}")

    n_zero = sum(1 for s in sources if s == "none")
    print(f"True zero-vector items: {n_zero}")

    np.savez(out_text, item_ids=np.array(item_ids), embeddings=all_feats, source=np.array(sources))

    print("Encoding category-name phrases...")
    cat_feats = encode_texts(CATEGORY_LIST)
    np.savez(out_cat, categories=np.array(CATEGORY_LIST), embeddings=cat_feats)

    volume.commit()
    return {
        "status": "done",
        "n_items": len(item_ids),
        "source_counts": dict(source_counts),
        "n_zero": n_zero,
    }


@app.local_entrypoint()
def main():
    import json
    result = extract.remote()
    print(json.dumps(result, indent=2))
