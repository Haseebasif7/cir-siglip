"""
Phase 27, step 0: extract a SigLIP text embedding for every catalog item's
text description, plus one for each of the 11 semantic categories (used by
the "SigLIP-encoded category phrase" conditioning variant).

Decision #1 (approved): cascaded text-source fallback, description -> title
-> url_name, NOT the brief's literal single-field reading. Checked directly
against the actual metadata before building anything: `description` is
filled for only 28.7% of the full catalog (72,152 / 251,008 items) and
31.2% of items that actually appear in the CIR benchmark -- using it alone
would silently zero-vector the large majority of items, defeating the whole
point of testing whether text helps. `url_name` (a cleaned product title,
e.g. "river island green tropical bardot") is filled for 100% of items and
is genuinely descriptive text, so it's the final fallback rung. This makes
the "use a zero vector if missing" case in the brief a defensive path that
should essentially never trigger here -- reported below, not assumed.

Decision #2 (approved): SigLIP's text tower needs `padding="max_length",
max_length=64` explicitly -- it was trained with a fixed 64-token canonical
sequence length (confirmed via SiglipConfig.text_config.max_position_embeddings
== 64), unlike CLIP-style dynamic padding. Left at HF defaults, embeddings
come out degraded. Documented here as a correctness detail, not an
efficiency one.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"

METADATA_JSON = PHASE9_DIR / "data" / "polyvore_raw" / "polyvore_item_metadata.json"
IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"  # for item_id ordering

OUT_TEXT_NPZ = BASE_DIR / "data" / "text_embeddings.npz"
OUT_CATEGORY_NPZ = BASE_DIR / "data" / "category_text_embeddings.npz"
OUT_COVERAGE_MD = BASE_DIR / "implementation_notes_text_coverage.md"  # appended into implementation_notes.md later

MODEL_NAME = "google/siglip-base-patch16-224"
MAX_LEN = 64
BATCH_SIZE = 256
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Fixed, alphabetical category order -- this IS the index order the learned
# category-embedding table (variant 2) and the SigLIP-phrase lookup
# (variant 3) both use, so it's saved once here as the single source of truth.
CATEGORY_LIST = [
    "accessories", "all-body", "bags", "bottoms", "hats", "jewellery",
    "outerwear", "scarves", "shoes", "sunglasses", "tops",
]


def pick_text(meta_entry):
    """Cascaded fallback per decision #1. Returns (text, source_used)."""
    desc = str(meta_entry.get("description", "")).strip()
    if desc:
        return desc, "description"
    title = str(meta_entry.get("title", "")).strip()
    if title:
        return title, "title"
    url_name = str(meta_entry.get("url_name", "")).strip()
    if url_name:
        return url_name, "url_name"
    return "", "none"


@torch.no_grad()
def encode_texts(model, processor, texts):
    inputs = processor(text=texts, padding="max_length", max_length=MAX_LEN,
                        truncation=True, return_tensors="pt").to(DEVICE)
    out = model.get_text_features(**inputs)
    pooled = out.pooler_output if hasattr(out, "pooler_output") else out
    feats = pooled.cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return feats / norms


def main():
    from transformers import AutoModel, AutoProcessor

    print(f"Loading {MODEL_NAME} (text tower)...")
    model = AutoModel.from_pretrained(MODEL_NAME).eval().to(DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    with open(METADATA_JSON) as f:
        meta = json.load(f)

    # Item order matches the image embeddings npz exactly, so downstream
    # training/eval code can zip the two by position without re-indexing.
    img_data = np.load(IMAGE_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img_data["item_ids"]]

    source_counts = Counter()
    texts, sources, empty_mask = [], [], []
    for item_id in item_ids:
        entry = meta.get(item_id, {})
        text, source = pick_text(entry)
        source_counts[source] += 1
        texts.append(text if text else "")
        sources.append(source)
        empty_mask.append(text == "")

    print(f"Text source breakdown across {len(item_ids)} items: {dict(source_counts)}")

    print("Encoding item texts through SigLIP's text tower...")
    all_feats = np.zeros((len(item_ids), 768), dtype=np.float32)
    for start in range(0, len(item_ids), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        # Defensive-only path (decision #1's fallback exhausted): truly-empty
        # strings get a literal zero vector rather than being encoded, so a
        # blank string never silently becomes SigLIP's embedding of "".
        nonempty_idx = [i for i, t in enumerate(batch_texts) if t]
        if nonempty_idx:
            feats = encode_texts(model, processor, [batch_texts[i] for i in nonempty_idx])
            for local_i, global_i in enumerate(nonempty_idx):
                all_feats[start + global_i] = feats[local_i]
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"  {start}/{len(item_ids)}")

    n_zero = sum(empty_mask)
    print(f"Items with NO usable text (true zero vector, all 3 fallback rungs empty): {n_zero} "
          f"({100 * n_zero / len(item_ids):.3f}%)")

    np.savez(OUT_TEXT_NPZ, item_ids=np.array(item_ids), embeddings=all_feats,
             source=np.array(sources))
    print(f"Saved {OUT_TEXT_NPZ}")

    print("Encoding the 11 category-name phrases...")
    cat_feats = encode_texts(model, processor, CATEGORY_LIST)
    np.savez(OUT_CATEGORY_NPZ, categories=np.array(CATEGORY_LIST), embeddings=cat_feats)
    print(f"Saved {OUT_CATEGORY_NPZ}")

    lines = [
        "## Text-source coverage (decision #1)\n",
        f"Checked directly against `polyvore_item_metadata.json` before choosing the cascade "
        f"(catalog-wide, {len(item_ids)} items):\n",
        "| Source actually used | Count | % of catalog |",
        "|---|---|---|",
    ]
    total = len(item_ids)
    for src in ["description", "title", "url_name", "none"]:
        c = source_counts.get(src, 0)
        lines.append(f"| {src} | {c} | {100 * c / total:.1f}% |")
    lines.append("")
    lines.append(f"Items with a true zero-vector fallback (all three fields empty): **{n_zero}** "
                  f"({100 * n_zero / total:.3f}%). "
                  f"Confirms the concern that motivated decision #1: the brief's literal "
                  f"`description`-only reading would have zero-vectored the majority of the catalog; "
                  f"the cascade brings that down to effectively nothing.")
    OUT_COVERAGE_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_COVERAGE_MD}")


if __name__ == "__main__":
    main()
