"""
Phase 13, step 2 (data prep): precompute semi-hard negative CANDIDATE lists
for the outfit ranking loss.

The paper (section 4.2) says negatives are "semi-hard" and same-category as
the positive, but doesn't specify how the semi-hard subset is selected
before/around the margin comparison. True online semi-hard mining (embed the
ENTIRE same-category pool with the current, still-training CSA-Net model at
every step, then pick margin-satisfying candidates) is computationally
prohibitive here -- same-category pools run up to ~51k items (shoes), and
CSA-Net's embeddings depend on the (context-item-category, positive-category)
pair, so there's no single fixed "current embedding" per item to rank against
without re-encoding the whole pool through the CNN on every step.

Practical approximation adopted here, documented as a deliberate deviation
(see ../implementation_notes.md): use the project's own ALREADY-COMPUTED raw
SigLIP embeddings (phase 9's embeddings/siglip_base.npz, a fixed, frozen,
general-purpose visual similarity space, unrelated to CSA-Net's own trained
embeddings) as a cheap, static proxy for "visually similar, plausibly hard"
same-category negatives -- precomputed ONCE, offline, reused unchanged for
every training epoch. This exactly mirrors phase 9's own hard-negative-mining
convention (04_mine_hard_negatives.py: top-20 same-... candidates by raw
SigLIP cosine similarity, capped at 0.97 to exclude near-duplicate images),
applied here with an added same-category restriction (CSA-Net's own
requirement, phase 9's MNRL setup didn't need this).

At training time, the actual negatives used in each step are a random subset
of a positive's top-20 candidate list -- "semi-hard" in the loose sense of
"visually plausible same-category near neighbors, not scored by the model
under training," not the paper's literal within-margin online definition.
"""
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA_JSON = BASE_DIR / "data" / "training_data.json"
OUT_JSON = BASE_DIR / "data" / "negative_candidates.json"
REPORT_MD = BASE_DIR / "data" / "negative_candidates_summary.md"

TOP_K = 20
SIMILARITY_CAP = 0.97  # exclude near-duplicate images, same convention as phase 9
CHUNK_SIZE = 2000


def mine_category(item_ids_cat, embeddings, idx):
    """Top-K same-category neighbors by cosine similarity, chunked matmul,
    restricted to this category's own item set (much cheaper than the full
    251k x 251k matrix phase 9 used, since categories are far smaller)."""
    cat_idx = np.array([idx[i] for i in item_ids_cat])
    cat_emb = embeddings[cat_idx]  # (Ncat, D), already L2-normalized
    n = len(item_ids_cat)
    candidates = {}
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        chunk_emb = cat_emb[start:end]
        sims = chunk_emb @ cat_emb.T  # (chunk, Ncat)
        for row_i in range(end - start):
            global_row = start + row_i
            row_sims = sims[row_i].copy()
            row_sims[global_row] = -1.0  # exclude self
            order = np.argsort(-row_sims)
            picked = []
            for j in order:
                s = row_sims[j]
                if s < -0.5:
                    break
                if s > SIMILARITY_CAP:
                    continue  # near-duplicate, skip (same as phase 9)
                picked.append(item_ids_cat[j])
                if len(picked) == TOP_K:
                    break
            candidates[item_ids_cat[global_row]] = picked
    return candidates


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    idx = {a: i for i, a in enumerate(item_ids)}

    with open(TRAINING_DATA_JSON) as f:
        td = json.load(f)
    categories = td["categories"]

    all_candidates = {}
    lines = [
        "# Phase 13, Step 2: Negative Candidate Mining Summary",
        "",
        f"Top-{TOP_K} same-category candidates by raw SigLIP cosine similarity "
        f"(similarity cap {SIMILARITY_CAP}, same convention as phase 9's hard-negative "
        "mining), computed separately for train and validation item pools "
        "(no cross-split leakage).",
        "",
        "| Split | Category | Items | Items with >=1 candidate | Mean candidates/item |",
        "|---|---|---|---|---|",
    ]
    for split_key in ("train_items_by_category", "val_items_by_category"):
        split_name = "train" if "train" in split_key else "val"
        for cat in categories:
            item_ids_cat = sorted(set(td[split_key][cat]))
            if not item_ids_cat:
                continue
            cat_candidates = mine_category(item_ids_cat, embeddings, idx)
            all_candidates.setdefault(split_name, {}).update(cat_candidates)
            n_with = sum(1 for v in cat_candidates.values() if len(v) > 0)
            mean_n = np.mean([len(v) for v in cat_candidates.values()]) if cat_candidates else 0.0
            lines.append(f"| {split_name} | {cat} | {len(item_ids_cat)} | {n_with} | {mean_n:.1f} |")
            print(f"{split_name}/{cat}: {len(item_ids_cat)} items, mean {mean_n:.1f} candidates")

    with open(OUT_JSON, "w") as f:
        json.dump(all_candidates, f)
    lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_JSON}")
    print(f"Saved {REPORT_MD}")


if __name__ == "__main__":
    main()
