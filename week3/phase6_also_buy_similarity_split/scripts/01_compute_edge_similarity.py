"""
Phase 6, step 1: cosine similarity for every within-sample also_buy edge.

Reuses phase 1b's exact sample (1,872 products) and its existing SigLIP
embeddings (`week2/phase1b_category_balanced/embeddings/siglip_base.npz`) --
no new embeddings, no new model inference. also_viewed is not used at all
(phase 5 confirmed it's empty for this entire sample).

An edge is (source_asin, target_asin) where target_asin is also inside the
sample -- same restriction phase 5 used, since these are the only edges that
can ever be evaluated against anything computed on this sample. Edges are
kept directed (not deduplicated by symmetrizing A-B and B-A into one), since
also_buy lists are themselves directional and a symmetric dedup would throw
away real edges where only one side lists the other.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced" / "data" / "sample_data.csv"
PHASE1B_EMBEDDINGS_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced" / "embeddings"


def load_sample():
    df = pd.read_csv(SAMPLE_CSV)
    also_buy = {}
    category = {}
    title = {}
    image_path = {}
    for _, row in df.iterrows():
        asin = row["asin"]
        also_buy[asin] = set(ast.literal_eval(row["also_buy"])) if pd.notna(row["also_buy"]) else set()
        category[asin] = row["category_bucket"]
        title[asin] = row["title"]
        image_path[asin] = row["image_path"]
    return also_buy, category, title, image_path


def main(technique="siglip_base"):
    out_json = BASE_DIR / "data" / f"edge_similarities_{technique.replace('_base', '')}.json"
    embeddings_npz = PHASE1B_EMBEDDINGS_DIR / f"{technique}.npz"

    also_buy, category, title, image_path = load_sample()
    sample_asins = set(also_buy.keys())

    data = np.load(embeddings_npz, allow_pickle=True)
    asins_arr = [str(a) for a in data["asins"]]
    embeddings = data["embeddings"]
    # normalize once so dot product = cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    idx = {a: i for i, a in enumerate(asins_arr)}

    edges = []
    missing_embedding = 0
    for src, targets in also_buy.items():
        if src not in idx:
            continue
        for tgt in targets:
            if tgt not in sample_asins:
                continue
            if tgt not in idx:
                missing_embedding += 1
                continue
            sim = float(np.dot(embeddings[idx[src]], embeddings[idx[tgt]]))
            edges.append({
                "source": src,
                "target": tgt,
                "similarity": sim,
                "source_category": category.get(src),
                "target_category": category.get(tgt),
            })

    print(f"Computed similarity for {len(edges)} within-sample also_buy edges "
          f"({missing_embedding} skipped: target had no embedding, e.g. failed download).")

    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "edges": edges,
            "title": title,
            "image_path": image_path,
        }, f)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "siglip_base")
