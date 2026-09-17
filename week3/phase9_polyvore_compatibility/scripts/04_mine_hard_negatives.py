"""
Phase 9, step 4 (prep): mine hard-negative candidates for Model B training.

Reuses phase 7's chunked-matmul pattern (week3/phase7_learned_compatibility/
scripts/04_build_training_pairs.py) unchanged in approach: top-20 candidates
per anchor by raw SigLIP cosine similarity, excluding self and all true
positives (train+val combined -- a co-outfit partner is a real relationship
even if it's not the training-split edge under consideration), capped at
0.97 similarity to exclude near-duplicate images.

At ~251k items this is a bigger matmul than phase 7/8's ~25k-item pool
(chunked at 2000 anchors/chunk against the full embedding matrix, ~110-125
chunks) -- run as its own step since it's a substantial, separable
computation, not folded into the training script.
"""
import json
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_NPZ = BASE_DIR / "embeddings" / "siglip_base.npz"
POSITIVE_EDGES_JSON = BASE_DIR / "data" / "positive_edges.json"
HARD_NEG_JSON = BASE_DIR / "data" / "hard_negative_candidates.json"

TOP_K_CANDIDATES = 20
SIMILARITY_CAP = 0.97
CHUNK_SIZE = 2000


def load_data():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)

    with open(POSITIVE_EDGES_JSON) as f:
        edge_records = json.load(f)
    all_positive_targets = {}
    for e in edge_records:  # train + val combined, deliberately
        all_positive_targets.setdefault(e["source"], set()).add(e["target"])

    return item_ids, embeddings, all_positive_targets


def mine_hard_negatives(item_ids, embeddings, all_positive_targets):
    idx = {a: i for i, a in enumerate(item_ids)}
    anchors_with_positives = [a for a in all_positive_targets if a in idx]
    print(f"Mining hard negatives for {len(anchors_with_positives)} anchors "
          f"(of {len(item_ids)} total items)...")

    hard_neg_candidates = {}
    for start in range(0, len(anchors_with_positives), CHUNK_SIZE):
        chunk_anchors = anchors_with_positives[start:start + CHUNK_SIZE]
        chunk_idx = [idx[a] for a in chunk_anchors]
        chunk_emb = embeddings[chunk_idx]
        sims = chunk_emb @ embeddings.T

        for row_i, anchor in enumerate(chunk_anchors):
            row_sims = sims[row_i]
            positives = all_positive_targets.get(anchor, set())
            exclude_idx = {idx[anchor]}
            exclude_idx.update(idx[p] for p in positives if p in idx)

            order = np.argsort(-row_sims)
            candidates = []
            for j in order:
                if j in exclude_idx:
                    continue
                sim = float(row_sims[j])
                if sim > SIMILARITY_CAP:
                    continue
                candidates.append((item_ids[j], sim))
                if len(candidates) >= TOP_K_CANDIDATES:
                    break
            hard_neg_candidates[anchor] = candidates

        done = min(start + CHUNK_SIZE, len(anchors_with_positives))
        print(f"  {done}/{len(anchors_with_positives)} anchors done")

    return hard_neg_candidates


def main():
    item_ids, embeddings, all_positive_targets = load_data()
    hard_neg_candidates = mine_hard_negatives(item_ids, embeddings, all_positive_targets)

    n_with = sum(1 for v in hard_neg_candidates.values() if len(v) > 0)
    n_empty = sum(1 for v in hard_neg_candidates.values() if len(v) == 0)
    print(f"Anchors with >=1 hard-negative candidate: {n_with}, with 0: {n_empty}")

    with open(HARD_NEG_JSON, "w") as f:
        json.dump(hard_neg_candidates, f)
    print(f"Saved {HARD_NEG_JSON}")


if __name__ == "__main__":
    main()
