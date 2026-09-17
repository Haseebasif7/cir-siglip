"""
Phase 7, step 4: build training pairs from the cleaned also_buy edges.

- Positives: directed (anchor, positive) pairs from the cleaned pool's
  also_buy field, restricted to targets present in the cleaned pool with a
  surviving embedding. Not symmetrized -- also_buy is directional, and
  phase 6 established the convention of keeping edges as found.
- Train/val split: edge-level random 90/10 (not anchor-disjoint -- matches
  VBPR's own leave-one-out validation style, and the brief only asks for an
  edge-level holdout to "monitor training").
- Hard-negative mining: brute-force chunked matmul over the full embedding
  matrix (no FAISS needed at this pool's scale, ~1-2 min expected). Computed
  once per unique anchor (not per edge) -- top-20 candidates by cosine
  similarity, excluding:
    (a) the anchor itself,
    (b) every true positive for that anchor (train AND val combined -- a val
        positive leaking in as a "hard negative" would train the model
        against a fact it's later evaluated on),
    (c) candidates with similarity > 0.97 (near-duplicate variants that
        survived MD5 dedup are not genuine hard negatives, they're plausible
        false negatives).
"""
import ast
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "sample_data_cleaned.csv"
EMBEDDINGS_NPZ = BASE_DIR / "embeddings" / "siglip_base.npz"
POSITIVE_EDGES_JSON = BASE_DIR / "data" / "positive_edges.json"
HARD_NEG_JSON = BASE_DIR / "data" / "hard_negative_candidates.json"
REPORT_MD = BASE_DIR / "training_pool_summary.md"

SEED = 42
VAL_FRACTION = 0.10
TOP_K_CANDIDATES = 20
SIMILARITY_CAP = 0.97
CHUNK_SIZE = 2000


def load_data():
    df = pd.read_csv(SAMPLE_CSV)
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    asins_arr = [str(a) for a in data["asins"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return df, asins_arr, embeddings


def build_positive_edges(df, valid_asins):
    edges = []
    for _, row in df.iterrows():
        src = row["asin"]
        if src not in valid_asins:
            continue
        also_buy = ast.literal_eval(row["also_buy"]) if pd.notna(row["also_buy"]) else []
        for tgt in also_buy:
            if tgt in valid_asins and tgt != src:
                edges.append((src, tgt))
    return edges


def split_edges(edges):
    rng = random.Random(SEED)
    shuffled = edges[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_FRACTION)
    val_edges = shuffled[:n_val]
    train_edges = shuffled[n_val:]
    return train_edges, val_edges


def mine_hard_negatives(asins_arr, embeddings, all_positive_targets):
    """all_positive_targets: dict anchor -> set of ALL true positive targets (train+val combined)."""
    idx = {a: i for i, a in enumerate(asins_arr)}
    n = len(asins_arr)
    anchors_with_positives = [a for a in all_positive_targets if a in idx]

    hard_neg_candidates = {}
    for start in range(0, len(anchors_with_positives), CHUNK_SIZE):
        chunk_anchors = anchors_with_positives[start:start + CHUNK_SIZE]
        chunk_idx = [idx[a] for a in chunk_anchors]
        chunk_emb = embeddings[chunk_idx]  # (C, 768)
        sims = chunk_emb @ embeddings.T  # (C, n)

        for row_i, anchor in enumerate(chunk_anchors):
            row_sims = sims[row_i]
            positives = all_positive_targets.get(anchor, set())
            exclude_idx = {idx[anchor]}
            exclude_idx.update(idx[p] for p in positives if p in idx)

            # top candidates by similarity, excluding self/positives, capped at 0.97
            order = np.argsort(-row_sims)
            candidates = []
            for j in order:
                if j in exclude_idx:
                    continue
                sim = float(row_sims[j])
                if sim > SIMILARITY_CAP:
                    continue
                candidates.append((asins_arr[j], sim))
                if len(candidates) >= TOP_K_CANDIDATES:
                    break
            hard_neg_candidates[anchor] = candidates
        print(f"  hard-negative mining: {min(start + CHUNK_SIZE, len(anchors_with_positives))}/"
              f"{len(anchors_with_positives)} anchors done")

    return hard_neg_candidates


def main():
    df, asins_arr, embeddings = load_data()
    valid_asins = set(asins_arr)
    print(f"Cleaned pool with embeddings: {len(valid_asins)} products.")

    edges = build_positive_edges(df, valid_asins)
    print(f"Positive also_buy edges (both endpoints in pool w/ embeddings): {len(edges)}")

    train_edges, val_edges = split_edges(edges)
    print(f"Train edges: {len(train_edges)}, Val edges: {len(val_edges)} "
          f"({100*len(val_edges)/len(edges):.1f}% held out)")

    all_positive_targets = {}
    for src, tgt in edges:  # train + val combined, for hard-negative exclusion
        all_positive_targets.setdefault(src, set()).add(tgt)

    print("Mining hard-negative candidates (chunked matmul over full embedding matrix)...")
    hard_neg_candidates = mine_hard_negatives(asins_arr, embeddings, all_positive_targets)
    n_with_candidates = sum(1 for v in hard_neg_candidates.values() if len(v) > 0)
    n_empty = sum(1 for v in hard_neg_candidates.values() if len(v) == 0)
    print(f"Anchors with >=1 hard-negative candidate: {n_with_candidates}, "
          f"anchors with 0 (all neighbors excluded/too-similar): {n_empty}")

    edge_records = (
        [{"source": s, "target": t, "split": "train"} for s, t in train_edges] +
        [{"source": s, "target": t, "split": "val"} for s, t in val_edges]
    )
    with open(POSITIVE_EDGES_JSON, "w") as f:
        json.dump(edge_records, f)
    print(f"Saved {POSITIVE_EDGES_JSON}")

    with open(HARD_NEG_JSON, "w") as f:
        json.dump(hard_neg_candidates, f)
    print(f"Saved {HARD_NEG_JSON}")

    with open(REPORT_MD, "a") as f:
        f.write("\n## Training pairs (step 4)\n\n")
        f.write(f"- Positive also_buy edges: {len(edges)}\n")
        f.write(f"- Train edges: {len(train_edges)}\n")
        f.write(f"- Val edges: {len(val_edges)} ({100*len(val_edges)/len(edges):.1f}%)\n")
        f.write(f"- Anchors with >=1 hard-negative candidate: {n_with_candidates}\n")
        f.write(f"- Anchors with 0 hard-negative candidates (all too-similar/excluded): {n_empty}\n")
    print(f"Appended step 4 summary to {REPORT_MD}")


if __name__ == "__main__":
    main()
