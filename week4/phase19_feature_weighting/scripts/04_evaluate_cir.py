"""
Phase 19, step 5: evaluate the trained feature-weighting mechanism on the
identical CIR benchmark used throughout this project. The mechanism is
context-independent (no transformer, no aggregation of its own) -- apply
the weight vector to every catalog item's raw SigLIP embedding once,
renormalize, then reuse `cir_eval.py`'s standard mean-pooled-query
evaluator unchanged (mean-pooling commutes with the element-wise weighting,
so pooling the already-weighted vectors is mathematically identical to
weighting after pooling -- no ambiguity in how "applied before
L2-normalization" interacts with query aggregation).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FeatureWeighting
from cir_eval import evaluate_recall, load_benchmark

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
CHECKPOINT_PT = BASE_DIR / "models" / "feature_weighting.pt"
OUT_JSON = BASE_DIR / "data" / "cir_results.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 8192


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    print(f"Loaded {len(item_ids)} SigLIP embeddings.")

    model = FeatureWeighting().to(DEVICE).eval()
    state_dict = torch.load(CHECKPOINT_PT, map_location=DEVICE)
    model.load_state_dict(state_dict)

    outs = []
    emb_t = torch.tensor(embeddings, device=DEVICE)
    with torch.no_grad():
        for start in range(0, len(item_ids), BATCH):
            chunk = emb_t[start:start + BATCH]
            outs.append(model(chunk).cpu().numpy())
    weighted_embeddings = np.concatenate(outs, axis=0).astype(np.float32)

    pools, queries = load_benchmark()
    print(f"Loaded benchmark: {len(queries)} queries, {len(pools)} category pools")

    recall, n_total, n_skipped = evaluate_recall(pools, queries, item_ids, weighted_embeddings)
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"Recall@10={recall[10]:.4f} Recall@30={recall[30]:.4f} Recall@50={recall[50]:.4f}")

    import json
    with open(OUT_JSON, "w") as f:
        json.dump({"recall": recall, "n_total": n_total, "n_skipped": n_skipped}, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
