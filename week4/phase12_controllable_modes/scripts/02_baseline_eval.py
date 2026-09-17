"""
Phase 12, step 2: baseline numbers on the new CIR benchmark, before building
anything new. Raw SigLIP and phase 9's existing Model A (Polyvore-trained,
random negatives) checkpoint, reused exactly as-is -- no retraining, no new
embeddings extracted (phase 9's siglip_base.npz already covers all 251,008
Polyvore items, a strict superset of what this benchmark's queries/pools need).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval import evaluate_recall, load_benchmark
from model import ProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
PHASE9_MODEL_A = PHASE9_DIR / "models" / "model_a_random_negs.pt"

OUT_JSON = BASE_DIR / "data" / "baseline_recall.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

LITERATURE_ANCHOR = {"10": 0.0958, "30": 0.1796, "50": 0.2198}


def load_raw_embeddings():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return item_ids, (embeddings / norms).astype(np.float32)


@torch.no_grad()
def project(model, embeddings):
    x = torch.tensor(embeddings, device=DEVICE)
    out = model(x).cpu().numpy()
    return out


def main():
    pools, queries = load_benchmark()
    print(f"Loaded benchmark: {len(queries)} queries, {sum(len(v) for v in pools.values())} pool slots.")

    item_ids, raw_emb = load_raw_embeddings()
    print(f"Loaded {len(item_ids)} raw SigLIP embeddings.")

    results = {}

    recall, n, n_skip = evaluate_recall(pools, queries, item_ids, raw_emb)
    results["Raw SigLIP (alone)"] = {"recall": recall, "n_queries": n, "n_skipped": n_skip}
    print(f"Raw SigLIP: {recall} (n={n}, {n_skip} skipped)")

    model_a = ProjectionHead().to(DEVICE)
    model_a.load_state_dict(torch.load(PHASE9_MODEL_A, map_location=DEVICE))
    model_a.eval()
    p9_emb = project(model_a, raw_emb)
    recall, n, n_skip = evaluate_recall(pools, queries, item_ids, p9_emb)
    results["Phase 9 Model A (Polyvore-trained)"] = {"recall": recall, "n_queries": n, "n_skipped": n_skip}
    print(f"Phase 9 Model A: {recall} (n={n}, {n_skip} skipped)")

    results["OutfitTransformer (literature anchor, not independently reproduced -- see cir_protocol_notes.md)"] = {
        "recall": LITERATURE_ANCHOR, "n_queries": None, "n_skipped": None,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
