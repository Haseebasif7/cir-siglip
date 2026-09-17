"""
Phase 23, step 2: evaluate phase 9's existing, untuned checkpoint
(model_a_random_negs.pt) on the new validation benchmark, so every tuning
decision that follows has a concrete number to beat rather than an
assumption that any change helps.
"""
import sys
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProjectionHead
from cir_eval import evaluate_recall, load_benchmark

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
PHASE9_CKPT = PHASE9_DIR / "models" / "model_a_random_negs.pt"
VAL_BENCHMARK = BASE_DIR / "data" / "cir_val_benchmark.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    model = ProjectionHead().to(DEVICE).eval()
    model.load_state_dict(torch.load(PHASE9_CKPT, map_location=DEVICE))
    with torch.no_grad():
        proj = model(torch.tensor(embeddings, device=DEVICE)).cpu().numpy()

    pools, queries = load_benchmark(VAL_BENCHMARK)
    recall, n_total, n_skipped = evaluate_recall(pools, queries, item_ids, proj)
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"Reference point (phase 9, untuned), validation benchmark: "
          f"Recall@10={recall[10]:.4f} Recall@30={recall[30]:.4f} Recall@50={recall[50]:.4f}")

    out_md = BASE_DIR / "reference_point.md"
    out_md.write_text(
        "# Phase 23, Step 2: Reference Point\n\n"
        "Phase 9's existing, untuned checkpoint (`model_a_random_negs.pt`, "
        "LR=1e-3, BATCH_SIZE=128, WEIGHT_DECAY=1e-5, TAU=0.07, R=8/H=0), "
        "evaluated on this phase's own validation benchmark "
        "(`data/cir_val_benchmark.json`, never the test benchmark):\n\n"
        f"| Recall@10 | Recall@30 | Recall@50 | n_queries | n_skipped |\n"
        f"|---|---|---|---|---|\n"
        f"| {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} | {n_total} | {n_skipped} |\n\n"
        "Every configuration tried in this phase is compared against this "
        "number (validation-benchmark Recall@10 as the primary selection "
        "signal), not against phase 9's cited test-benchmark number "
        "(0.1317/0.2464/0.3216) directly -- those two numbers are not "
        "expected to match exactly since they come from different query "
        "sets, but should be in a broadly similar range as a sanity check.\n"
    )
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
