"""
Phase 20, step 1: re-run phase 9's model (`model_a_random_negs.pt`,
the ProjectionHead trained on real Polyvore outfit co-occurrence with
random negatives, no dial, no attention) directly against the current
CIR benchmark file, confirming the cited Recall@10/30/50
(0.1317/0.2464/0.3216) still holds exactly -- not assumed. Same discipline
phase 19 used to confirm raw SigLIP's number before trusting it.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProjectionHead
from cir_eval import evaluate_recall, load_benchmark

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
PHASE9_MODEL_A = PHASE9_DIR / "models" / "model_a_random_negs.pt"
OUT_MD = BASE_DIR / "verification_check.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CITED_RECALL = {10: 0.1317, 30: 0.2464, 50: 0.3216}


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    print(f"Loaded {len(item_ids)} raw SigLIP embeddings.")

    model = ProjectionHead().to(DEVICE).eval()
    model.load_state_dict(torch.load(PHASE9_MODEL_A, map_location=DEVICE))

    with torch.no_grad():
        x = torch.tensor(embeddings, device=DEVICE)
        projected = model(x).cpu().numpy()

    pools, queries = load_benchmark()
    print(f"Loaded benchmark: {len(queries)} queries, {len(pools)} category pools")

    recall, n_total, n_skipped = evaluate_recall(pools, queries, item_ids, projected)
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"Recall@10={recall[10]:.4f} Recall@30={recall[30]:.4f} Recall@50={recall[50]:.4f}")

    match = all(abs(recall[k] - CITED_RECALL[k]) < 1e-4 for k in CITED_RECALL)

    lines = [
        "# Phase 20, Step 1: Phase 9 Model A Re-Verification",
        "",
        "Re-ran phase 9's existing checkpoint (`week3/phase9_polyvore_compatibility/models/model_a_random_negs.pt`, "
        "a plain ProjectionHead: frozen SigLIP 768-d -> 256 -> 128, trained on real Polyvore outfit "
        "co-occurrence with random negatives, no dial, no attention, no additional mechanism -- this is "
        "exactly what \"projection\" means per the brief) directly against the current CIR benchmark "
        "(`week4/phase12_controllable_modes/data/cir_benchmark.json`), the same file used throughout "
        "this entire redirected sequence. No retraining -- this checkpoint is unchanged since phase 9.",
        "",
        "| | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
        f"| Cited (phase 12 onward) | {CITED_RECALL[10]:.4f} | {CITED_RECALL[30]:.4f} | {CITED_RECALL[50]:.4f} |",
        f"| Re-measured here | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} |",
        "",
        f"n_total={n_total}, n_skipped={n_skipped}.",
        "",
        f"**{'CONFIRMED: exact match.' if match else 'MISMATCH -- investigate before treating this as the winning configuration.'}**",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")
    if not match:
        print("WARNING: mismatch found, see verification_check.md")
    return recall, match


if __name__ == "__main__":
    main()
