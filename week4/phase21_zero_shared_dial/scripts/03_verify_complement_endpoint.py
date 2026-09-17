"""
Phase 21, step 4 (the central check): does the complement head, trained
with zero shared parameters touched by any other objective, reproduce
phase 9's own confirmed Recall@10/30/50 (0.1317/0.2464/0.3216, re-verified
directly in phase 20's `verification_check.md`) exactly or within normal
training variance? Compared directly, not assumed.
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
CHECKPOINT_PT = BASE_DIR / "models" / "complement_head.pt"
OUT_MD = BASE_DIR / "complement_endpoint_verification.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PHASE9_RECALL = {10: 0.1317, 30: 0.2464, 50: 0.3216}
# normal-training-variance tolerance: this project's own precedent for
# "same setup, different run" numerical drift (e.g. MPS's non-deterministic
# parallel float reductions across runs) -- generous enough to not flag
# genuine run-to-run noise as a real discrepancy, tight enough to catch an
# actual bug (a training-data or architecture mismatch would show up far
# larger than this).
TOLERANCE = 0.01


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    print(f"Loaded {len(item_ids)} SigLIP embeddings.")

    model = ProjectionHead().to(DEVICE).eval()
    model.load_state_dict(torch.load(CHECKPOINT_PT, map_location=DEVICE))

    with torch.no_grad():
        x = torch.tensor(embeddings, device=DEVICE)
        projected = model(x).cpu().numpy()

    pools, queries = load_benchmark()
    print(f"Loaded benchmark: {len(queries)} queries, {len(pools)} category pools")

    recall, n_total, n_skipped = evaluate_recall(pools, queries, item_ids, projected)
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"Recall@10={recall[10]:.4f} Recall@30={recall[30]:.4f} Recall@50={recall[50]:.4f}")

    deltas = {k: recall[k] - PHASE9_RECALL[k] for k in PHASE9_RECALL}
    exact_match = all(abs(deltas[k]) < 1e-4 for k in PHASE9_RECALL)
    within_tolerance = all(abs(deltas[k]) < TOLERANCE for k in PHASE9_RECALL)

    if exact_match:
        verdict = "EXACT MATCH (within floating-point noise, <0.0001 at every K)"
    elif within_tolerance:
        verdict = f"MATCHES WITHIN NORMAL TRAINING VARIANCE (largest delta {max(abs(v) for v in deltas.values()):.4f}, tolerance {TOLERANCE})"
    else:
        verdict = f"FALLS SHORT / DIVERGES beyond normal training variance (largest delta {max(abs(v) for v in deltas.values()):.4f}, tolerance {TOLERANCE}) -- investigate before concluding anything"

    lines = [
        "# Phase 21, Step 4: Complement Head Endpoint Verification -- The Central Check",
        "",
        "Phase 21's complement head, trained with a fully independent set of parameters "
        "(zero shared layer, zero shared capacity with the substitute head), evaluated "
        "ALONE (no blending) on the identical CIR benchmark used throughout this project, "
        "compared directly against phase 9's own confirmed Recall@10/30/50.",
        "",
        "| | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
        f"| Phase 9 (confirmed, phase 20) | {PHASE9_RECALL[10]:.4f} | {PHASE9_RECALL[30]:.4f} | {PHASE9_RECALL[50]:.4f} |",
        f"| Phase 21 complement head (this phase) | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} |",
        f"| Delta | {deltas[10]:+.4f} | {deltas[30]:+.4f} | {deltas[50]:+.4f} |",
        "",
        f"n_total={n_total}, n_skipped={n_skipped}.",
        "",
        f"**Verdict: {verdict}.**",
        "",
        "This is the central check the entire zero-shared-capacity premise rests on: if "
        "there is truly no other objective touching these parameters at any point during "
        "training, this endpoint should reproduce phase 9's own number exactly or within "
        "ordinary run-to-run training noise (this project has repeatedly observed such "
        "noise even for identical setups, e.g. MPS's non-deterministic parallel float "
        "reductions across separate runs). A larger gap than that would mean something in "
        "the setup diverged from phase 9's own -- an accidental difference in data loading, "
        "hyperparameters, or the RNG seeding sequence -- and would need direct "
        "investigation before trusting this phase's design at all, per the brief's own "
        "explicit instruction.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
