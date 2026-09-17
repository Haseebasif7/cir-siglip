"""
Phase 35, final step: the SINGLE test-benchmark touch for this entire phase.
Loads stage 2's checkpoint only, evaluates once on
week4/phase12_controllable_modes/data/cir_benchmark.json (29,681 queries).
No ensembling (per the brief's explicit "do not" list) -- single model only.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OutfitTransformerSigLIP
import train_lib as tl

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CKPT = MODELS_DIR / "stage2_retrieval_finetune.pt"

# Reference numbers from prior phases, for the required comparison table.
PHASE28_ENSEMBLE = {10: 0.1904, 30: 0.3267, 50: 0.4079}
PHASE32_SOLO = {10: 0.1799, 30: 0.3111, 50: 0.3844}     # = phase 31's single model, reused as ot32 seed42 -- the true matched (same arch/backbone/hparams, no recipe) comparison
PHASE32_ENSEMBLE = {10: 0.1897, 30: 0.3246, 50: 0.4019}  # 3-seed ensemble, cited for broader context only (phase 35 doesn't ensemble)
PHASE34_CSANET = {10: 0.1674, 30: 0.2860, 50: 0.3586}
PHASE14B_ORIGINAL = {10: 0.0588, 30: 0.1286, 50: 0.1809}


def main():
    device = tl.get_device()
    cat = tl.load_catalog(device)

    model = OutfitTransformerSigLIP().to(device)
    sd = torch.load(CKPT, map_location=device)
    model.load_state_dict(sd)
    print(f"Loaded {CKPT}")

    with open(tl.TEST_BENCHMARK) as f:
        b = json.load(f)
    bench_override = (b["pools"], b["queries"])

    recall, n_total, n_skipped = tl.evaluate_recall_targeted(
        model, cat["base_repr"], cat["id_to_gidx"], cat["cat_raw_t"], cat["cat_row_of"], device,
        bench_override=bench_override,
    )
    print(f"Phase 35 full-recipe OutfitTransformer, test benchmark: "
          f"R@10={recall[10]:.4f} R@30={recall[30]:.4f} R@50={recall[50]:.4f} "
          f"(n_total={n_total}, n_skipped={n_skipped})")

    def rel(a, b):
        return (a - b) / b if b else float("nan")

    result = {
        "recall": recall, "n_total": n_total, "n_skipped": n_skipped,
        "vs_phase32_solo": {k: rel(recall[k], PHASE32_SOLO[k]) for k in recall},
        "vs_phase32_ensemble": {k: rel(recall[k], PHASE32_ENSEMBLE[k]) for k in recall},
        "vs_phase28_ensemble": {k: rel(recall[k], PHASE28_ENSEMBLE[k]) for k in recall},
        "vs_phase34_csanet": {k: rel(recall[k], PHASE34_CSANET[k]) for k in recall},
        "vs_phase14b_original": {k: rel(recall[k], PHASE14B_ORIGINAL[k]) for k in recall},
        "beats_phase32_solo": all(recall[k] > PHASE32_SOLO[k] for k in recall),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "final_test_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
