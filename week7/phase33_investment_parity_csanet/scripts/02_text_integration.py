"""
Phase 33, step 2: text integration. Zero architecture changes (siglip_dim
was already a constructor parameter, exactly the situation phase 31 found
for OutfitTransformer) -- base_repr = normalize(concat(image_768, text_768)),
identical construction to phase 27/28/31/32. Runs A3 (recall10 selection +
text, patience=5, same hyperparameters as A2) locally, isolating text's
contribution against A2 (0.0779, image-only + selection fix).
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_core

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE13_DIR = REPO_ROOT / "week4/phase13_csa_net_baseline"
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"
NEG_CANDIDATES = PHASE13_DIR / "data" / "negative_candidates.json"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"

OUT_JSON = BASE_DIR / "data" / "text_integration_result.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print(f"Loading catalog with TEXT (device={DEVICE})...")
    catalog = train_core.load_catalog(EMBEDDINGS_NPZ, TRAINING_DATA, NEG_CANDIDATES,
                                        text_embeddings_npz=TEXT_EMBEDDINGS_NPZ, device=DEVICE)
    print(f"  in_dim={catalog['in_dim']}")
    with open(VAL_BENCHMARK) as f:
        bench = json.load(f)
    val_benchmark = (bench["pools"], bench["queries"])

    print("Running A3: image+text, recall10 selection, patience=5, lr=5e-5, batch_size=96, seed=42")
    t0 = time.time()
    result = train_core.run_training(
        catalog, DEVICE, max_epochs=40, batch_size=96, lr=5e-5, patience=5,
        seed=42, selection_metric="recall10", val_benchmark=val_benchmark, log_every=300,
    )
    print(f"\nA3: best_epoch={result['best_epoch']} val_recall10={result['best_recall10']:.4f} "
          f"n_epochs_run={result['n_epochs_run']} wall_time={result['wall_time_sec']:.0f}s "
          f"n_params={result['n_params']}")
    for row in result["curve"]:
        print(" ", row)

    with open(OUT_JSON, "w") as f:
        json.dump({"best_epoch": result["best_epoch"], "best_recall10": result["best_recall10"],
                    "n_epochs_run": result["n_epochs_run"], "wall_time_sec": result["wall_time_sec"],
                    "n_params": result["n_params"], "curve": result["curve"]}, f, indent=2)
    print(f"Saved {OUT_JSON}")

    if result["best_state"] is not None:
        ckpt_path = BASE_DIR / "models" / "A3_text.pt"
        torch.save(result["best_state"], ckpt_path)
        print(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
