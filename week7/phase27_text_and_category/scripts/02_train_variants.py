"""
Phase 27, step 2: upload this phase's data files onto its own Modal volume,
then launch the three training runs (text-only; text+category via a learned
table; text+category via a SigLIP-encoded phrase), download checkpoints,
save combined results.

Single-seed runs per the brief -- this is an initial signal check, not a
final result; ensembling is explicitly deferred to a follow-up phase if
these show a clear improvement.

Credit-safety (added after switching Modal accounts mid-phase to a fresh
workspace with an unknown credit budget -- see implementation_notes.md):
runs are launched ONE AT A TIME via .remote() (not .map()), each config's
result is written to disk and its checkpoint downloaded IMMEDIATELY after
that specific run finishes, not batched until all three complete. If this
script itself gets interrupted, or Modal credits run out partway through,
already-completed variants are never lost, and re-running this script skips
any variant that already has a local result -- train_one's own internal
idempotent skip (see modal_app.py) makes the remote side redundant-safe too.
"""
import json
import subprocess
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VOLUME_NAME = "phase27-text-category-data"
OUT_JSON = DATA_DIR / "train_variants_results.json"

CONFIGS = [
    {"name": "text_only", "use_category": "none",
     "max_epochs": 12, "patience": 4, "save_checkpoint": True},
    {"name": "text_category_learned", "use_category": "learned",
     "max_epochs": 12, "patience": 4, "save_checkpoint": True},
    {"name": "text_category_siglip_phrase", "use_category": "siglip_phrase",
     "max_epochs": 12, "patience": 4, "save_checkpoint": True},
]


def upload_new_data_files():
    files = ["text_embeddings.npz", "category_index.npz", "category_text_embeddings.npz"]
    for fname in files:
        local_path = DATA_DIR / fname
        assert local_path.exists(), f"missing {local_path}, run 00_extract_text_embeddings_modal.py / 01 first"
        print(f"Uploading {fname}...")
        subprocess.run(
            ["modal", "volume", "put", VOLUME_NAME, str(local_path), fname, "--force"],
            check=True,
        )


def load_existing_results():
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            return {r["config"]["name"]: r for r in json.load(f)}
    return {}


def save_results(results_by_name):
    with open(OUT_JSON, "w") as f:
        json.dump(list(results_by_name.values()), f, indent=2)


def download_checkpoints(cfg, result):
    ckpt_path = result.get("checkpoint_path")
    if ckpt_path:
        remote_name = Path(ckpt_path).name
        local_ckpt = MODELS_DIR / f"{cfg['name']}.pt"
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote_name, str(local_ckpt), "--force"],
                        check=True)
        print(f"  downloaded -> {local_ckpt}")
    cat_ckpt_path = result.get("category_checkpoint_path")
    if cat_ckpt_path:
        remote_name = Path(cat_ckpt_path).name
        local_ckpt = MODELS_DIR / f"{cfg['name']}_category_table.pt"
        subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote_name, str(local_ckpt), "--force"],
                        check=True)
        print(f"  downloaded category table -> {local_ckpt}")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    upload_new_data_files()

    train_one = modal.Function.from_name("phase27-text-category", "train_one")
    results_by_name = load_existing_results()

    for cfg in CONFIGS:
        name = cfg["name"]
        local_ckpt = MODELS_DIR / f"{name}.pt"
        if name in results_by_name and local_ckpt.exists():
            print(f"{name}: already completed locally (val_recall10="
                  f"{results_by_name[name]['best_recall10']:.4f}), skipping.")
            continue

        print(f"Launching {name} (use_category={cfg['use_category']})...")
        try:
            result = train_one.remote(cfg)
        except Exception as e:
            print(f"  FAILED: {e!r} -- credits or an interruption likely. "
                  f"Re-run this script later; train_one's own idempotent/warm-start "
                  f"logic (modal_app.py) will pick up from wherever it left off on the volume.")
            continue

        print(f"  best_epoch={result['best_epoch']} val_recall10={result['best_recall10']:.4f} "
              f"wall_time={result['wall_time_sec']:.0f}s")
        results_by_name[name] = result
        save_results(results_by_name)  # persisted immediately, not batched to the end
        download_checkpoints(cfg, result)

    print(f"\n{len(results_by_name)}/{len(CONFIGS)} variants completed. Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
