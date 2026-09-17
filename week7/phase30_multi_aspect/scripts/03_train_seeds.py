"""
Phase 30, step 4 (only run if the single-seed gate says GO): train 10
independent seeds of the multi-aspect architecture, same methodology as
phases 26/28 (score averaging, validation-benchmark checkpoint selection,
ensemble size sweep to confirm returns don't diminish).

Seed 42 is NOT retrained -- the single-seed gate run (01_single_seed_gate.py)
already trained it to convergence with save_checkpoint=True; its checkpoint
and result are reused directly as this phase's own seed=42 ensemble member,
exactly the precedent phase 26/28 set reusing an earlier phase's seed=42.
"""
import json
import shutil
import subprocess
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VOLUME_NAME = "phase27-text-category-data"
OUT_JSON = DATA_DIR / "train_seeds_results.json"

NEW_SEEDS = list(range(1, 10))  # seed 42 reused from the single-seed gate run


def reuse_seed42():
    src_ckpt = MODELS_DIR / "multiaspect_seed42.pt"
    dst_ckpt = MODELS_DIR / "multiaspect_ensemble_seed42.pt"
    shutil.copy(src_ckpt, dst_ckpt)

    with open(DATA_DIR / "single_seed_result.json") as f:
        gate = json.load(f)
    seed42_result = gate["result"]
    reused = dict(seed42_result)
    reused["config"] = dict(seed42_result["config"], name="multiaspect_ensemble_seed42")
    reused["reused_from"] = "week7/phase30_multi_aspect/models/multiaspect_seed42.pt (single-seed gate run)"
    reused["local_checkpoint"] = str(dst_ckpt.relative_to(BASE_DIR.parent.parent))
    print(f"seed=42: reused from the single-seed gate (val_recall10={reused['best_recall10']:.4f}), "
          f"no retraining needed.")
    return reused


def load_existing_results():
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            return {r["config"]["name"]: r for r in json.load(f)}
    return {}


def save_results(results_by_name):
    with open(OUT_JSON, "w") as f:
        json.dump(list(results_by_name.values()), f, indent=2)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    results_by_name = load_existing_results()

    if "multiaspect_ensemble_seed42" not in results_by_name:
        results_by_name["multiaspect_ensemble_seed42"] = reuse_seed42()
        save_results(results_by_name)

    train_one = modal.Function.from_name("phase30-multi-aspect", "train_one")

    pending_configs = []
    for seed in NEW_SEEDS:
        name = f"multiaspect_ensemble_seed{seed}"
        local_ckpt = MODELS_DIR / f"{name}.pt"
        if name in results_by_name and local_ckpt.exists():
            print(f"{name}: already completed locally (val_recall10="
                  f"{results_by_name[name]['best_recall10']:.4f}), skipping.")
            continue
        pending_configs.append({
            "name": name,
            "seed": seed,
            "max_epochs": 12,
            "patience": 4,
            "save_checkpoint": True,
        })

    if pending_configs:
        print(f"Launching {len(pending_configs)} seeds in parallel via .map()...")
        try:
            for cfg, result in zip(pending_configs, train_one.map(pending_configs)):
                name = cfg["name"]
                print(f"{name}: best_epoch={result['best_epoch']} "
                      f"val_recall10={result['best_recall10']:.4f} "
                      f"wall_time={result['wall_time_sec']:.0f}s")
                ckpt_path = result.get("checkpoint_path")
                if ckpt_path:
                    remote_name = Path(ckpt_path).name
                    local_ckpt = MODELS_DIR / f"{name}.pt"
                    subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote_name,
                                     str(local_ckpt), "--force"], check=True)
                    result["local_checkpoint"] = str(local_ckpt.relative_to(BASE_DIR.parent.parent))
                    print(f"  downloaded -> {local_ckpt}")

                results_by_name[name] = result
                save_results(results_by_name)
        except Exception as e:
            print(f"  .map() raised: {e!r} -- credits or an interruption likely. "
                  f"Re-run this script; already-finished seeds are already saved above, "
                  f"and train_one's own idempotent/warm-start logic covers the rest.")

    print(f"\n{len(results_by_name)}/10 seeds available. Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
