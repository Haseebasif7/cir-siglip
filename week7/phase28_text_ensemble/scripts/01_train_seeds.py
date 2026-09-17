"""
Phase 28, step 2: train 10 independent seeds of phase 27's text-only
architecture (image+text concat, 1536-d input, use_category="none", no
architectural changes -- per this phase's explicit "do not modify the
projection head shape" instruction).

Seed 42 is NOT retrained: phase 27's own text_only run already used
seed=42 (modal_app.py's DEFAULT_CONFIG, never overridden in phase 27's
CONFIGS), so its checkpoint (week7/phase27_text_and_category/models/
text_only.pt, val_recall10=0.18194) is reused directly as this phase's
seed=42 ensemble member -- exactly the precedent phase 26 set reusing
phase 25's own seed=42 checkpoint.

Seeds 1-9 are trained fresh via phase 27's already-deployed Modal app
("phase27-text-category", function "train_one") -- same architecture code,
just use_category="none" and a new seed each time. No new Modal app/volume
needed; phase 27's own credit-safety features (immediate checkpoint persist
on improvement, idempotent skip-if-done, warm-start resume) apply here too,
unchanged, since this calls the identical function.
"""
import json
import shutil
import subprocess
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE27_DIR = BASE_DIR.parent / "phase27_text_and_category"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
VOLUME_NAME = "phase27-text-category-data"
OUT_JSON = DATA_DIR / "train_seeds_results.json"

NEW_SEEDS = list(range(1, 10))  # seed 42 reused from phase 27 directly


def reuse_seed42():
    """Copies phase 27's text_only checkpoint + result locally, annotated
    to make the reuse explicit and traceable (matches phase 26's own
    convention for reusing phase 25's seed=42 checkpoint)."""
    src_ckpt = PHASE27_DIR / "models" / "text_only.pt"
    dst_ckpt = MODELS_DIR / "text_ensemble_seed42.pt"
    shutil.copy(src_ckpt, dst_ckpt)

    with open(PHASE27_DIR / "data" / "train_variants_results.json") as f:
        phase27_results = json.load(f)
    seed42_result = next(r for r in phase27_results if r["config"]["name"] == "text_only")
    reused = dict(seed42_result)
    reused["config"] = dict(seed42_result["config"], name="text_ensemble_seed42")
    reused["reused_from"] = "week7/phase27_text_and_category/models/text_only.pt"
    reused["local_checkpoint"] = str(dst_ckpt.relative_to(BASE_DIR.parent.parent))
    print(f"seed=42: reused from phase 27 (val_recall10={reused['best_recall10']:.4f}), "
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

    if "text_ensemble_seed42" not in results_by_name:
        results_by_name["text_ensemble_seed42"] = reuse_seed42()
        save_results(results_by_name)

    train_one = modal.Function.from_name("phase27-text-category", "train_one")

    pending_configs = []
    for seed in NEW_SEEDS:
        name = f"text_ensemble_seed{seed}"
        local_ckpt = MODELS_DIR / f"{name}.pt"
        if name in results_by_name and local_ckpt.exists():
            print(f"{name}: already completed locally (val_recall10="
                  f"{results_by_name[name]['best_recall10']:.4f}), skipping.")
            continue
        pending_configs.append({
            "name": name,
            "use_category": "none",
            "seed": seed,
            "max_epochs": 12,
            "patience": 4,
            "save_checkpoint": True,
        })

    if pending_configs:
        # .map() launches all pending seeds as parallel Modal containers
        # (~11 min wall clock instead of ~97 min sequential for 9 runs) --
        # each result is still persisted to disk and its checkpoint
        # downloaded the moment IT individually finishes, not batched until
        # every seed in the map completes, so the same credit-safety
        # property as the sequential version holds: a mid-map interruption
        # never loses an already-finished seed's result.
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
