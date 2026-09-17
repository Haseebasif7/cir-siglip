"""
Phase 31, step 4: scale testing. One axis at a time (n_heads, n_layers,
d_ffn, d_model, d_embed), each with a full 3-point LR recheck at the new
size (phase 25's own discipline -- a size's true winner can be invisible
under an inherited LR, matching phase 25's own width=1024 finding). Batch
size, margin, uniformity_weight held at step 3's winners.

"Wider hidden dimension" (the brief's phrase) is ambiguous for a
transformer -- tested as BOTH readings, labelled separately: d_ffn (the
feed-forward width) and d_model (the item-token/attention width, a coupled
change since d_ffn must scale with it to stay proportional). d_embed (final
query/candidate dim) goes beyond the brief's literal 3 axes but is included
because 64 was a deliberate phase-14 handicap vs. the reference repo's 128
(chosen then for CSA-Net comparability) -- undoing a known handicap is in
scope for a fairness phase.

Sweep budget matches step 3a/3b's own (max_epochs=60, patience=20), for the
same reason: the full 100-epoch/patience=25 budget across 30 configs would
be prohibitively expensive, and step 3c already validated the cheaper
budget as a reasonably reliable (if slightly conservative) ranking proxy.
The eventual scale winner gets its own full-budget confirmation in
04g_select_best_final_train.py, mirroring step 3c.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_JSON = DATA_DIR / "scale_sweep_results.json"

# Step 3's winners, held fixed across this entire sweep.
BASE = {"input_mode": "image_text", "selection_metric": "recall10",
        "batch_size": 384, "uniformity_weight": 0.1, "margin": 0.2}
WINNER_LR = 1.5e-4
LR_RECHECK = [WINNER_LR * 0.5, WINNER_LR, WINNER_LR * 2.0]
SWEEP_BUDGET = {"max_epochs": 60, "patience": 20}

# axis -> list of (name_suffix, overrides_dict)
AXES = {
    "n_heads": [("16", {"n_heads": 16}), ("32", {"n_heads": 32})],
    "n_layers": [("6", {"n_layers": 6}), ("8", {"n_layers": 8})],
    "d_ffn": [("1024", {"d_ffn": 1024}), ("2048", {"d_ffn": 2048})],
    "d_model": [("256", {"d_model": 256, "d_ffn": 1024}), ("384", {"d_model": 384, "d_ffn": 1536})],
    "d_embed": [("128", {"d_embed": 128}), ("256", {"d_embed": 256})],
}


def build_configs():
    configs = []
    for axis, variants in AXES.items():
        for suffix, overrides in variants:
            for i, lr in enumerate(LR_RECHECK):
                lr_tag = ["half", "same", "double"][i]
                name = f"ot31_scale_{axis}{suffix}_lr{lr_tag}"
                configs.append({"name": name, "lr": lr, "save_checkpoint": False,
                                 **BASE, **SWEEP_BUDGET, **overrides,
                                 "_axis": axis, "_value": suffix, "_lr_tag": lr_tag})
    return configs


def load_existing():
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            return {r["config"]["name"]: r for r in json.load(f)}
    return {}


def save(results):
    with open(OUT_JSON, "w") as f:
        json.dump(list(results.values()), f, indent=2)


def main():
    configs = build_configs()
    # strip the bookkeeping keys before sending to Modal (train_one doesn't need them)
    clean_configs = [{k: v for k, v in c.items() if not k.startswith("_")} for c in configs]
    results = load_existing()
    pending = [c for c in clean_configs if c["name"] not in results]

    if pending:
        train_one = modal.Function.from_name("phase31-fair-baseline-ot", "train_one")
        print(f"Launching {len(pending)} configs via .map()...")
        try:
            for cfg, result in zip(pending, train_one.map(pending)):
                print(f"{cfg['name']}: lr={cfg['lr']:.1e} val_recall10={result['best_recall10']:.4f} "
                      f"best_epoch={result['best_epoch']} n_epochs_run={result['n_epochs_run']} "
                      f"n_params={result['n_params']}")
                results[cfg["name"]] = result
                save(results)
        except Exception as e:
            print(f"  .map() raised: {e!r} -- re-run this script; finished configs are saved.")

    print(f"\n{len(results)}/{len(configs)} configs done. Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
