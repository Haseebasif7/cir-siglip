"""
Phase 31, step 3b (extension beyond the brief's literal ask, labelled as
such -- see tuning_log.md): at the step 3a winner (lr=1.5e-4, bs=384,
confirmed after a batch-size bracket check against bs=768 which scored
worse, 0.0701 vs 0.0757), sequentially sweep uniformity_weight then margin.

Justified because (i) phase 23's own discipline, which this brief invokes
by name, included an analogous loss-shaping sweep (temperature) after its
LR/BS grid; (ii) checkpoint_selection_check.md's own finding shows the
uniformity term dominated phase 14b's ORIGINAL (broken) val_loss selection
signal -- leaving it at its inherited value 1.0 untouched would be an
odd place to stop investing, in a phase whose entire purpose is removing
under-investment from this baseline.
"""
import json
from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_JSON = DATA_DIR / "loss_shape_results.json"

WINNER_LR = 1.5e-4
WINNER_BS = 384
SWEEP_BUDGET = {"max_epochs": 60, "patience": 20}

UNIFORMITY_GRID = [0.0, 0.25, 0.5, 1.0]  # 1.0 = phase 14b's original value, reused as the baseline point
UNIFORMITY_REFINE_GRID = [0.1, 0.15, 0.2, 0.3]  # refinement around the coarse grid's 0.25 peak
MARGIN_GRID = [0.1, 0.2, 0.3, 0.5]        # 0.3 = phase 14b's original value


def build_configs():
    # uw=1.0 with the default margin=0.3 is IDENTICAL to the step 3a grid
    # winner (lr=1.5e-4, bs=384, default uniformity_weight=1.0) already run
    # as ot31_grid_lr1.5e-4_bs384 (val_recall10=0.0757) -- reused as this
    # sweep's own baseline point below rather than re-run.
    configs = []
    for uw in UNIFORMITY_GRID:
        if uw == 1.0:
            continue
        configs.append({
            "name": f"ot31_loss_uw{uw}", "input_mode": "image_text", "selection_metric": "recall10",
            "lr": WINNER_LR, "batch_size": WINNER_BS, "uniformity_weight": uw,
            "save_checkpoint": False, **SWEEP_BUDGET,
        })
    return configs


def build_margin_configs(best_uw):
    configs = []
    for m in MARGIN_GRID:
        if m == 0.3 and best_uw == 1.0:
            continue  # identical to the uw sweep's own baseline point (uw=1.0, margin=0.3), don't rerun
        configs.append({
            "name": f"ot31_loss_margin{m}", "input_mode": "image_text", "selection_metric": "recall10",
            "lr": WINNER_LR, "batch_size": WINNER_BS, "uniformity_weight": best_uw, "margin": m,
            "save_checkpoint": False, **SWEEP_BUDGET,
        })
    return configs


def load_existing():
    if OUT_JSON.exists():
        with open(OUT_JSON) as f:
            return {r["config"]["name"]: r for r in json.load(f)}
    return {}


def save(results):
    with open(OUT_JSON, "w") as f:
        json.dump(list(results.values()), f, indent=2)


def run_pending(pending, results, train_one):
    if not pending:
        return
    print(f"Launching {len(pending)} configs via .map()...")
    for cfg, result in zip(pending, train_one.map(pending)):
        print(f"{cfg['name']}: val_recall10={result['best_recall10']:.4f} best_epoch={result['best_epoch']} "
              f"n_epochs_run={result['n_epochs_run']}")
        results[cfg["name"]] = result
        save(results)


def main():
    results = load_existing()
    train_one = modal.Function.from_name("phase31-fair-baseline-ot", "train_one")

    # --- Stage 1: uniformity_weight sweep (margin held at 0.3, phase 14b's original) ---
    uw_configs = build_configs()
    pending = [c for c in uw_configs if c["name"] not in results]
    run_pending(pending, results, train_one)

    with open(DATA_DIR / "lr_bs_grid_results.json") as f:
        grid_results = {r["config"]["name"]: r for r in json.load(f)}
    # NOTE: the grid script's f"{lr:.0e}" name formatting rounds 1.5e-4 to
    # the DISPLAY string "1e-4" (a naming quirk only -- the actual trained
    # cfg["lr"] value is the real 1.5e-4 in every case, confirmed directly
    # against the JSON's own config.lr field). Looked up by actual lr/bs
    # values, not by the misleading name string, to avoid relying on it.
    winner_row = next(r for r in grid_results.values()
                       if abs(r["config"]["lr"] - WINNER_LR) < 1e-10 and r["config"]["batch_size"] == WINNER_BS)
    uw_scores = {1.0: winner_row["best_recall10"]}  # reused, not re-run
    for uw in UNIFORMITY_GRID:
        if uw != 1.0 and f"ot31_loss_uw{uw}" in results:
            uw_scores[uw] = results[f"ot31_loss_uw{uw}"]["best_recall10"]
    best_uw = max(uw_scores, key=uw_scores.get)
    print(f"\nStage 1 winner: uniformity_weight={best_uw} (val_recall10={uw_scores[best_uw]:.4f})")

    # --- Stage 1.5: refine around the coarse grid's peak (0.25 dramatically
    # beat both neighbors, 0.0 and 0.5 -- worth bracketing more precisely
    # before locking in a value, given the scale of the gain: +132% over
    # phase 14b's original uniformity_weight=1.0). ---
    if best_uw == 0.25:
        refine_configs = [{
            "name": f"ot31_loss_uw{uw}", "input_mode": "image_text", "selection_metric": "recall10",
            "lr": WINNER_LR, "batch_size": WINNER_BS, "uniformity_weight": uw,
            "save_checkpoint": False, **SWEEP_BUDGET,
        } for uw in UNIFORMITY_REFINE_GRID]
        pending = [c for c in refine_configs if c["name"] not in results]
        run_pending(pending, results, train_one)
        for uw in UNIFORMITY_REFINE_GRID:
            if f"ot31_loss_uw{uw}" in results:
                uw_scores[uw] = results[f"ot31_loss_uw{uw}"]["best_recall10"]
        best_uw = max(uw_scores, key=uw_scores.get)
        print(f"\nStage 1.5 (refined) winner: uniformity_weight={best_uw} (val_recall10={uw_scores[best_uw]:.4f})")
        print("All uniformity_weight scores:", {k: round(v, 4) for k, v in sorted(uw_scores.items())})

    # --- Stage 2: margin sweep at the uniformity_weight winner ---
    margin_configs = build_margin_configs(best_uw)
    pending = [c for c in margin_configs if c["name"] not in results]
    run_pending(pending, results, train_one)

    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
