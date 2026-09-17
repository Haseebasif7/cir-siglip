"""
Phase 36, step 4: measured inference cost of each final system's full
test-benchmark evaluation, as built (torch parts on MPS; ours' pooling/ranking in
NumPy on CPU, as its original evaluator did -- the device split is reported per
component). REPEATS full passes, median reported. Also compiles the RECORDED
training wall-clocks from each phase's JSON (no retraining), with device caveats.

Usage: python 04_inference_cost.py [--cpu]   (--cpu forces every system onto CPU
for a device-controlled comparison; written to a separate JSON/section)
"""
import gc
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPEATS = 3
FORCE_CPU = "--cpu" in sys.argv
DEVICE = "cpu" if FORCE_CPU else C.DEVICE
SYSTEMS = [
    ("ours_solo", "ours", [C.PHASE27_MODELS / "text_only.pt"]),
    ("ours_ens", "ours", [C.PHASE28_MODELS / f"text_ensemble_seed{s}.pt" for s in [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
    ("ot_solo", "ot", [C.PHASE32_DIR / "models" / "ot32_seed42.pt"]),
    ("ot_ens", "ot", [C.PHASE32_DIR / "models" / f"ot32_seed{s}.pt" for s in [42, 1, 2]]),
    ("csa_ens", "csa", [C.PHASE34_DIR / "models" / f"csanet34_seed{s}.pt" for s in [42, 1, 2]]),
]
DEVICE_SPLIT = {
    "ours": "candidate projection: torch/" + DEVICE + "; query pooling + cosine + ranking: NumPy/CPU (as the phase 28 evaluator)",
    "ot": "candidate embed_item_alone + query set-encoder forward: torch/" + DEVICE + "; cosine + ranking: NumPy/CPU (as phase 32)",
    "csa": "everything (pool candidate tensors, conditioned context embeddings, distances, ranking): torch/" + DEVICE + " (as phase 34)",
}


def recorded_training_cost():
    """Wall-clock per seed as each phase recorded it. Read defensively -- the JSON
    shapes differ per phase."""
    rows = []
    # ours: phase 28 train_seeds_results.json (list of per-seed records)
    try:
        d = json.load(open(C.REPO_ROOT / "week7/phase28_text_ensemble/data/train_seeds_results.json"))
        recs = d if isinstance(d, list) else list(d.values())
        for r in recs:
            seed = r.get("seed", r.get("config", {}).get("seed"))
            rows.append(("ours", seed, r.get("wall_time_sec"), r.get("device"), "Modal GPU (phase 28)"))
    except Exception as e:  # noqa: BLE001
        rows.append(("ours", None, None, None, f"read failed: {e}"))
    # OT: phase 32 seed_results.json (seeds 1,2) + phase 31 budget_check_results.json (seed 42)
    try:
        d = json.load(open(C.PHASE32_DIR / "data/seed_results.json"))
        recs = d if isinstance(d, list) else list(d.values())
        for r in recs:
            rows.append(("ot", r.get("config", {}).get("seed", r.get("seed")), r.get("wall_time_sec"), r.get("device"), "Modal GPU (phase 32)"))
    except Exception as e:  # noqa: BLE001
        rows.append(("ot", None, None, None, f"read failed: {e}"))
    try:
        d = json.load(open(C.REPO_ROOT / "week7/phase31_fair_baseline_outfittransformer/data/budget_check_results.json"))
        rec = None
        if isinstance(d, dict):
            rec = d.get("ot31_budget_check_full") or next((v for v in d.values() if isinstance(v, dict) and v.get("config", {}).get("name") == "ot31_budget_check_full"), None)
        elif isinstance(d, list):
            rec = next((v for v in d if v.get("config", {}).get("name") == "ot31_budget_check_full" or v.get("name") == "ot31_budget_check_full"), None)
        if rec:
            rows.append(("ot", 42, rec.get("wall_time_sec"), rec.get("device"), "Modal GPU (phase 31, reused as seed 42)"))
        else:
            rows.append(("ot", 42, None, None, "record ot31_budget_check_full not located"))
    except Exception as e:  # noqa: BLE001
        rows.append(("ot", 42, None, None, f"read failed: {e}"))
    # CSA: phase 34 seed_results.json (dict keyed csanet34_seedN) + seed42_gate_result.json
    try:
        d = json.load(open(C.PHASE34_DIR / "data/seed_results.json"))
        recs = list(d.values()) if isinstance(d, dict) else d
        for r in recs:
            rows.append(("csa", r.get("config", {}).get("seed", r.get("seed")), r.get("wall_time_sec"), r.get("device", "mps"), "local M4 MPS (phase 34)"))
        g = json.load(open(C.PHASE34_DIR / "data/seed42_gate_result.json"))
        rows.append(("csa", 42, g.get("wall_time_sec"), g.get("device", "mps"), "local M4 MPS (phase 34 gate run)"))
    except Exception as e:  # noqa: BLE001
        rows.append(("csa", None, None, None, f"read failed: {e}"))
    return rows


def main():
    print(f"device={DEVICE} force_cpu={FORCE_CPU} repeats={REPEATS}")
    item_ids, idx, image_emb, base_repr = C.load_item_data(device=DEVICE)
    del image_emb
    pools, queries = C.load_benchmark()
    cats, cat_to_idx, item_cat = C.load_category_data()
    n_q = len(queries)

    runs = {name: [] for name, _, _ in SYSTEMS}
    for r in range(REPEATS):
        for name, kind, ckpts in SYSTEMS:
            if kind == "ours":
                system = C.OursSystem(name, ckpts, device=DEVICE)
            elif kind == "ot":
                system = C.OTSystem(name, ckpts, device=DEVICE)
            else:
                system = C.CSASystem(name, ckpts, cat_to_idx, item_cat, device=DEVICE)
            if DEVICE == "mps":
                torch.mps.synchronize()
            t_pre = system.precompute(base_repr)
            if DEVICE == "mps":
                torch.mps.synchronize()
            if kind == "ours":
                ranks, n_total, n_skipped, t_eval = system.cir_ranks(pools, queries, idx)
            else:
                ranks, n_total, n_skipped, t_eval = system.cir_ranks(pools, queries, idx, base_repr)
            assert n_total == n_q and n_skipped == 0
            rec = {"precompute_s": t_pre, "query_and_score_s": t_eval, "total_s": t_pre + t_eval,
                   "peak_rss_mb": C.peak_rss_mb(), "mps_allocated_mb": C.mps_allocated_mb() if DEVICE == "mps" else 0.0}
            if kind == "csa":
                rec["csa_pool_candidate_precompute_within_loop_s"] = system.precompute_in_loop_s
            runs[name].append(rec)
            print(f"[rep {r+1}/{REPEATS}] {name}: precompute {t_pre:.2f}s  query+score {t_eval:.2f}s  total {t_pre+t_eval:.2f}s")
            del system, ranks
            gc.collect()
            if DEVICE == "mps":
                torch.mps.empty_cache()

    summary = {"device": DEVICE, "force_cpu": FORCE_CPU, "repeats": REPEATS, "n_queries": n_q, "n_catalog": len(item_ids),
               "systems": {}, "device_split": DEVICE_SPLIT}
    for name, kind, ckpts in SYSTEMS:
        med = {k: statistics.median(rr[k] for rr in runs[name]) for k in runs[name][0] if k.endswith("_s")}
        summary["systems"][name] = {
            "kind": kind, "n_members": len(ckpts), "n_params_per_member": C.N_PARAMS[kind],
            "n_params_total": C.N_PARAMS[kind] * len(ckpts),
            "median": med, "all_runs": runs[name],
            "per_query_ms_median": 1000.0 * med["query_and_score_s"] / n_q,
            "total_per_member_s_median": med["total_s"] / len(ckpts),
            "peak_rss_mb_max": max(rr["peak_rss_mb"] for rr in runs[name]),
            "mps_allocated_mb_max": max(rr["mps_allocated_mb"] for rr in runs[name]),
        }
    summary["recorded_training_cost"] = [
        {"system": s, "seed": seed, "wall_time_sec": w, "device": dev, "note": note} for s, seed, w, dev, note in recorded_training_cost()]
    tag = "_cpu" if FORCE_CPU else ""
    with open(C.DATA_DIR / f"inference_cost{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---------------- markdown (MPS run writes the main file; CPU run appends a section)
    S = summary["systems"]
    L = []
    if not FORCE_CPU:
        L += ["# Phase 36: Measured Inference Cost (full test-benchmark evaluation, as built)", "",
              f"Machine: local M4 (Apple Silicon). {REPEATS} full passes per system, median reported. Each pass = load checkpoints, "
              f"precompute candidate embeddings, embed and score all {n_q} test queries against their category pools (3,000 candidates "
              f"or fewer), rank. Catalog: {len(item_ids)} items. Device use differs by system exactly as in each phase's own evaluator "
              f"and is stated per row -- this is the cost of the pipelines that produced the paper's numbers, not a device-controlled "
              f"benchmark (see the CPU-only section below for that).", "",
              "| System | members | trainable params (total) | precompute (s) | query + score (s) | total (s) | total per member (s) | per query (ms) | peak RSS (MB) | device split |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for name, kind, _ in SYSTEMS:
            e = S[name]
            m = e["median"]
            L.append(f"| `{name}` | {e['n_members']} | {e['n_params_total']:,} | {m['precompute_s']:.1f} | {m['query_and_score_s']:.1f} | "
                     f"{m['total_s']:.1f} | {e['total_per_member_s_median']:.1f} | {e['per_query_ms_median']:.2f} | {e['peak_rss_mb_max']:.0f} | {DEVICE_SPLIT[kind]} |")
        L += ["", "For CSA-Net, candidate embeddings are built per category pool inside the query loop (as its evaluator does); the "
              "time spent on that is included in `query + score` and separately measured at "
              f"{S['csa_ens']['median'].get('csa_pool_candidate_precompute_within_loop_s', 0):.1f} s (median).", "",
              "## Recorded training cost (from each phase's own JSON; different hardware, NOT directly comparable across rows)", "",
              "| System | seed | wall-clock (s) | device | where recorded |", "|---|---|---|---|---|"]
        for r in summary["recorded_training_cost"]:
            w = f"{r['wall_time_sec']:.0f}" if isinstance(r["wall_time_sec"], (int, float)) else "n/a"
            L.append(f"| {r['system']} | {r['seed']} | {w} | {r['device']} | {r['note']} |")
        tot = {}
        for r in summary["recorded_training_cost"]:
            if isinstance(r["wall_time_sec"], (int, float)):
                tot[r["system"]] = tot.get(r["system"], 0) + r["wall_time_sec"]
        L += ["", "Summed over the seeds actually used in each final ensemble: " +
              ", ".join(f"{k}: {v/60:.0f} min ({v:.0f} s)" for k, v in tot.items()) +
              ". Ours and OutfitTransformer trained on Modal cloud GPUs; CSA-Net trained on the laptop's MPS -- so wall-clock is "
              "only comparable within a row group, not across.", "",
              "Interpretation in `phase36_notes.md`.", ""]
        (C.PHASE_DIR / "inference_cost.md").write_text("\n".join(L))
    else:
        L += ["", "## Device-controlled comparison: every system forced onto CPU", "",
              f"{REPEATS} passes, median. Same machine, no MPS use anywhere.", "",
              "| System | members | precompute (s) | query + score (s) | total (s) | total per member (s) | per query (ms) |",
              "|---|---|---|---|---|---|---|"]
        for name, kind, _ in SYSTEMS:
            e = S[name]
            m = e["median"]
            L.append(f"| `{name}` | {e['n_members']} | {m['precompute_s']:.1f} | {m['query_and_score_s']:.1f} | {m['total_s']:.1f} | "
                     f"{e['total_per_member_s_median']:.1f} | {e['per_query_ms_median']:.2f} |")
        L.append("")
        with open(C.PHASE_DIR / "inference_cost.md", "a") as f:
            f.write("\n".join(L))
    print(f"saved inference_cost{tag}.json" + ("" if FORCE_CPU else " and inference_cost.md"))


if __name__ == "__main__":
    main()
