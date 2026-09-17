"""
Phase 36, step 1: re-score the five frozen, already-test-evaluated systems on the
CIR test benchmark, this time RETAINING the per-query rank of the true target
(every prior evaluator computed it and discarded it).

Reproduction guard: each system's aggregate Recall@10/30/50 must equal the value
its own phase reported (4 decimals; exact float where a JSON exists). Any mismatch
is a hard failure -- the phase stops and the mismatch is reported, not rounded
away. No selection decision is made here; see phase36_notes.md.
"""
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (sets PYTORCH_ENABLE_MPS_FALLBACK before importing torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402

SYSTEMS = [
    ("ours_solo", "ours", [C.PHASE27_MODELS / "text_only.pt"]),
    ("ours_ens", "ours", [C.PHASE28_MODELS / f"text_ensemble_seed{s}.pt" for s in [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]]),
    ("ot_solo", "ot", [C.PHASE32_DIR / "models" / "ot32_seed42.pt"]),
    ("ot_ens", "ot", [C.PHASE32_DIR / "models" / f"ot32_seed{s}.pt" for s in [42, 1, 2]]),
    ("csa_ens", "csa", [C.PHASE34_DIR / "models" / f"csanet34_seed{s}.pt" for s in [42, 1, 2]]),
]


def recall_from_ranks(ranks, ks=C.KS):
    valid = ranks > 0
    n_total = int(valid.sum())
    return {k: float((ranks[valid] <= k).mean()) for k in ks}, n_total


def main():
    t_start = time.perf_counter()
    print(f"device={C.DEVICE}")
    item_ids, idx, image_emb, base_repr = C.load_item_data()
    del image_emb
    pools, queries = C.load_benchmark()
    categories = np.array([q["category"] for q in queries])
    print(f"catalog {len(item_ids)} items, base_repr {tuple(base_repr.shape)} on {base_repr.device}; "
          f"{len(queries)} test queries")
    cats, cat_to_idx, item_cat = C.load_category_data()

    report = {"device": C.DEVICE, "n_queries": len(queries), "systems": {}, "all_reproduced": True}
    for name, kind, ckpts in SYSTEMS:
        for p in ckpts:
            assert p.exists(), p
        print(f"\n=== {name} ({len(ckpts)} member(s)) ===")
        if kind == "ours":
            system = C.OursSystem(name, ckpts)
        elif kind == "ot":
            system = C.OTSystem(name, ckpts)
        else:
            system = C.CSASystem(name, ckpts, cat_to_idx, item_cat)

        t_pre = system.precompute(base_repr)
        if kind == "ours":
            ranks, n_total, n_skipped, t_eval = system.cir_ranks(pools, queries, idx)
        elif kind == "ot":
            ranks, n_total, n_skipped, t_eval = system.cir_ranks(pools, queries, idx, base_repr)
        else:
            ranks, n_total, n_skipped, t_eval = system.cir_ranks(pools, queries, idx, base_repr)
        recall, n_valid = recall_from_ranks(ranks)
        assert n_valid == n_total, (n_valid, n_total)

        rec4 = C.RECORDED_4DP[name]
        match4 = {k: round(recall[k], 4) == rec4[k] for k in C.KS}
        exact = None
        if name in C.RECORDED_EXACT:
            exact = {k: abs(recall[k] - C.RECORDED_EXACT[name][k]) < 1e-9 for k in C.KS}
        ok = all(match4.values()) and (exact is None or all(exact.values())) and n_skipped == 0 and n_total == len(queries)
        report["all_reproduced"] &= ok

        np.savez_compressed(C.DATA_DIR / f"per_query_ranks_{name}.npz",
                            qidx=np.arange(len(queries), dtype=np.int32), rank=ranks.astype(np.int32),
                            category=categories)
        entry = {
            "kind": kind, "n_members": len(ckpts), "n_params_per_member": system.n_params_per_member,
            "recall": {str(k): recall[k] for k in C.KS}, "recorded_4dp": {str(k): rec4[k] for k in C.KS},
            "match_4dp": {str(k): bool(v) for k, v in match4.items()},
            "match_exact": None if exact is None else {str(k): bool(v) for k, v in exact.items()},
            "n_total": n_total, "n_skipped": n_skipped, "reproduced": bool(ok),
            "timing_s": {"precompute": t_pre, "query_and_score": t_eval, "total": t_pre + t_eval,
                         **({"csa_pool_candidate_precompute_within_loop": system.precompute_in_loop_s} if kind == "csa" else {})},
            "peak_rss_mb_so_far": C.peak_rss_mb(), "mps_allocated_mb_after": C.mps_allocated_mb(),
        }
        report["systems"][name] = entry
        print(f"  R@10/30/50 = {recall[10]:.4f}/{recall[30]:.4f}/{recall[50]:.4f}  recorded {rec4[10]}/{rec4[30]}/{rec4[50]}"
              f"  match4={all(match4.values())} exact={None if exact is None else all(exact.values())}"
              f"  n_total={n_total} n_skipped={n_skipped}")
        print(f"  timing: precompute {t_pre:.1f}s, query+score {t_eval:.1f}s; n_params/member={system.n_params_per_member}")
        if not ok:
            print("  !!! REPRODUCTION FAILURE -- see report")

        del system
        gc.collect()
        if C.DEVICE == "mps":
            torch.mps.empty_cache()

    report["wall_time_s"] = time.perf_counter() - t_start
    with open(C.DATA_DIR / "reproduction_check.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nall_reproduced={report['all_reproduced']}  wall={report['wall_time_s']:.0f}s")
    print(f"saved {C.DATA_DIR / 'reproduction_check.json'}")
    if not report["all_reproduced"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
