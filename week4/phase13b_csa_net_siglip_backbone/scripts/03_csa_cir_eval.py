"""
Phase 13b, step 4: evaluate the trained CSA-Net-on-frozen-SigLIP checkpoint
under this project's own CIR harness. Identical scoring logic to phase 13's
04_csa_cir_eval.py (per-context-item average pairwise distance, eq. 5 of the
paper -- see that file's docstring for the full explanation of why a
single-vector scoring scheme can't be used here), just pointed at this
phase's own checkpoint/features/model class.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CSANetSigLIP

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE12_DIR = BASE_DIR.parent / "phase12_controllable_modes"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"
BENCHMARK_JSON = PHASE12_DIR / "data" / "cir_benchmark.json"
FEATURES_NPZ = BASE_DIR / "embeddings" / "csa_siglip_base_features.npz"
CHECKPOINT_PT = BASE_DIR / "models" / "csa_net_siglip_best.pt"
TRAINING_DATA_JSON = PHASE13_DIR / "data" / "training_data.json"  # reused, backbone-independent

OUT_JSON = BASE_DIR / "data" / "csa_siglip_cir_results.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_features():
    data = np.load(FEATURES_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    return item_ids, embeddings


def evaluate_csa_recall(pools, queries, item_ids, base_features, model, categories, item_cat_lookup, ks=(10, 30, 50)):
    idx = {a: i for i, a in enumerate(item_ids)}
    cat_to_i = {c: i for i, c in enumerate(categories)}
    hits = {k: 0 for k in ks}
    n_total, n_skipped = 0, 0

    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)

    eye = torch.eye(len(categories), device=DEVICE)

    with torch.no_grad():
        for cat, qidxs in by_cat.items():
            if cat not in pools:
                n_skipped += len(qidxs)
                continue
            pool_ids = pools[cat]
            pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
            pool_gidx = [idx[i] for i in pool_ids if i in idx]
            if len(pool_gidx) != len(pool_ids):
                n_skipped += len(qidxs)
                continue
            x_pool = torch.tensor(base_features[pool_gidx], device=DEVICE)
            cat_t_fixed = torch.zeros(len(pool_gidx), len(categories), device=DEVICE)
            cat_t_fixed[:, cat_to_i[cat]] = 1.0
            cand_all = model.all_as_candidate_embeddings_from_feature(x_pool, cat_t_fixed)

            for qi in qidxs:
                q = queries[qi]
                ctx_items = [i for i in q["query_items"] if i in idx]
                if not ctx_items or q["target_item"] not in pool_pos:
                    n_skipped += 1
                    continue
                ctx_gidx = [idx[i] for i in ctx_items]
                x_ctx = torch.tensor(base_features[ctx_gidx], device=DEVICE)

                dist_sum = torch.zeros(len(pool_gidx), device=DEVICE)
                for ci, item_id in enumerate(ctx_items):
                    c_i = item_cat_lookup[item_id]
                    c_idx = cat_to_i[c_i]
                    f_ctx = model.embed_from_feature(
                        x_ctx[ci:ci + 1],
                        eye[c_idx:c_idx + 1],
                        cat_t_fixed[0:1],
                    )
                    cand_slice = cand_all[:, c_idx, :]
                    d = ((f_ctx - cand_slice) ** 2).sum(dim=-1)
                    dist_sum += d

                dist_avg = (dist_sum / len(ctx_items)).cpu().numpy()
                target_pos = pool_pos[q["target_item"]]
                target_dist = dist_avg[target_pos]
                rank = int((dist_avg <= target_dist).sum())
                n_total += 1
                for k in ks:
                    if rank <= k:
                        hits[k] += 1

    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped


def main():
    with open(TRAINING_DATA_JSON) as f:
        td = json.load(f)
    categories = td["categories"]
    item_cat_lookup = {}
    for split_key in ("train_items_by_category", "val_items_by_category"):
        for cat, items in td[split_key].items():
            for i in items:
                item_cat_lookup[i] = cat
    phase9_dir = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
    with open(phase9_dir / "data" / "polyvore_raw" / "polyvore_item_metadata.json") as f:
        meta = json.load(f)
    for item_id, v in meta.items():
        cat = v.get("semantic_category")
        if item_id not in item_cat_lookup and cat in categories:
            item_cat_lookup[item_id] = cat

    with open(BENCHMARK_JSON) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]

    item_ids, base_features = load_features()
    print(f"Loaded {len(item_ids)} item base features.")

    model = CSANetSigLIP().to(DEVICE).eval()
    state_dict = torch.load(CHECKPOINT_PT, map_location=DEVICE)
    model.load_state_dict(state_dict)

    recall, n_total, n_skipped = evaluate_csa_recall(pools, queries, item_ids, base_features, model, categories, item_cat_lookup)
    print(f"CSA-Net-on-SigLIP: n_total={n_total} n_skipped={n_skipped}")
    print(recall)

    with open(OUT_JSON, "w") as f:
        json.dump({"recall": recall, "n_total": n_total, "n_skipped": n_skipped}, f, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
