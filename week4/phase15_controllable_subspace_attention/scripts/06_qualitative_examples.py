"""
Phase 15, step 8: qualitative examples for the better-performing of the two
trained versions (continuous vs. discrete, decided by ablation_results.md).
Picks 5-6 example queries across different categories, shows top-5 retrieved
items at alpha in {0.0, 0.33, 0.67, 1.0}, mirroring phase 12d's own
qualitative sweep format.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import CSANetSigLIPControllable
from importlib import import_module

sweep = import_module("03_cir_eval_sweep")

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"

BENCHMARK_JSON = BASE_DIR.parent / "phase12_controllable_modes" / "data" / "cir_benchmark.json"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA_JSON = PHASE13_DIR / "data" / "training_data.json"

DEVICE = "cpu"
QUAL_ALPHAS = [0.0, 0.33, 0.67, 1.0]
TOP_N = 5
SEED = 7


def main(checkpoint_path, out_dir, n_examples=6):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(TRAINING_DATA_JSON) as f:
        categories = json.load(f)["categories"]
    item_cat_lookup = sweep.build_item_cat_lookup(categories)

    with open(BENCHMARK_JSON) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]

    item_ids, raw_embeddings = sweep.load_features()
    idx = {a: i for i, a in enumerate(item_ids)}

    model = CSANetSigLIPControllable().to(DEVICE).eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    with torch.no_grad():
        base_features = model.encode_feature(torch.tensor(raw_embeddings, device=DEVICE)).numpy()

    rng = random.Random(SEED)
    by_cat = {}
    for qi, q in enumerate(queries):
        by_cat.setdefault(q["category"], []).append(qi)
    cats_available = [c for c in categories if by_cat.get(c)]
    chosen_cats = rng.sample(cats_available, min(n_examples, len(cats_available)))

    examples = []
    for cat in chosen_cats:
        qi = rng.choice(by_cat[cat])
        q = queries[qi]
        example = {"category": cat, "query_items": q["query_items"], "target_item": q["target_item"], "by_alpha": {}}

        pool_ids = pools[cat]
        pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
        pool_gidx = [idx[i] for i in pool_ids if i in idx]
        cat_i = categories.index(cat)
        x_pool = torch.tensor(base_features[pool_gidx], device=DEVICE)
        cat_t_fixed = torch.zeros(len(pool_gidx), len(categories), device=DEVICE)
        cat_t_fixed[:, cat_i] = 1.0

        ctx_items = [i for i in q["query_items"] if i in idx]
        ctx_gidx = [idx[i] for i in ctx_items]
        x_ctx = torch.tensor(base_features[ctx_gidx], device=DEVICE)
        eye = torch.eye(len(categories), device=DEVICE)

        with torch.no_grad():
            for alpha in QUAL_ALPHAS:
                cand_all = model.all_as_candidate_embeddings_from_feature(x_pool, cat_t_fixed, alpha)
                dist_sum = torch.zeros(len(pool_gidx), device=DEVICE)
                for ci, item_id in enumerate(ctx_items):
                    c_i = categories.index(item_cat_lookup[item_id])
                    f_ctx = model.embed_from_feature(x_ctx[ci:ci + 1], eye[c_i:c_i + 1], cat_t_fixed[0:1], alpha)
                    dist_sum += ((f_ctx - cand_all[:, c_i, :]) ** 2).sum(dim=-1)
                dist_avg = (dist_sum / len(ctx_items)).numpy()
                top_local = np.argsort(dist_avg)[:TOP_N]
                example["by_alpha"][str(alpha)] = {
                    "top_items": [pool_ids[i] for i in top_local],
                    "target_rank": int((dist_avg <= dist_avg[pool_pos[q["target_item"]]]).sum()) if q["target_item"] in pool_pos else None,
                }
        examples.append(example)
        print(f"category={cat} query_items={ctx_items} target={q['target_item']}")
        for alpha in QUAL_ALPHAS:
            print(f"  alpha={alpha}: top5={example['by_alpha'][str(alpha)]['top_items']} "
                  f"target_rank={example['by_alpha'][str(alpha)]['target_rank']}")

    with open(out_dir / "qualitative_examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved {out_dir / 'qualitative_examples.json'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=str(BASE_DIR / "qualitative_examples"))
    args = parser.parse_args()
    main(args.checkpoint, args.out)
