"""
Export the same honestly-mixed example set 04_qualitative_examples.py picked
(same SEED, same selection logic, duplicated here rather than imported since
04_'s filename can't be imported as a module) as a JSON manifest of item
ids, titles, scores, and image file paths -- consumed by a separate
HTML-gallery builder so the images can be shown as real, native,
theme-aware HTML rather than static matplotlib grids. Not a phase26
deliverable, generated on request.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
IMAGES_DIR = PHASE9_DIR / "data" / "images"
METADATA_JSON = PHASE9_DIR / "data" / "polyvore_raw" / "polyvore_item_metadata.json"
TEST_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"
RESULTS_JSON = BASE_DIR / "data" / "train_seeds_results.json"
BEST_CONFIG_JSON = BASE_DIR / "data" / "best_ensemble_config.json"

OUT_JSON = BASE_DIR / "qualitative_examples" / "manifest.json"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K = 5
N_EXAMPLES = 6
SEED = 7


class ProjectionHeadGeneral(nn.Module):
    def __init__(self, in_dim=768, hidden_dims=(1024,), out_dim=128, dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


def project_all(ckpt_path, hidden_dims, out_dim, base_emb_t):
    model = ProjectionHeadGeneral(hidden_dims=hidden_dims, out_dim=out_dim).to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    with torch.no_grad():
        proj = model(base_emb_t).cpu().numpy()
    return proj


def load_ensemble_projections():
    with open(RESULTS_JSON) as f:
        results = json.load(f)
    with open(BEST_CONFIG_JSON) as f:
        best_cfg = json.load(f)
    seeds_used = best_cfg["best_seeds"]

    main_by_seed = {
        r["config"]["seed"]: r for r in results
        if r["config"]["name"].startswith("ensemble_seed") or r["config"]["name"] == "final_scaled"
    }

    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    base_emb = data["embeddings"].astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb, axis=1, keepdims=True)
    base_emb_t = torch.tensor(base_emb, device=DEVICE)

    projections = []
    for seed in seeds_used:
        r = main_by_seed[seed]
        ckpt = REPO_ROOT / r["local_checkpoint"]
        proj = project_all(ckpt, r["config"]["hidden_dims"], r["config"]["out_dim"], base_emb_t)
        projections.append(proj)

    return item_ids, projections, seeds_used


def query_vector(embeddings, idx, query_items):
    item_idx = [idx[i] for i in query_items if i in idx]
    v = embeddings[item_idx].mean(axis=0)
    return v / np.linalg.norm(v)


def ensemble_scores_for_query(projections, idx, query_items, pool_idx):
    sims_sum = None
    for emb in projections:
        qv = query_vector(emb, idx, query_items)
        pool_emb = emb[pool_idx]
        sims = pool_emb @ qv
        sims_sum = sims if sims_sum is None else sims_sum + sims
    return sims_sum / len(projections)


def pick_examples(pools, queries, item_ids, projections, idx):
    rng = random.Random(SEED)
    by_cat = {}
    for qi, q in enumerate(queries):
        by_cat.setdefault(q["category"], []).append(qi)

    cats = [c for c in pools if c in by_cat]
    rng.shuffle(cats)

    hits, misses = [], []
    for cat in cats:
        pool_ids = pools[cat]
        pool_idx = [idx[i] for i in pool_ids]
        pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
        cat_qidxs = by_cat[cat][:]
        rng.shuffle(cat_qidxs)
        for qi in cat_qidxs[:40]:
            q = queries[qi]
            if q["target_item"] not in pool_pos:
                continue
            scores = ensemble_scores_for_query(projections, idx, q["query_items"], pool_idx)
            order = np.argsort(-scores)
            rank = int(np.where(order == pool_pos[q["target_item"]])[0][0]) + 1
            record = (qi, cat, rank, pool_idx, pool_ids, scores, order)
            if rank <= TOP_K and len(hits) < N_EXAMPLES - 1:
                hits.append(record)
                break
            elif rank > TOP_K and len(misses) < 1:
                misses.append(record)
                break
        if len(hits) >= N_EXAMPLES - 1 and len(misses) >= 1:
            break

    return hits + misses


def item_info(meta, item_id):
    m = meta.get(item_id, {})
    title = m.get("url_name") or m.get("title") or item_id
    img = f"{item_id}.jpg"
    return {"id": item_id, "title": title, "image": img,
            "image_path": str(IMAGES_DIR / img)}


def main():
    with open(TEST_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]

    item_ids, projections, seeds_used = load_ensemble_projections()
    idx = {a: i for i, a in enumerate(item_ids)}

    with open(METADATA_JSON) as f:
        meta = json.load(f)

    examples = pick_examples(pools, queries, item_ids, projections, idx)

    manifest = {"seeds_used": seeds_used, "examples": []}
    for qi, cat, rank, pool_idx, pool_ids, scores, order in examples:
        q = queries[qi]
        context = [item_info(meta, i) for i in q["query_items"]]
        target = item_info(meta, q["target_item"])
        top5 = []
        for c in range(TOP_K):
            j = order[c]
            item_id = pool_ids[j]
            info = item_info(meta, item_id)
            info["score"] = round(float(scores[j]), 4)
            info["is_hit"] = (item_id == q["target_item"])
            top5.append(info)
        manifest["examples"].append({
            "category": cat,
            "rank": rank,
            "pool_size": len(pool_ids),
            "is_hit": rank <= TOP_K,
            "context": context,
            "target": target,
            "top5": top5,
        })

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {OUT_JSON}")
    print(f"{sum(1 for e in manifest['examples'] if e['is_hit'])} hits, "
          f"{sum(1 for e in manifest['examples'] if not e['is_hit'])} misses")


if __name__ == "__main__":
    main()
