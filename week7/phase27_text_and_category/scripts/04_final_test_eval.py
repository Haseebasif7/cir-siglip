"""
Phase 27, final step: evaluate all three trained variants on the actual test
CIR benchmark, exactly once, alongside week 6's image-only baseline (single
model and the 10-member ensemble) for direct comparison. No decisions are
made from this script's output -- early stopping already happened via
validation Recall@10 inside each Modal training run; this is the one-time
report of what those already-locked-in checkpoints score on held-out data.

Mirrors modal_app.py's evaluate_recall_gpu mechanism exactly (same
post-projection pooling for use_category="none", same pre-projection
pooling + category concat for the two category-conditioned variants) so the
numbers here are computed the same way they were computed during training's
own validation checks -- just against cir_benchmark.json instead of
cir_val_benchmark.json.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"

IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = BASE_DIR / "data" / "text_embeddings.npz"
CATEGORY_INDEX_NPZ = BASE_DIR / "data" / "category_index.npz"
CATEGORY_TEXT_NPZ = BASE_DIR / "data" / "category_text_embeddings.npz"
TEST_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"
RESULTS_JSON = BASE_DIR / "data" / "train_variants_results.json"
OUT_MD = BASE_DIR / "results_table.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CATEGORY_LIST = [
    "accessories", "all-body", "bags", "bottoms", "hats", "jewellery",
    "outerwear", "scarves", "shoes", "sunglasses", "tops",
]
CATEGORY_TO_IDX = {c: i for i, c in enumerate(CATEGORY_LIST)}

# Week 6 reference numbers, hardcoded per this project's own convention of
# citing already-locked-in prior results rather than re-deriving them.
WEEK6_SINGLE = {10: 0.1505, 30: 0.2740, 50: 0.3503}   # phase 25 scaled single model
WEEK6_ENSEMBLE = {10: 0.1767, 30: 0.3054, 50: 0.3828}  # phase 26 10-model ensemble


class ProjectionHeadGeneral(nn.Module):
    def __init__(self, in_dim, hidden_dims=(1024,), out_dim=128, dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


def load_shared_data():
    img_npz = np.load(IMAGE_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img_npz["item_ids"]]
    id_to_gidx = {a: i for i, a in enumerate(item_ids)}
    image_emb = img_npz["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)

    text_npz = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=True)
    assert [str(a) for a in text_npz["item_ids"]] == item_ids
    text_emb = text_npz["embeddings"].astype(np.float32)

    cat_npz = np.load(CATEGORY_INDEX_NPZ, allow_pickle=True)
    assert [str(a) for a in cat_npz["item_ids"]] == item_ids
    category_idx = cat_npz["category_idx"].astype(np.int64)

    cat_text_npz = np.load(CATEGORY_TEXT_NPZ, allow_pickle=True)
    assert [str(c) for c in cat_text_npz["categories"]] == CATEGORY_LIST
    cat_phrase_vecs = cat_text_npz["embeddings"].astype(np.float32)
    cat_phrase_vecs = cat_phrase_vecs / np.linalg.norm(cat_phrase_vecs, axis=1, keepdims=True)

    image_t = torch.tensor(image_emb, device=DEVICE)
    text_t = torch.tensor(text_emb, device=DEVICE)
    base_repr = F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)
    category_idx_t = torch.tensor(category_idx, device=DEVICE)
    cat_phrase_t = torch.tensor(cat_phrase_vecs, device=DEVICE)

    return item_ids, id_to_gidx, base_repr, category_idx_t, cat_phrase_t


def evaluate_variant(name, use_category, base_repr, id_to_gidx, category_idx_t, cat_phrase_lookup,
                      pools, queries, ks=(10, 30, 50)):
    ckpt_path = BASE_DIR / "models" / f"{name}.pt"
    with open(RESULTS_JSON) as f:
        results = {r["config"]["name"]: r for r in json.load(f)}
    cfg = results[name]["config"]
    in_dim = 1536 if use_category == "none" else 2304

    model = ProjectionHeadGeneral(in_dim=in_dim, hidden_dims=cfg["hidden_dims"],
                                   out_dim=cfg["out_dim"]).to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    category_table = None
    if use_category == "learned":
        cat_ckpt = BASE_DIR / "models" / f"{name}_category_table.pt"
        category_table = nn.Embedding(len(CATEGORY_LIST), 768).to(DEVICE)
        category_table.load_state_dict(torch.load(cat_ckpt, map_location=DEVICE))
        category_table.eval()

    def category_vecs_for(cat_idx_batch):
        if category_table is not None:
            v = category_table(cat_idx_batch)
        else:
            v = cat_phrase_lookup[cat_idx_batch]
        return F.normalize(v, p=2, dim=-1)

    if use_category == "none":
        candidate_repr = base_repr
    else:
        candidate_repr = F.pad(base_repr, (0, 768))

    with torch.no_grad():
        proj_candidates = model(candidate_repr).cpu().numpy()

    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)

    hits = {k: 0 for k in ks}
    n_total, n_skipped = 0, 0

    for cat, qidxs in by_cat.items():
        if cat not in pools:
            n_skipped += len(qidxs)
            continue
        pool_ids = pools[cat]
        pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
        pool_idx = [id_to_gidx[i] for i in pool_ids]
        pool_emb = proj_candidates[pool_idx]

        kept_qidx, kept_ctx_idx, tpos = [], [], []
        for qi in qidxs:
            q = queries[qi]
            ctx_idx = [id_to_gidx[i] for i in q["query_items"] if i in id_to_gidx]
            if not ctx_idx or q["target_item"] not in pool_pos:
                n_skipped += 1
                continue
            kept_qidx.append(qi)
            kept_ctx_idx.append(ctx_idx)
            tpos.append(pool_pos[q["target_item"]])
        if not kept_qidx:
            continue

        if use_category == "none":
            qvecs = []
            for ctx_idx in kept_ctx_idx:
                v = proj_candidates[ctx_idx].mean(axis=0)
                n = np.linalg.norm(v)
                qvecs.append(v / n if n > 0 else v)
            qmat = np.stack(qvecs)
        else:
            target_cat_idx = CATEGORY_TO_IDX[cat]
            with torch.no_grad():
                pooled_raws = [F.normalize(base_repr[torch.tensor(ctx_idx, device=DEVICE)].mean(dim=0),
                                            p=2, dim=-1) for ctx_idx in kept_ctx_idx]
                pooled_raw_t = torch.stack(pooled_raws)
                cat_idx_t = torch.full((len(kept_qidx),), target_cat_idx, dtype=torch.long, device=DEVICE)
                cat_vecs_t = category_vecs_for(cat_idx_t)
                q_repr = F.normalize(torch.cat([pooled_raw_t, cat_vecs_t], dim=-1), p=2, dim=-1)
                qmat = model(q_repr).cpu().numpy()

        tpos = np.array(tpos)
        sims = qmat @ pool_emb.T
        tsims = sims[np.arange(len(kept_qidx)), tpos]
        ranks = (sims >= tsims[:, None]).sum(axis=1)
        n_total += len(kept_qidx)
        for k in ks:
            hits[k] += int((ranks <= k).sum())

    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped


def main():
    print("Loading shared image/text/category data...")
    item_ids, id_to_gidx, base_repr, category_idx_t, cat_phrase_lookup = load_shared_data()

    with open(TEST_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]

    variants = [
        ("text_only", "none"),
        ("text_category_learned", "learned"),
        ("text_category_siglip_phrase", "siglip_phrase"),
    ]

    rows = []
    for name, use_category in variants:
        print(f"Evaluating {name} (use_category={use_category}) on the TEST benchmark...")
        recall, n_total, n_skipped = evaluate_variant(
            name, use_category, base_repr, id_to_gidx, category_idx_t, cat_phrase_lookup, pools, queries)
        print(f"  {recall} (n={n_total}, skip={n_skipped})")
        rows.append((name, recall))

    lines = [
        "# Phase 27: Results Table -- Test Benchmark, Touched Once\n",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 |",
        "|---|---|---|---|",
        f"| Week 6 image-only, single model (phase 25) | {WEEK6_SINGLE[10]:.4f} | {WEEK6_SINGLE[30]:.4f} | {WEEK6_SINGLE[50]:.4f} |",
        f"| Week 6 image-only, 10-model ensemble (phase 26) | {WEEK6_ENSEMBLE[10]:.4f} | {WEEK6_ENSEMBLE[30]:.4f} | {WEEK6_ENSEMBLE[50]:.4f} |",
    ]
    label_map = {
        "text_only": "Phase 27: text only (single model)",
        "text_category_learned": "Phase 27: text + category, learned table (single model)",
        "text_category_siglip_phrase": "Phase 27: text + category, SigLIP phrase (single model)",
    }
    for name, recall in rows:
        lines.append(f"| {label_map[name]} | {recall[10]:.4f} | {recall[30]:.4f} | {recall[50]:.4f} |")
    lines.append("")
    lines.append("All phase 27 rows are single-seed runs (per the brief: this phase is an initial "
                  "signal check, not a final result). Week 6 rows are cited from phase 25/26's own "
                  "already-reported, already-finalized numbers, not re-derived here.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()
