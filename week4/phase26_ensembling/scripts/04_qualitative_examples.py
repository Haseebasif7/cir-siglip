"""
Phase 26 follow-up: qualitative, by-eye examples of the final 10-member
ensemble (the current best model, see phase26_notes.md) doing the actual
complementary item retrieval task on real test-benchmark queries.

Not a required phase26 deliverable -- generated on request to show real
retrieval behavior visually, reusing this phase's own checkpoints and
score-averaging evaluator (cir_eval_ensemble.py) rather than re-deriving
anything. Picks a small, honestly mixed set of queries (a majority of hits,
at least one miss) across different categories -- Recall@10 on the full test
benchmark is 0.1767, so most queries do NOT land the true target in the top
10; cherry-picking only hits would misrepresent typical behavior.
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

OUT_DIR = BASE_DIR / "qualitative_examples"
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
    seeds_used = best_cfg["best_seeds"]  # the winning all-10 same-architecture composition

    main_by_seed = {
        r["config"]["seed"]: r for r in results
        if r["config"]["name"].startswith("ensemble_seed") or r["config"]["name"] == "final_scaled"
    }

    print("Loading base SigLIP embeddings...")
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    base_emb = data["embeddings"].astype(np.float32)
    base_emb = base_emb / np.linalg.norm(base_emb, axis=1, keepdims=True)
    base_emb_t = torch.tensor(base_emb, device=DEVICE)

    print(f"Projecting all {len(seeds_used)} ensemble members...")
    projections = []
    for seed in seeds_used:
        r = main_by_seed[seed]
        ckpt = REPO_ROOT / r["local_checkpoint"]
        proj = project_all(ckpt, r["config"]["hidden_dims"], r["config"]["out_dim"], base_emb_t)
        projections.append(proj)

    return item_ids, base_emb, projections, seeds_used


def query_vector(embeddings, idx, query_items):
    item_idx = [idx[i] for i in query_items if i in idx]
    v = embeddings[item_idx].mean(axis=0)
    return v / np.linalg.norm(v)


def ensemble_scores_for_query(projections, idx, query_items, pool_idx):
    """Average per-member cosine similarity scores (score-averaging, same
    principle as cir_eval_ensemble.py's evaluate_recall_ensemble)."""
    sims_sum = None
    for emb in projections:
        qv = query_vector(emb, idx, query_items)
        pool_emb = emb[pool_idx]
        sims = pool_emb @ qv
        sims_sum = sims if sims_sum is None else sims_sum + sims
    return sims_sum / len(projections)


def pick_examples(pools, queries, item_ids, projections, idx):
    """Deterministically picks a small, honest mix of queries across
    categories: mostly hits (true target lands in the top TOP_K), at least
    one clear miss, so the sample isn't a cherry-picked highlight reel."""
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
        for qi in cat_qidxs[:40]:  # cap work per category
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


def item_title(meta, item_id):
    m = meta.get(item_id, {})
    return m.get("url_name") or m.get("title") or item_id


def build_figure(query_record, item_ids, meta, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    qi, cat, rank, pool_idx, pool_ids, scores, order = query_record
    q = QUERIES[qi]
    context_items = q["query_items"]
    target_item = q["target_item"]

    n_ctx = len(context_items)
    n_cols = max(n_ctx + 1, TOP_K)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.6 * n_cols, 6.0))

    # Row 0: query context items, then the true target (ground truth), highlighted in blue.
    for c, item_id in enumerate(context_items):
        ax = axes[0, c]
        _show(ax, IMAGES_DIR / f"{item_id}.jpg", item_title(meta, item_id)[:26], fontsize=7)
    ax = axes[0, n_ctx]
    _show(ax, IMAGES_DIR / f"{target_item}.jpg", "TRUE TARGET\n" + item_title(meta, target_item)[:26],
          fontsize=7, border="blue")
    for c in range(n_ctx + 1, n_cols):
        axes[0, c].axis("off")

    # Row 1: ensemble's top-K retrieved candidates from this category's pool,
    # green border if it's the true target, otherwise plain.
    for c in range(TOP_K):
        ax = axes[1, c]
        j = order[c]
        item_id = pool_ids[j]
        is_hit = item_id == target_item
        border = "green" if is_hit else None
        label = f"#{c+1} score={scores[j]:.3f}\n{item_title(meta, item_id)[:26]}"
        _show(ax, IMAGES_DIR / f"{item_id}.jpg", label, fontsize=7, border=border)
    for c in range(TOP_K, n_cols):
        axes[1, c].axis("off")

    axes[0, 0].text(-0.35, 0.5, "Query outfit\n(context items)\n+ true target",
                     transform=axes[0, 0].transAxes, fontsize=8, ha="right", va="center", weight="bold")
    axes[1, 0].text(-0.35, 0.5, f"Ensemble top {TOP_K}\nretrieved from\n{cat} pool (n={len(pool_ids)})",
                     transform=axes[1, 0].transAxes, fontsize=8, ha="right", va="center", weight="bold")

    verdict = f"HIT -- true target ranked #{rank} of {len(pool_ids)}" if rank <= TOP_K \
        else f"MISS -- true target ranked #{rank} of {len(pool_ids)} (outside top {TOP_K})"
    fig.suptitle(f"Category: {cat}    {verdict}", fontsize=11,
                 color=("darkgreen" if rank <= TOP_K else "firebrick"))
    plt.tight_layout(rect=[0.06, 0, 1, 0.95])
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"Saved {out_path}")


def _show(ax, img_path, label, fontsize=7, border=None):
    from PIL import Image
    try:
        ax.imshow(Image.open(img_path).convert("RGB"))
    except Exception:
        ax.text(0.5, 0.5, "[missing]", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=fontsize)
    if border:
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(4)
            spine.set_visible(True)
    else:
        for spine in ax.spines.values():
            spine.set_visible(False)


def main():
    global QUERIES
    OUT_DIR.mkdir(exist_ok=True)

    with open(TEST_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]
    QUERIES = queries

    item_ids, base_emb, projections, seeds_used = load_ensemble_projections()
    idx = {a: i for i, a in enumerate(item_ids)}

    with open(METADATA_JSON) as f:
        meta = json.load(f)

    print(f"Ensemble members used: seeds {seeds_used}")
    print("Selecting examples (honest mix: mostly hits, at least one miss)...")
    examples = pick_examples(pools, queries, item_ids, projections, idx)
    print(f"Selected {len(examples)} examples: "
          f"{sum(1 for e in examples if e[2] <= TOP_K)} hits, "
          f"{sum(1 for e in examples if e[2] > TOP_K)} misses")

    for i, record in enumerate(examples):
        qi, cat, rank, *_ = record
        out_path = OUT_DIR / f"example_{i+1}_{cat}_{'hit' if rank <= TOP_K else 'miss'}.png"
        build_figure(record, item_ids, meta, out_path)


if __name__ == "__main__":
    main()
