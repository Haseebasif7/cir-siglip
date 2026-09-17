"""
Phase 14b, step 3: evaluate on the exact same CIR benchmark used throughout
this project (`week4/phase12_controllable_modes/data/cir_benchmark.json`),
same protocol as phase 14's own `03_cir_eval.py` -- unchanged logic, just
repointed at this phase's candidate features and checkpoint.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OutfitTransformerSigLIP

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE12_DIR = BASE_DIR.parent / "phase12_controllable_modes"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
BENCHMARK_JSON = PHASE12_DIR / "data" / "cir_benchmark.json"
CANDIDATE_NPZ = BASE_DIR / "embeddings" / "outfit_transformer_candidate_features.npz"
CHECKPOINT_PT = BASE_DIR / "models" / "outfit_transformer_siglip_category_neg_best.pt"
OUT_JSON = BASE_DIR / "data" / "cir_results.json"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
QUERY_BATCH = 1024


def load_siglip():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"].astype(np.float32)
    idx = {a: i for i, a in enumerate(item_ids)}
    return idx, embeddings


def compute_query_embeddings(model, siglip_idx, siglip_emb, query_item_lists, device):
    outs = []
    for start in range(0, len(query_item_lists), QUERY_BATCH):
        chunk = query_item_lists[start:start + QUERY_BATCH]
        lengths = [len(q) for q in chunk]
        Lmax = max(lengths)
        B = len(chunk)
        ctx = np.zeros((B, Lmax, siglip_emb.shape[1]), dtype=np.float32)
        mask = np.ones((B, Lmax), dtype=bool)
        for i, items in enumerate(chunk):
            for j, it in enumerate(items):
                ctx[i, j] = siglip_emb[siglip_idx[it]]
                mask[i, j] = False
        ctx_t = torch.tensor(ctx, device=device)
        mask_t = torch.tensor(mask, device=device)
        with torch.no_grad():
            B_, L_, D_ = ctx_t.shape
            tokens = model.encode_item_tokens(ctx_t.view(B_ * L_, D_)).view(B_, L_, -1)
            q_emb = model.embed_query(tokens, mask_t)
        outs.append(q_emb.cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate_recall(pools, queries, candidate_idx, candidate_emb, model, siglip_idx, siglip_emb, device, ks=(10, 30, 50)):
    hits = {k: 0 for k in ks}
    n_total, n_skipped = 0, 0

    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)

    for cat, qidxs in by_cat.items():
        pool_ids = pools[cat]
        pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
        pool_idx = [candidate_idx[i] for i in pool_ids]
        pool_emb = candidate_emb[pool_idx]

        valid_qidxs, query_item_lists, target_positions = [], [], []
        for qi in qidxs:
            q = queries[qi]
            items = [i for i in q["query_items"] if i in siglip_idx]
            if not items or q["target_item"] not in pool_pos:
                n_skipped += 1
                continue
            valid_qidxs.append(qi)
            query_item_lists.append(items)
            target_positions.append(pool_pos[q["target_item"]])
        if not query_item_lists:
            continue

        query_mat = compute_query_embeddings(model, siglip_idx, siglip_emb, query_item_lists, device)
        sims = query_mat @ pool_emb.T
        target_positions = np.array(target_positions)
        target_sims = sims[np.arange(len(query_item_lists)), target_positions]
        ranks = (sims >= target_sims[:, None]).sum(axis=1)

        n_total += len(query_item_lists)
        for k in ks:
            hits[k] += int((ranks <= k).sum())

    recall = {k: hits[k] / n_total for k in ks} if n_total else {k: 0.0 for k in ks}
    return recall, n_total, n_skipped


def main():
    with open(BENCHMARK_JSON) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]
    print(f"Loaded benchmark: {len(queries)} queries, {len(pools)} category pools")

    siglip_idx, siglip_emb = load_siglip()

    cand = np.load(CANDIDATE_NPZ, allow_pickle=True)
    candidate_item_ids = [str(a) for a in cand["item_ids"]]
    candidate_idx = {a: i for i, a in enumerate(candidate_item_ids)}
    candidate_emb = cand["embeddings"].astype(np.float32)
    print(f"Loaded {candidate_emb.shape} candidate embeddings")

    model = OutfitTransformerSigLIP().to(DEVICE).eval()
    state_dict = torch.load(CHECKPOINT_PT, map_location=DEVICE)
    model.load_state_dict(state_dict)

    recall, n_total, n_skipped = evaluate_recall(
        pools, queries, candidate_idx, candidate_emb, model, siglip_idx, siglip_emb, DEVICE
    )
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"Recall@10={recall[10]:.4f} Recall@30={recall[30]:.4f} Recall@50={recall[50]:.4f}")

    with open(OUT_JSON, "w") as f:
        json.dump({"recall": recall, "n_total": n_total, "n_skipped": n_skipped}, f, indent=2)
    print(f"Saved to {OUT_JSON}")


if __name__ == "__main__":
    main()
