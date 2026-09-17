"""
Phase 31, step 0 (A0): evaluate phase 14b's EXISTING checkpoint
(week4/phase14b_outfittransformer_category_negatives/models_random_negatives/
outfit_transformer_siglip_random_neg_best.pt) on the validation benchmark
(week4/phase23_hyperparameter_tuning/data/cir_val_benchmark.json) -- NO
retraining, NO test-benchmark touch. This is the val-side anchor point the
whole phase 31 progression table needs: phase 14b's own reported number
(0.0588/0.1286/0.1809) was measured against cir_benchmark.json (the TEST
set, per the benchmark-discipline correction -- see checkpoint_selection_
check.md), so there is otherwise no val-benchmark number to compare A1/A2/A3
against. Runs locally on MPS/CPU, matching phase 14b's own evaluation setup.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OutfitTransformerSigLIP

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE14B_DIR = REPO_ROOT / "week4/phase14b_outfittransformer_category_negatives"
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"

EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"
CHECKPOINT_PT = PHASE14B_DIR / "models_random_negatives" / "outfit_transformer_siglip_random_neg_best.pt"
OUT_MD = Path(__file__).resolve().parent.parent / "data" / "reference_point_A0.json"

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


def evaluate_recall(pools, queries, siglip_idx, candidate_emb, model, siglip_emb, device, ks=(10, 30, 50)):
    hits = {k: 0 for k in ks}
    n_total, n_skipped = 0, 0
    by_cat = defaultdict(list)
    for qi, q in enumerate(queries):
        by_cat[q["category"]].append(qi)

    for cat, qidxs in by_cat.items():
        if cat not in pools:
            n_skipped += len(qidxs)
            continue
        pool_ids = pools[cat]
        pool_pos = {pid: i for i, pid in enumerate(pool_ids)}
        pool_idx = [siglip_idx[i] for i in pool_ids if i in siglip_idx]
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
    with open(VAL_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]
    print(f"Loaded VALIDATION benchmark: {len(queries)} queries, {len(pools)} category pools")

    siglip_idx, siglip_emb = load_siglip()

    model = OutfitTransformerSigLIP().to(DEVICE).eval()  # phase 14b's exact architecture, 768-d image-only
    state_dict = torch.load(CHECKPOINT_PT, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Loaded phase 14b's checkpoint: {CHECKPOINT_PT}")

    # Candidate features (embed_item_alone), computed fresh here for the whole
    # catalog -- context-independent, deterministic given the checkpoint.
    print("Computing candidate features for the full catalog...")
    siglip_t = torch.tensor(siglip_emb, device=DEVICE)
    outs = []
    with torch.no_grad():
        for start in range(0, len(siglip_emb), 4096):
            chunk = siglip_t[start:start + 4096]
            tokens = model.encode_item_tokens(chunk)
            outs.append(model.embed_item_alone(tokens).cpu().numpy())
    candidate_emb = np.concatenate(outs, axis=0)

    recall, n_total, n_skipped = evaluate_recall(pools, queries, siglip_idx, candidate_emb, model, siglip_emb, DEVICE)
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"A0 (phase 14b's checkpoint, evaluated on VAL benchmark): "
          f"Recall@10={recall[10]:.4f} Recall@30={recall[30]:.4f} Recall@50={recall[50]:.4f}")

    result = {
        "description": "Phase 14b's existing checkpoint (random-negatives run), evaluated on the "
                        "validation benchmark for the first time -- no retraining, no test-benchmark touch.",
        "checkpoint": str(CHECKPOINT_PT.relative_to(REPO_ROOT)),
        "benchmark": str(VAL_BENCHMARK.relative_to(REPO_ROOT)),
        "recall": recall, "n_total": n_total, "n_skipped": n_skipped,
        "phase14b_reported_test_recall": {"10": 0.0588, "30": 0.1286, "50": 0.1809},
    }
    with open(OUT_MD, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()
