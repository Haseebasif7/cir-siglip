"""
Phase 31, step 6: the single final test-benchmark evaluation.

BUDGET-CONSTRAINED DEVIATION FROM THE PLAN, disclosed here and in
phase31_notes.md: steps 4 (scale sweep) and 5 (10-seed ensembling) were not
run to completion -- this project's Modal account hit a hard budget wall
mid-phase (~$22.28 spent on phase 31 against ~$7 remaining when the
constraint surfaced), and both the 30-config scale sweep and a 10-seed
ensemble would have cost far more than what remained. The scale sweep was
stopped after 0 of 30 configs completed (nothing checkpointed, nothing
lost). Rather than either quietly absorbing a truncated/misleading result or
spending money the account doesn't have, this evaluates the single best
configuration actually reached through validation-benchmark comparisons --
step 3's fully-tuned, full-budget-trained single model (val Recall@10 =
0.1924, checkpoint models/ot31_budget_check_full.pt) -- exactly once on the
real test benchmark. No ensemble. This is a real, honestly-reported result
at reduced scope, not a scaled-up claim.

Runs entirely locally (MPS/CPU) -- the checkpoint is already trained and
downloaded; this script spends zero further Modal GPU compute.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import OutfitTransformerSigLIP

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE12_DIR = REPO_ROOT / "week4/phase12_controllable_modes"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"

IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
TEST_BENCHMARK = PHASE12_DIR / "data" / "cir_benchmark.json"  # the ONE test-benchmark touch of this phase
CHECKPOINT_PT = BASE_DIR / "models" / "ot31_budget_check_full.pt"

OUT_MD = BASE_DIR / "final_evaluation.md"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
QUERY_BATCH = 1024

# Winning config's architecture (step 3's winner -- default architecture,
# only lr/batch_size/uniformity_weight/margin/input_mode changed from
# phase 14b's original; see tuning_log.md).
MODEL_KWARGS = dict(siglip_dim=1536, d_model=128, d_embed=64, n_heads=8, n_layers=4, d_ffn=512, dropout=0.1)

PHASE14B_ORIGINAL = {10: 0.0588, 30: 0.1286, 50: 0.1809}
PUBLISHED = {10: 0.0958, 30: 0.1796, 50: 0.2198}
PHASE28_TEXT_ENSEMBLE = {10: 0.1904, 30: 0.3267, 50: 0.4079}
A0_VAL = {10: 0.0659, 30: 0.1403, 50: 0.1979}


def build_base_repr():
    img = np.load(IMAGE_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img["item_ids"]]
    idx = {a: i for i, a in enumerate(item_ids)}
    image_emb = img["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)

    txt = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=True)
    assert [str(a) for a in txt["item_ids"]] == item_ids
    text_emb = txt["embeddings"].astype(np.float32)

    image_t = torch.tensor(image_emb, device=DEVICE)
    text_t = torch.tensor(text_emb, device=DEVICE)
    base_repr = F.normalize(torch.cat([image_t, text_t], dim=1), p=2, dim=-1)
    return idx, base_repr


def compute_query_embeddings(model, base_repr, idx, query_item_lists):
    outs = []
    for start in range(0, len(query_item_lists), QUERY_BATCH):
        chunk = query_item_lists[start:start + QUERY_BATCH]
        Lmax = max(len(q) for q in chunk)
        B = len(chunk)
        ctx_gidx = np.zeros((B, Lmax), dtype=np.int64)
        mask = np.ones((B, Lmax), dtype=bool)
        for i, items in enumerate(chunk):
            for j, it in enumerate(items):
                ctx_gidx[i, j] = idx[it]
                mask[i, j] = False
        ctx_t = base_repr[torch.tensor(ctx_gidx, device=DEVICE)]
        mask_t = torch.tensor(mask, device=DEVICE)
        with torch.no_grad():
            B_, L_, D_ = ctx_t.shape
            tokens = model.encode_item_tokens(ctx_t.reshape(B_ * L_, D_)).reshape(B_, L_, -1)
            q_emb = model.embed_query(tokens, mask_t)
        outs.append(q_emb.cpu().numpy())
    return np.concatenate(outs, axis=0)


def evaluate_recall(pools, queries, idx, candidate_emb, model, base_repr, ks=(10, 30, 50)):
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
        pool_idx = [idx[i] for i in pool_ids if i in idx]
        pool_emb = candidate_emb[pool_idx]

        query_item_lists, target_positions = [], []
        for qi in qidxs:
            q = queries[qi]
            items = [i for i in q["query_items"] if i in idx]
            if not items or q["target_item"] not in pool_pos:
                n_skipped += 1
                continue
            query_item_lists.append(items)
            target_positions.append(pool_pos[q["target_item"]])
        if not query_item_lists:
            continue

        query_mat = compute_query_embeddings(model, base_repr, idx, query_item_lists)
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
    with open(TEST_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]
    print(f"Loaded TEST benchmark: {len(queries)} queries, {len(pools)} category pools "
          f"-- the single touch of this phase.")

    idx, base_repr = build_base_repr()

    model = OutfitTransformerSigLIP(**MODEL_KWARGS).to(DEVICE).eval()
    model.load_state_dict(torch.load(CHECKPOINT_PT, map_location=DEVICE))
    print(f"Loaded {CHECKPOINT_PT}")

    print("Computing candidate features for the full catalog...")
    outs = []
    with torch.no_grad():
        for start in range(0, base_repr.shape[0], 4096):
            chunk = base_repr[start:start + 4096]
            tokens = model.encode_item_tokens(chunk)
            outs.append(model.embed_item_alone(tokens).cpu().numpy())
    candidate_emb = np.concatenate(outs, axis=0)

    recall, n_total, n_skipped = evaluate_recall(pools, queries, idx, candidate_emb, model, base_repr)
    print(f"n_total={n_total} n_skipped={n_skipped}")
    print(f"Phase 31 single model, test benchmark: "
          f"R@10={recall[10]:.4f} R@30={recall[30]:.4f} R@50={recall[50]:.4f}")

    beats_14b = all(recall[k] > PHASE14B_ORIGINAL[k] for k in (10, 30, 50))
    pct_of_published = {k: recall[k] / PUBLISHED[k] for k in (10, 30, 50)}
    pct_of_phase28 = {k: recall[k] / PHASE28_TEXT_ENSEMBLE[k] for k in (10, 30, 50)}
    gain_over_14b = {k: recall[k] / PHASE14B_ORIGINAL[k] - 1 for k in (10, 30, 50)}

    lines = [
        "# Phase 31, Step 6: Final Evaluation -- Test Benchmark, Once\n",
        "**Scope note (budget-constrained)**: steps 4 (scale sweep) and 5 (10-seed ensembling) were not "
        "completed -- this project's Modal account hit a hard budget wall mid-phase (~$22.28 spent on "
        "phase 31 against ~$7 remaining). This is a SINGLE-MODEL result (step 3's fully-tuned winner), "
        "not an ensemble. See `phase31_notes.md` for the full disclosure.\n",
        "Best configuration identified entirely through validation-benchmark comparisons (steps 1-3): "
        "`input_mode=image_text, lr=1.5e-4, batch_size=384, uniformity_weight=0.1, margin=0.2`, "
        "val Recall@10=0.1924. Evaluated here, exactly once, on the actual test CIR benchmark.\n",
        "## Full progression\n",
        "| Configuration | Recall@10 | Recall@30 | Recall@50 | Notes |",
        "|---|---|---|---|---|",
        f"| Phase 14b original (single config) | {PHASE14B_ORIGINAL[10]:.4f} | {PHASE14B_ORIGINAL[30]:.4f} | {PHASE14B_ORIGINAL[50]:.4f} | val_loss selection, no text |",
        f"| OutfitTransformer published | {PUBLISHED[10]:.4f} | {PUBLISHED[30]:.4f} | {PUBLISHED[50]:.4f} | different backbone, benchmark not verified |",
        f"| **Phase 31 strengthened, single model** | **{recall[10]:.4f}** | **{recall[30]:.4f}** | **{recall[50]:.4f}** | text + selection fix + tuning, no ensemble |",
        f"| Phase 28 text ensemble (project's own best) | {PHASE28_TEXT_ENSEMBLE[10]:.4f} | {PHASE28_TEXT_ENSEMBLE[30]:.4f} | {PHASE28_TEXT_ENSEMBLE[50]:.4f} | 10-model ensemble |",
        "",
        "## Derived comparisons\n",
        f"- vs. phase 14b original: {gain_over_14b[10]:+.1%} / {gain_over_14b[30]:+.1%} / {gain_over_14b[50]:+.1%} relative "
        f"({'beats' if beats_14b else 'does NOT beat'} it at every K).",
        f"- vs. OutfitTransformer published: {pct_of_published[10]:.1%} / {pct_of_published[30]:.1%} / {pct_of_published[50]:.1%} of published numbers.",
        f"- vs. phase 28's text ensemble (project's own best, 10-model): {pct_of_phase28[10]:.1%} / {pct_of_phase28[30]:.1%} / {pct_of_phase28[50]:.1%}.",
        "",
        f"n_total={n_total} n_skipped={n_skipped}.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")

    with open(BASE_DIR / "data" / "final_test_result.json", "w") as f:
        json.dump({"recall": recall, "n_total": n_total, "n_skipped": n_skipped,
                    "checkpoint": str(CHECKPOINT_PT.relative_to(REPO_ROOT))}, f, indent=2)


if __name__ == "__main__":
    main()
