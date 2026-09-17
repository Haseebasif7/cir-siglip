"""
Phase 30, step 6 (interpretability): loads the single-seed gate checkpoint
and inspects, on a sample of validation queries, what the learned
aggregation weights and per-aspect similarity contributions actually look
like. Runs locally on CPU/MPS -- no Modal needed, mirrors phase 29's
03_attention_qualitative.py in spirit (direct inspection of a learned
weighting mechanism, not just its aggregate retrieval number).

Uses validation queries only, matching phase 29's precedent of not touching
the test benchmark until (if) a phase reaches the final single-check step.
Runs regardless of the single-seed gate's GO/NO-GO outcome -- required output
per the brief's list, not conditional on ensemble scale (unlike
individual_seeds.md / ensemble_size_sweep.md / final_evaluation.md).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import MultiAspectProjectionHead, ASPECT_NAMES, weight_entropy

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = REPO_ROOT / "week3/phase9_polyvore_compatibility"
PHASE23_DIR = REPO_ROOT / "week4/phase23_hyperparameter_tuning"
PHASE27_DIR = REPO_ROOT / "week7/phase27_text_and_category"

IMAGE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TEXT_EMBEDDINGS_NPZ = PHASE27_DIR / "data" / "text_embeddings.npz"
VAL_BENCHMARK = PHASE23_DIR / "data" / "cir_val_benchmark.json"
CKPT_PATH = BASE_DIR / "models" / "multiaspect_seed42.pt"
OUT_MD = BASE_DIR / "aggregation_qualitative.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_SAMPLE_QUERIES = 8
MIN_CONTEXT_LEN = 2
SEED = 7


def main():
    print("Loading base image+text representation...")
    img_data = np.load(IMAGE_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img_data["item_ids"]]
    id_to_idx = {a: i for i, a in enumerate(item_ids)}
    image_emb = img_data["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)

    text_data = np.load(TEXT_EMBEDDINGS_NPZ, allow_pickle=True)
    assert [str(a) for a in text_data["item_ids"]] == item_ids
    text_emb = text_data["embeddings"].astype(np.float32)

    base_repr = F.normalize(
        torch.tensor(np.concatenate([image_emb, text_emb], axis=1), device=DEVICE), p=2, dim=-1
    )

    model = MultiAspectProjectionHead().to(DEVICE).eval()
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

    with open(VAL_BENCHMARK) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]

    rng = np.random.default_rng(SEED)
    eligible = [qi for qi, q in enumerate(queries)
                if len(q["query_items"]) >= MIN_CONTEXT_LEN
                and all(i in id_to_idx for i in q["query_items"])
                and q["category"] in pools and q["target_item"] in pools[q["category"]]]
    sample_qidx = rng.choice(eligible, size=min(N_SAMPLE_QUERIES, len(eligible)), replace=False)

    with torch.no_grad():
        proj_aspects_all = model(base_repr)  # (n_items, K, D)

    lines = ["# Phase 30, Step 6: Aggregation Weight and Per-Aspect Contribution Inspection\n",
             f"Sample of {len(sample_qidx)} validation queries (context length >= {MIN_CONTEXT_LEN}, "
             f"seed={SEED}), single-seed gate checkpoint (`models/multiaspect_seed42.pt`). "
             "For each query: the learned aggregation weights (computed from the query's own "
             "mean-pooled-per-aspect context, never from the candidate) and the per-aspect "
             "similarity contribution to the true target vs. 3 random same-category negatives.\n"]

    all_weight_rows = []
    for qi in sample_qidx:
        q = queries[qi]
        cat = q["category"]
        ctx_idx = [id_to_idx[i] for i in q["query_items"]]
        target_gidx = id_to_idx[q["target_item"]]

        with torch.no_grad():
            q_aspects = F.normalize(proj_aspects_all[ctx_idx].mean(dim=0), p=2, dim=-1)  # (K,D)
            q_weights = model.agg_weights(q_aspects.unsqueeze(0)).squeeze(0)  # (K,)

        raw_ent, norm_ent = weight_entropy(q_weights)
        all_weight_rows.append(q_weights.cpu().numpy())

        pool_ids = pools[cat]
        pool_gidx = [id_to_idx[p] for p in pool_ids if p in id_to_idx]
        candidates = [q["target_item"]]
        other = [p for p in pool_ids if p != q["target_item"] and p in id_to_idx]
        neg_sample = rng.choice(other, size=min(3, len(other)), replace=False).tolist()
        candidates += neg_sample
        labels = ["TRUE TARGET"] + [f"random negative {i+1}" for i in range(len(neg_sample))]

        lines.append(f"## Query {q['target_item']} (category={cat}, context length={len(ctx_idx)})\n")
        lines.append(f"Aggregation weights: " +
                     ", ".join(f"{n}={w:.3f}" for n, w in zip(ASPECT_NAMES, q_weights.tolist())) +
                     f"  (entropy raw={raw_ent.item():.3f}, normalized={norm_ent.item():.3f})\n")
        lines.append("| Candidate | " + " | ".join(f"sim_{n}" for n in ASPECT_NAMES) + " | weighted score |")
        lines.append("|---|" + "---|" * (len(ASPECT_NAMES) + 1))
        for cand_id, label in zip(candidates, labels):
            c_idx = id_to_idx[cand_id]
            with torch.no_grad():
                c_aspects = proj_aspects_all[c_idx]  # (K,D)
                sim_k = (q_aspects * c_aspects).sum(dim=-1)  # (K,)
                score = (q_weights * sim_k).sum().item()
            lines.append(f"| {label} | " + " | ".join(f"{s:.3f}" for s in sim_k.tolist()) +
                         f" | {score:.3f} |")
        lines.append("")

    weight_matrix = np.stack(all_weight_rows)  # (n_queries, K)
    mean_weights = weight_matrix.mean(axis=0)
    std_weights = weight_matrix.std(axis=0)
    raw_ents, norm_ents = weight_entropy(torch.tensor(weight_matrix))

    lines.append("## Summary across the sample\n")
    lines.append("| Aspect | Mean weight | Std across queries |")
    lines.append("|---|---|---|")
    for n, m, s in zip(ASPECT_NAMES, mean_weights, std_weights):
        lines.append(f"| {n} | {m:.3f} | {s:.3f} |")
    lines.append("")
    lines.append(f"Mean normalized entropy across the sample: {norm_ents.mean().item():.3f} "
                 f"(1.0 = fully uniform/aspect-bias-disengaged, 0.0 = one-hot collapse onto a single aspect).")
    lines.append("")

    # Head-similarity check: are the aspect heads producing distinct outputs,
    # or did some pair collapse onto the same function -- per the brief's
    # explicit "if things don't go smoothly" instruction.
    with torch.no_grad():
        sample_gidx = rng.choice(len(item_ids), size=2000, replace=False)
        sample_aspects = proj_aspects_all[sample_gidx]  # (2000, K, D)
        K = sample_aspects.shape[1]
        cos_matrix = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cos_matrix[i, j] = F.cosine_similarity(
                    sample_aspects[:, i, :], sample_aspects[:, j, :], dim=-1
                ).mean().item()
    lines.append("## Aspect head output similarity check (2000-item sample)\n")
    lines.append("Mean cosine similarity between each pair of aspect heads' outputs on the same items "
                 "-- near 1.0 would mean two heads collapsed onto the same function.\n")
    lines.append("| | " + " | ".join(ASPECT_NAMES) + " |")
    lines.append("|---|" + "---|" * K)
    for i, n in enumerate(ASPECT_NAMES):
        lines.append(f"| {n} | " + " | ".join(f"{cos_matrix[i,j]:.3f}" for j in range(K)) + " |")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()
