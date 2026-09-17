"""
Phase 9, step 5.1: evaluate on Polyvore's own official test split, using
its standard compatibility-prediction and fill-in-the-blank protocols --
confirmed formats from step 1's inspection (dataset_structure_check.md):

- compatibility_test.txt: "<label> <set_id>_<index> ..." per line. label=1
  = a real outfit, label=0 = an artificial negative "outfit" assembled from
  items across different set_ids (1:1 balanced, 10,000 of each in test).
  Scored here as the mean pairwise cosine similarity across all C(n,2) item
  pairs in the line (a standard, simple compatibility-score convention) --
  AUC computed via the rank-sum (Mann-Whitney U) formula using
  scipy.stats.rankdata (no sklearn, consistent with this project's
  convention of avoiding it).
- fill_in_blank_test.json: {"question": [...], "blank_position": int,
  "answers": [4 candidates]}. Confirmed on a 2,000-record sample:
  answers[0] is always the correct (same-outfit) candidate. Scored here by
  picking the answer with the highest mean similarity to the known question
  items; accuracy = fraction where that pick is answers[0].

set_id_index references (e.g. "210750761_1") are resolved to real item_ids
via test.json's outfit definitions (index -> item_id per set_id) -- the
compatibility/FITB files reference outfit position, not item_id, directly.

Reports raw SigLIP as a reference point alongside Model A/B, and notes
Vasileva's own published numbers (AUC 0.88, FITB 57.6%) as a rough
literature anchor -- not an expected match, since a frozen-SigLIP+small-MLP
setup is architecturally simpler than their fully-trained type-aware network.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "polyvore_raw"
EMBEDDINGS_NPZ = BASE_DIR / "embeddings" / "siglip_base.npz"
MODELS_DIR = BASE_DIR / "models"
RESULTS_MD = BASE_DIR / "results_table.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

LITERATURE_ANCHOR = {"auc": 0.88, "fitb_accuracy": 0.576}  # Vasileva et al. ECCV'18, full type-aware model


def load_embeddings():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return item_ids, embeddings


def load_projection(path):
    model = ProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def project(model, embeddings):
    x = torch.tensor(embeddings.astype(np.float32), device=DEVICE)
    return model(x).cpu().numpy()


def build_setid_index_resolver(split):
    with open(RAW_DIR / "nondisjoint" / f"{split}.json") as f:
        outfits = json.load(f)
    resolver = {}
    for outfit in outfits:
        set_id = outfit["set_id"]
        for it in outfit["items"]:
            resolver[f"{set_id}_{it['index']}"] = it["item_id"]
    return resolver


def auc_rank_sum(scores, labels):
    """Mann-Whitney U / rank-sum AUC, no sklearn."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    ranks = rankdata(scores)  # average rank for ties
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def evaluate_compatibility(embeddings, idx, resolver):
    """embeddings: (N, D) L2-normalized array -- pairwise dot products are
    computed only for the specific pairs referenced in each test line, NOT
    as a full N x N matrix (at N=251,008 that matrix would be ~252GB and
    OOM the machine -- confirmed the hard way on the first attempt)."""
    labels, scores = [], []
    n_skipped = 0
    with open(RAW_DIR / "nondisjoint" / "compatibility_test.txt") as f:
        for line in f:
            parts = line.split()
            label = int(parts[0])
            refs = parts[1:]
            item_ids = [resolver.get(r) for r in refs]
            if any(i is None or i not in idx for i in item_ids):
                n_skipped += 1
                continue
            vecs = embeddings[[idx[a] for a in item_ids]]
            pair_sims = [float(np.dot(vecs[i], vecs[j])) for i, j in combinations(range(len(vecs)), 2)]
            score = float(np.mean(pair_sims)) if pair_sims else 0.0
            labels.append(label)
            scores.append(score)
    auc = auc_rank_sum(scores, labels)
    return auc, len(labels), n_skipped


def evaluate_fitb(embeddings, idx, resolver):
    with open(RAW_DIR / "nondisjoint" / "fill_in_blank_test.json") as f:
        questions = json.load(f)

    n_correct, n_total, n_skipped = 0, 0, 0
    for q in questions:
        q_items = [resolver.get(r) for r in q["question"]]
        ans_items = [resolver.get(r) for r in q["answers"]]
        if any(i is None or i not in idx for i in q_items) or any(i is None or i not in idx for i in ans_items):
            n_skipped += 1
            continue
        q_vecs = embeddings[[idx[i] for i in q_items]]
        best_ans, best_score = None, -np.inf
        for ai, ans_item in enumerate(ans_items):
            a_vec = embeddings[idx[ans_item]]
            score = float(np.mean(q_vecs @ a_vec))
            if score > best_score:
                best_score = score
                best_ans = ai
        if best_ans == 0:  # answers[0] confirmed the correct candidate in step 1
            n_correct += 1
        n_total += 1
    accuracy = n_correct / n_total if n_total else 0.0
    return accuracy, n_total, n_skipped


def main():
    item_ids, raw_embeddings = load_embeddings()
    idx = {a: i for i, a in enumerate(item_ids)}
    resolver = build_setid_index_resolver("test")
    print(f"Loaded {len(item_ids)} embeddings, resolved {len(resolver)} set_id_index references.")

    configs = [
        ("Raw SigLIP (reference)", raw_embeddings),
        ("Model A (random negs)", None),
        ("Model B (hard negs)", None),
    ]
    model_paths = {
        "Model A (random negs)": MODELS_DIR / "model_a_random_negs.pt",
        "Model B (hard negs)": MODELS_DIR / "model_b_hard_negs.pt",
    }

    rows = []
    for label, precomputed in configs:
        if precomputed is not None:
            embs = precomputed
        else:
            model = load_projection(model_paths[label])
            embs = project(model, raw_embeddings)
            norm = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / norm

        auc, n_compat, n_compat_skip = evaluate_compatibility(embs, idx, resolver)
        fitb_acc, n_fitb, n_fitb_skip = evaluate_fitb(embs, idx, resolver)
        rows.append((label, auc, n_compat, n_compat_skip, fitb_acc, n_fitb, n_fitb_skip))
        print(f"{label}: AUC={auc:.4f} (n={n_compat}, {n_compat_skip} skipped) | "
              f"FITB acc={fitb_acc:.4f} (n={n_fitb}, {n_fitb_skip} skipped)")

    lines = [
        "# Phase 9, Step 5.1: Official Polyvore Test-Split Evaluation",
        "",
        f"Literature anchor (Vasileva et al. ECCV'18, full type-aware trained network, "
        f"NOT directly comparable to this phase's frozen-SigLIP+small-MLP setup): "
        f"AUC={LITERATURE_ANCHOR['auc']}, FITB accuracy={LITERATURE_ANCHOR['fitb_accuracy']:.3f}.",
        "",
        "| Configuration | Compatibility AUC | FITB Accuracy |",
        "|---|---|---|",
    ]
    for label, auc, n_compat, _, fitb_acc, n_fitb, _ in rows:
        lines.append(f"| {label} | {auc:.4f} | {fitb_acc:.4f} |")
    lines.append("")
    lines.append(f"(Compatibility test: n={rows[0][2]} lines scored; FITB test: n={rows[0][5]} questions scored.)")
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {RESULTS_MD}")


if __name__ == "__main__":
    main()
