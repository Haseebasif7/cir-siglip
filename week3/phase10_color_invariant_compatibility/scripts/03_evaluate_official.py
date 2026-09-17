"""
Phase 10, step 4.1: evaluate on Polyvore's own official test split, same
protocol as phase 9 (week3/phase9_polyvore_compatibility/scripts/
06_evaluate_official.py, copied with only the model swapped) -- compatibility
AUC (rank-sum/Mann-Whitney formula) and FITB accuracy. Uses the ORIGINAL
(non-perturbed) SigLIP embeddings, since the official test split is real
Polyvore product photography, not color-shifted.

Reports raw SigLIP and phase 9 Model A's numbers alongside this phase's
color-invariant model for direct comparison (phase 9's numbers carried
forward verbatim from its own results_table.md, not recomputed, consistent
with how phase 9 itself carried forward phase 8's numbers).
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
PHASE9_DIR = BASE_DIR.parent / "phase9_polyvore_compatibility"
RAW_DIR = PHASE9_DIR / "data" / "polyvore_raw"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
MODELS_DIR = BASE_DIR / "models"
RESULTS_MD = BASE_DIR / "results_table.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# carried forward verbatim from phase 9's results_table.md, not recomputed
PHASE9_OFFICIAL = {
    "Raw SigLIP (reference)": {"auc": 0.717, "fitb_accuracy": 0.484},
    "Phase 9 Model A (Polyvore-trained, random negs)": {"auc": 0.947, "fitb_accuracy": 0.703},
    "Phase 9 Model B (Polyvore-trained, hard negs)": {"auc": 0.937, "fitb_accuracy": 0.684},
}


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
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    ranks = rankdata(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def evaluate_compatibility(embeddings, idx, resolver):
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
        if best_ans == 0:
            n_correct += 1
        n_total += 1
    accuracy = n_correct / n_total if n_total else 0.0
    return accuracy, n_total, n_skipped


def main():
    item_ids, raw_embeddings = load_embeddings()
    idx = {a: i for i, a in enumerate(item_ids)}
    resolver = build_setid_index_resolver("test")
    print(f"Loaded {len(item_ids)} embeddings, resolved {len(resolver)} set_id_index references.")

    model = load_projection(MODELS_DIR / "model_color_invariant.pt")
    embs = project(model, raw_embeddings)
    norm = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / norm

    auc, n_compat, n_compat_skip = evaluate_compatibility(embs, idx, resolver)
    fitb_acc, n_fitb, n_fitb_skip = evaluate_fitb(embs, idx, resolver)
    print(f"Phase 10 color-invariant model: AUC={auc:.4f} (n={n_compat}, {n_compat_skip} skipped) | "
          f"FITB acc={fitb_acc:.4f} (n={n_fitb}, {n_fitb_skip} skipped)")

    lines = [
        "# Phase 10, Step 4.1: Official Polyvore Test-Split Evaluation",
        "",
        "Same protocol as phase 9 (compatibility AUC via rank-sum formula, FITB accuracy). "
        "Phase 9 numbers carried forward verbatim for direct comparison, not recomputed.",
        "",
        "| Configuration | Compatibility AUC | FITB Accuracy |",
        "|---|---|---|",
    ]
    for label, m in PHASE9_OFFICIAL.items():
        lines.append(f"| {label} | {m['auc']:.4f} | {m['fitb_accuracy']:.4f} |")
    lines.append(f"| **Phase 10 color-invariant model (random negs + invariance loss)** | "
                  f"**{auc:.4f}** | **{fitb_acc:.4f}** |")
    lines.append("")
    lines.append(f"(Compatibility test: n={n_compat} lines scored; FITB test: n={n_fitb} questions scored.)")
    lines.append("")

    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"\nSaved {RESULTS_MD}")


if __name__ == "__main__":
    main()
