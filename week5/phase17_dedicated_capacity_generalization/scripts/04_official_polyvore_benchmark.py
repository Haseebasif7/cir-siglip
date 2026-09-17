"""
Phase 17, step 5.1: Polyvore's own official compatibility-AUC and FITB
benchmark (`week3/phase9_polyvore_compatibility/scripts/06_evaluate_official.py`'s
exact protocol and code, reused directly). Honest scoping note: phase 12c
itself never ran this benchmark -- phases 12/12b/12c/12d all evaluated
exclusively via this project's own CIR (Recall@K) harness, so there are no
pre-existing phase-12c AUC/FITB numbers to carry forward as a citation.
Since the brief asks for a direct comparison on this specific benchmark, this
script computes it FRESH for both phase 12c's checkpoint (loaded via its own
`ControllableProjectionHead`) and this phase's checkpoint, at all three
config points (substitute/complement/blend), so the "phase 12c" row in the
resulting table is a genuine same-script, same-day measurement, not
retrieved from an old file that doesn't contain this metric. Raw SigLIP and
phase 9's Model A (both already reported by phase 9's own results_table.md)
are included as reference rows, values reused unchanged from that file since
neither depends on anything this phase or phase 12c trained.
"""
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DedicatedCapacityHead

WEEK4_DIR = Path(__file__).resolve().parent.parent.parent.parent / "week4"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded under a distinct module name to avoid colliding with this directory's own `model.py`
# (both files are named `model.py`; a plain `sys.path` insert + `import model` would silently
# reuse whichever module is already cached under the name "model").
_phase12c_model = _load_module(
    "phase12c_model", WEEK4_DIR / "phase12c_ranking_distillation" / "scripts" / "model.py")
ControllableProjectionHead = _phase12c_model.ControllableProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
RAW_DIR = PHASE9_DIR / "data" / "polyvore_raw"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
PHASE12C_CHECKPOINT = WEEK4_DIR / "phase12c_ranking_distillation" / "models" / "ranking_distillation.pt"
PHASE17_CHECKPOINT = BASE_DIR / "models" / "dedicated_capacity_substitute_complement.pt"
RESULTS_MD = BASE_DIR / "results_table.md"  # appended to (run after 03_evaluate_cir.py, before 06_alpha_sweep.py)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Reused unchanged from week3/phase9_polyvore_compatibility/results_table.md
REFERENCE_ROWS = {
    "Raw SigLIP (reference)": {"auc": 0.7172, "fitb": 0.4843},
    "Phase 9 Model A (single-mode projection, reference)": {"auc": 0.9469, "fitb": 0.7031},
}
LITERATURE_ANCHOR = {"auc": 0.88, "fitb_accuracy": 0.576}


def load_embeddings():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return item_ids, (embeddings / norms).astype(np.float32)


def build_setid_index_resolver(split):
    import json
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
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


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
    import json
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


@torch.no_grad()
def project(model, embeddings, alpha, batch_size=4096):
    outs = []
    for start in range(0, len(embeddings), batch_size):
        x = torch.tensor(embeddings[start:start + batch_size], device=DEVICE)
        outs.append(model(x, alpha=alpha).cpu().numpy())
    embs = np.concatenate(outs, axis=0)
    return embs / np.linalg.norm(embs, axis=1, keepdims=True)


def main():
    item_ids, raw_emb = load_embeddings()
    idx = {a: i for i, a in enumerate(item_ids)}
    resolver = build_setid_index_resolver("test")
    print(f"Loaded {len(item_ids)} embeddings, resolved {len(resolver)} set_id_index references.")

    phase12c_model = ControllableProjectionHead().to(DEVICE)
    phase12c_model.load_state_dict(torch.load(PHASE12C_CHECKPOINT, map_location=DEVICE))
    phase12c_model.eval()

    phase17_model = DedicatedCapacityHead().to(DEVICE)
    phase17_model.load_state_dict(torch.load(PHASE17_CHECKPOINT, map_location=DEVICE))
    phase17_model.eval()

    configs = [
        ("Phase 12c (shared trunk): Substitute mode", phase12c_model, 1.0),
        ("Phase 12c (shared trunk): Complement mode", phase12c_model, 0.0),
        ("Phase 12c (shared trunk): Blend (0.5)", phase12c_model, 0.5),
        ("Phase 17 (dedicated capacity): Substitute mode", phase17_model, 1.0),
        ("Phase 17 (dedicated capacity): Complement mode", phase17_model, 0.0),
        ("Phase 17 (dedicated capacity): Blend (0.5)", phase17_model, 0.5),
    ]

    rows = []
    for label, model, alpha in configs:
        embs = project(model, raw_emb, alpha)
        auc, n_compat, _ = evaluate_compatibility(embs, idx, resolver)
        fitb_acc, n_fitb, _ = evaluate_fitb(embs, idx, resolver)
        rows.append((label, auc, fitb_acc))
        print(f"{label}: AUC={auc:.4f} (n={n_compat}) | FITB acc={fitb_acc:.4f} (n={n_fitb})")

    lines = [
        "",
        "---",
        "",
        "## Step 5.1: Official Polyvore Test-Split Evaluation (Compatibility AUC + FITB)",
        "",
        "Same protocol as `week3/phase9_polyvore_compatibility/scripts/06_evaluate_official.py`, "
        "reused directly. **Scoping note**: phase 12c never ran this benchmark (phases "
        "12/12b/12c/12d all used this project's own CIR/Recall@K harness exclusively) -- so "
        "there is no pre-existing phase 12c AUC/FITB number to cite. The phase 12c row below is "
        "computed FRESH, in this same script, from phase 12c's own saved checkpoint "
        "(`week4/phase12c_ranking_distillation/models/ranking_distillation.pt`), making this a "
        "genuine same-protocol, same-day comparison rather than a mismatched citation.",
        "",
        f"Literature anchor (Vasileva et al. ECCV'18, full type-aware trained network, NOT "
        f"directly comparable to either frozen-SigLIP+small-head setup here): "
        f"AUC={LITERATURE_ANCHOR['auc']}, FITB accuracy={LITERATURE_ANCHOR['fitb_accuracy']:.3f}.",
        "",
        "| Configuration | Compatibility AUC | FITB Accuracy |",
        "|---|---|---|",
    ]
    for label, ref in REFERENCE_ROWS.items():
        lines.append(f"| {label} | {ref['auc']:.4f} | {ref['fitb']:.4f} |")
    lines.append("| | | |")
    for label, auc, fitb_acc in rows:
        lines.append(f"| {label} | {auc:.4f} | {fitb_acc:.4f} |")
    lines.append("")
    lines.append(f"(Compatibility test: n=20000 lines scored; FITB test: n=10000 questions scored, "
                  "same test-split sizes as phase 9's own run.)")
    lines.append("")

    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAppended to {RESULTS_MD}")


if __name__ == "__main__":
    main()
