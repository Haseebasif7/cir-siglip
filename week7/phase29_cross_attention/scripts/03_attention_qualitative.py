"""
Phase 29, step 6 (qualitative component): what do the learned attention
weights actually look like, on the trained seed=42 model, for a sample of
real validation queries. Run locally (CPU/MPS) -- a handful of forward
passes, no Modal GPU needed. Required output regardless of the step 3 gate
outcome (attention_qualitative.md is not conditional on reaching ensemble
scale in the brief's required-outputs list).

For each sample query, reports the attention distribution over context
items twice: once conditioned on the true target (the only conditioning
the model ever saw during training), and once conditioned on a handful of
random pool candidates (the situation it actually faces at evaluation
time, and never saw during training) -- directly testing this phase's own
diagnosed explanation for the single-seed gate's regression: a train/eval
conditioning mismatch.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
WEEK7_DIR = BASE_DIR.parent
PROJECT_ROOT = WEEK7_DIR.parent

SIGLIP_NPZ = PROJECT_ROOT / "week3" / "phase9_polyvore_compatibility" / "embeddings" / "siglip_base.npz"
TEXT_NPZ = WEEK7_DIR / "phase27_text_and_category" / "data" / "text_embeddings.npz"
VAL_BENCH = PROJECT_ROOT / "week4" / "phase23_hyperparameter_tuning" / "data" / "cir_val_benchmark.json"
META_JSON = PROJECT_ROOT / "week3" / "phase9_polyvore_compatibility" / "data" / "polyvore_raw" / "polyvore_item_metadata.json"

MODEL_CKPT = BASE_DIR / "models" / "xattn_seed42.pt"
XATTN_CKPT = BASE_DIR / "models" / "xattn_seed42_xattn.pt"
OUT_MD = BASE_DIR / "attention_qualitative.md"

N_QUERIES = 6
N_RANDOM_CANDIDATES = 3
SEED = 7


class ProjectionHeadGeneral(torch.nn.Module):
    def __init__(self, in_dim, hidden_dims=(1024,), out_dim=128, dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            prev = h
        layers.append(torch.nn.Linear(prev, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class CrossAttentionScorer(torch.nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.candidate_key = torch.nn.Linear(dim, dim)
        self.context_query = torch.nn.Linear(dim, dim)
        self.context_value = torch.nn.Linear(dim, dim)
        self.candidate_value = torch.nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)

    def weights_for_candidate(self, proj_context, proj_candidate):
        k_c = self.candidate_key(proj_candidate)
        q_x = self.context_query(proj_context)
        logits = (q_x @ k_c) / self.scale
        return F.softmax(logits, dim=-1)


def short_label(meta, item_id):
    m = meta.get(item_id, {})
    for field in ("description", "title", "url_name"):
        v = m.get(field)
        if v:
            return v[:50]
    return "(no text)"


def main():
    random.seed(SEED)
    device = "cpu"

    img_npz = np.load(SIGLIP_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in img_npz["item_ids"]]
    id_to_gidx = {a: i for i, a in enumerate(item_ids)}
    image_emb = img_npz["embeddings"].astype(np.float32)
    image_emb = image_emb / np.linalg.norm(image_emb, axis=1, keepdims=True)

    text_npz = np.load(TEXT_NPZ, allow_pickle=True)
    text_ids = [str(a) for a in text_npz["item_ids"]]
    assert text_ids == item_ids
    text_emb = text_npz["embeddings"].astype(np.float32)

    base_repr = F.normalize(torch.tensor(np.concatenate([image_emb, text_emb], axis=1)), p=2, dim=-1)

    with open(VAL_BENCH) as f:
        bench = json.load(f)
    pools, queries = bench["pools"], bench["queries"]

    with open(META_JSON) as f:
        meta = json.load(f)

    model = ProjectionHeadGeneral(in_dim=1536, hidden_dims=[1024], out_dim=128)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
    model.eval()
    xattn = CrossAttentionScorer(dim=128)
    xattn.load_state_dict(torch.load(XATTN_CKPT, map_location=device))
    xattn.eval()

    # Pick queries with context length >= 3 (more interesting to look at
    # than length-1/2), spread across a few categories, deterministic.
    candidates_pool = [(qi, q) for qi, q in enumerate(queries)
                        if len(q["query_items"]) >= 3 and q["category"] in pools]
    random.shuffle(candidates_pool)
    sample = candidates_pool[:N_QUERIES]

    lines = ["# Phase 29: Attention Qualitative Check\n",
             f"Sample of {len(sample)} real validation queries (context length >= 3), "
             f"using the single-seed (42) checkpoint from the step 3 gate. For each query, "
             f"the attention distribution over context items is shown twice: conditioned on "
             f"the TRUE target (what the model was actually trained on) and conditioned on a "
             f"few random pool candidates (what it actually faces at evaluation time).\n"]

    with torch.no_grad():
        proj_all = model(base_repr)

        for qi, q in sample:
            ctx_ids = [i for i in q["query_items"] if i in id_to_gidx]
            ctx_idx = [id_to_gidx[i] for i in ctx_ids]
            target_id = q["target_item"]
            cat = q["category"]
            pool_ids = pools[cat]

            proj_ctx = proj_all[ctx_idx]  # (L, 128)
            ctx_labels = [short_label(meta, i) for i in ctx_ids]

            lines.append(f"## Query {qi} (category: {cat}, context length {len(ctx_ids)})\n")
            lines.append("Context items: " + "; ".join(f"`{l}`" for l in ctx_labels))
            lines.append(f"\nTrue target: `{short_label(meta, target_id)}`\n")

            def report_for_candidate(label, cand_id):
                if cand_id not in id_to_gidx:
                    return
                cand_proj = proj_all[id_to_gidx[cand_id]]
                w = xattn.weights_for_candidate(proj_ctx, cand_proj).numpy()
                p = np.clip(w, 1e-12, None)
                ent = -(p * np.log(p)).sum()
                ent_norm = ent / math.log(len(w)) if len(w) > 1 else float("nan")
                w_str = ", ".join(f"{ctx_labels[i][:20]}={w[i]:.2f}" for i in range(len(w)))
                lines.append(f"- Conditioned on {label} (`{short_label(meta, cand_id)}`): "
                             f"[{w_str}] (entropy={ent:.2f}, normalized={ent_norm:.2f})")

            report_for_candidate("TRUE TARGET", target_id)
            random_negs = random.sample([p for p in pool_ids if p != target_id],
                                         min(N_RANDOM_CANDIDATES, len(pool_ids) - 1))
            for k, neg_id in enumerate(random_negs):
                report_for_candidate(f"random negative #{k+1}", neg_id)
            lines.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
