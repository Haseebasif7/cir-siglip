"""
Phase 9, step 6: qualitative check, two domains.

(a) Polyvore-domain: 6-8 query items from the official test split, top-5
    results under raw SigLIP vs. this phase's Model A/B, on Polyvore's own
    images.
(b) Amazon-domain (transfer): the same fixed 8-asin query list phase 8 used
    (week3/phase8_cross_category_retraining/scripts/05_qualitative_examples.py),
    including the pink drawstring laundry bag B01FWDLMYC -- phase 8's one
    partial-only case -- run through the transfer setup: raw SigLIP / phase
    8's Amazon-trained Model A / this phase's Polyvore-trained Model A/B.
"""
import ast
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "polyvore_raw"
POLYVORE_IMAGES_DIR = BASE_DIR / "data" / "images"
POLYVORE_EMBEDDINGS_NPZ = BASE_DIR / "embeddings" / "siglip_base.npz"

PHASE8_DIR = BASE_DIR.parent / "phase8_cross_category_retraining"
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EVAL_EMBEDDINGS_NPZ = EVAL_DIR / "embeddings" / "siglip_base.npz"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"

OUT_DIR = BASE_DIR / "qualitative_examples"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K = 5

AMAZON_QUERY_ASINS = [
    "B005NGRC0W", "B00ISL4KMW", "B01AIT77RQ", "B01FWDLMYC",
    "B01B5BIU0E", "B00505DPQG", "B000GB0G1G", "B005PQO6ZO",
]

# 8 Polyvore test items chosen for category diversity (one per common semantic_category)
N_POLYVORE_QUERIES = 8


def load_projection(path):
    model = ProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def project(model, embeddings):
    x = torch.tensor(embeddings.astype(np.float32), device=DEVICE)
    return model(x).cpu().numpy()


def top_k_excluding_self(embeddings, query_idx, k):
    """Computes a single similarity ROW on demand (embeddings @ query_vector)
    rather than ever materializing a full N x N matrix -- at Polyvore's
    ~251k-item scale, a full matrix would be ~252GB and OOM the machine
    (confirmed the hard way in step 5.1's first attempt). This is cheap:
    one (N, D) @ (D,) matvec per query, not (N, D) @ (D, N)."""
    row = embeddings @ embeddings[query_idx]
    row = row.copy()
    row[query_idx] = -np.inf
    top = np.argsort(-row)[:k]
    return top, row


def build_grid(query_label, query_img_path, query_title, rows_config, get_result_info, out_path):
    fig, axes = plt.subplots(len(rows_config), TOP_K + 1, figsize=(3 * (TOP_K + 1), 3.2 * len(rows_config)))
    if len(rows_config) == 1:
        axes = axes.reshape(1, -1)

    for row_i, (row_label, embeddings, query_idx) in enumerate(rows_config):
        ax = axes[row_i, 0]
        try:
            ax.imshow(Image.open(query_img_path).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "[missing]", ha="center", va="center")
        ax.axis("off")
        ax.set_title(f"QUERY\n{query_title[:30]}", fontsize=7)
        ax.text(-0.3, 0.5, row_label, transform=ax.transAxes, fontsize=9, ha="right", va="center", weight="bold")

        top, sim_row = top_k_excluding_self(embeddings, query_idx, TOP_K)
        for col_i, j in enumerate(top):
            ax2 = axes[row_i, col_i + 1]
            img_path_j, title_j = get_result_info(j)
            try:
                ax2.imshow(Image.open(img_path_j).convert("RGB"))
            except Exception:
                ax2.text(0.5, 0.5, "[missing]", ha="center", va="center")
            ax2.axis("off")
            ax2.set_title(f"#{col_i+1} sim={sim_row[j]:.3f}\n{title_j[:28]}", fontsize=6)

    fig.suptitle(query_label, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"Saved {out_path}")


def polyvore_domain_grids():
    data = np.load(POLYVORE_EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    idx_of = {a: i for i, a in enumerate(item_ids)}

    with open(RAW_DIR / "polyvore_item_metadata.json") as f:
        meta = json.load(f)

    model_a = load_projection(BASE_DIR / "models" / "model_a_random_negs.pt")
    model_b = load_projection(BASE_DIR / "models" / "model_b_hard_negs.pt")
    za = project(model_a, embeddings)
    zb = project(model_b, embeddings)
    za = za / np.linalg.norm(za, axis=1, keepdims=True)
    zb = zb / np.linalg.norm(zb, axis=1, keepdims=True)

    # pick one item per common semantic_category, present in our item set
    from collections import defaultdict
    by_cat = defaultdict(list)
    for item_id in item_ids:
        cat = meta.get(item_id, {}).get("semantic_category", "")
        by_cat[cat].append(item_id)
    chosen_cats = sorted(by_cat, key=lambda c: -len(by_cat[c]))[:N_POLYVORE_QUERIES]
    rng = np.random.default_rng(42)
    query_items = [by_cat[c][rng.integers(0, len(by_cat[c]))] for c in chosen_cats]

    def get_info(j):
        item_id = item_ids[j]
        return POLYVORE_IMAGES_DIR / f"{item_id}.jpg", meta.get(item_id, {}).get("url_name", item_id)

    for item_id in query_items:
        qi = idx_of[item_id]
        rows_config = [
            ("Raw SigLIP", embeddings, qi),
            ("Model A (random negs)", za, qi),
            ("Model B (hard negs)", zb, qi),
        ]
        title = meta.get(item_id, {}).get("url_name", item_id)
        out_path = OUT_DIR / f"polyvore_query_{item_id}.png"
        build_grid(f"Polyvore query: {item_id} ({meta.get(item_id, {}).get('semantic_category', '?')})",
                   POLYVORE_IMAGES_DIR / f"{item_id}.jpg", title, rows_config, get_info, out_path)


def amazon_domain_grids():
    df = pd.read_csv(EVAL_SAMPLE_CSV)
    title_map = dict(zip(df["asin"], df["title"]))
    image_path_map = dict(zip(df["asin"], df["image_path"]))

    data = np.load(EVAL_EMBEDDINGS_NPZ, allow_pickle=True)
    asins = [str(a) for a in data["asins"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    idx_of = {a: i for i, a in enumerate(asins)}

    phase8_a = load_projection(PHASE8_DIR / "models" / "model_a_random_negs.pt")
    z_p8a = project(phase8_a, embeddings)
    z_p8a = z_p8a / np.linalg.norm(z_p8a, axis=1, keepdims=True)

    model_a = load_projection(BASE_DIR / "models" / "model_a_random_negs.pt")
    model_b = load_projection(BASE_DIR / "models" / "model_b_hard_negs.pt")
    za = project(model_a, embeddings)
    zb = project(model_b, embeddings)
    za = za / np.linalg.norm(za, axis=1, keepdims=True)
    zb = zb / np.linalg.norm(zb, axis=1, keepdims=True)

    def get_info(j):
        asin_j = asins[j]
        return EVAL_DIR / image_path_map[asin_j], str(title_map.get(asin_j, ""))

    for query_asin in AMAZON_QUERY_ASINS:
        if query_asin not in idx_of:
            print(f"WARNING: {query_asin} not in Amazon eval sample, skipping.")
            continue
        qi = idx_of[query_asin]
        rows_config = [
            ("Raw SigLIP", embeddings, qi),
            ("Phase 8 Model A\n(Amazon-trained)", z_p8a, qi),
            ("Phase 9 Model A\n(Polyvore-trained)", za, qi),
            ("Phase 9 Model B\n(Polyvore-trained)", zb, qi),
        ]
        title = str(title_map.get(query_asin, ""))
        out_path = OUT_DIR / f"amazon_query_{query_asin}.png"
        build_grid(f"Amazon query (transfer test): {query_asin}",
                   EVAL_DIR / image_path_map[query_asin], title, rows_config, get_info, out_path)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Building Polyvore-domain grids...")
    polyvore_domain_grids()
    print("\nBuilding Amazon-domain transfer grids...")
    amazon_domain_grids()


if __name__ == "__main__":
    main()
