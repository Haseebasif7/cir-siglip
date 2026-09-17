"""
Phase 7, step 7: qualitative side-by-side comparison. Picks example queries
from the untouched eval sample and shows top-5 results under Raw SigLIP,
Model B alone, and Blended(alpha=0.5) -- the check for whether the character
of results actually shifts (plausible complements surfacing that raw
similarity alone would never retrieve), and a spot-check for whether any
Model B "hard negative" was actually a near-duplicate variant that slipped
through the 0.97 similarity cap (a failure mode flagged during design).
"""
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
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EVAL_EMBEDDINGS_NPZ = EVAL_DIR / "embeddings" / "siglip_base.npz"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"
MODELS_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "qualitative_examples"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_QUERIES = 8
TOP_K = 5


def load_data():
    df = pd.read_csv(EVAL_SAMPLE_CSV)
    title_map = dict(zip(df["asin"], df["title"]))
    image_path_map = dict(zip(df["asin"], df["image_path"]))

    data = np.load(EVAL_EMBEDDINGS_NPZ, allow_pickle=True)
    asins = [str(a) for a in data["asins"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return asins, embeddings, title_map, image_path_map


def load_projection(name):
    model = ProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def project(model, embeddings):
    x = torch.tensor(embeddings.astype(np.float32), device=DEVICE)
    return model(x).cpu().numpy()


def top_k_excluding_self(sims, query_idx, k):
    row = sims[query_idx].copy()
    row[query_idx] = -np.inf
    top = np.argsort(-row)[:k]
    return top


def build_comparison_grid(query_asin, query_idx, raw_sims, modelB_sims, blended_sims,
                           asins, title_map, image_path_map, out_path):
    rows_config = [
        ("Raw SigLIP", raw_sims),
        ("Model B (hard negs)", modelB_sims),
        ("Blended (alpha=0.5)", blended_sims),
    ]
    fig, axes = plt.subplots(len(rows_config), TOP_K + 1, figsize=(3 * (TOP_K + 1), 3.2 * len(rows_config)))

    for row_i, (label, sims) in enumerate(rows_config):
        # query image in first column, same for every row
        ax = axes[row_i, 0]
        img_path = EVAL_DIR / image_path_map[query_asin]
        try:
            ax.imshow(Image.open(img_path).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "[missing]", ha="center", va="center")
        ax.axis("off")
        qtitle = str(title_map.get(query_asin, ""))[:30]
        ax.set_title(f"QUERY\n{query_asin}\n{qtitle}", fontsize=7)
        ax.text(-0.3, 0.5, label, transform=ax.transAxes, fontsize=10, ha="right", va="center", weight="bold")

        top = top_k_excluding_self(sims, query_idx, TOP_K)
        for col_i, j in enumerate(top):
            ax2 = axes[row_i, col_i + 1]
            asin_j = asins[j]
            img_path_j = EVAL_DIR / image_path_map[asin_j]
            try:
                ax2.imshow(Image.open(img_path_j).convert("RGB"))
            except Exception:
                ax2.text(0.5, 0.5, "[missing]", ha="center", va="center")
            ax2.axis("off")
            title_j = str(title_map.get(asin_j, ""))[:28]
            ax2.set_title(f"#{col_i+1} sim={sims[query_idx, j]:.3f}\n{asin_j}\n{title_j}", fontsize=6)

    fig.suptitle(f"Query: {query_asin}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"Saved {out_path}")


def main():
    asins, raw_embeddings, title_map, image_path_map = load_data()
    raw_sims = raw_embeddings @ raw_embeddings.T

    model_b = load_projection("model_b_hard_negs")
    z_b = project(model_b, raw_embeddings)
    modelB_sims = z_b @ z_b.T

    blended_sims = 0.5 * raw_sims + 0.5 * modelB_sims

    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    query_indices = rng.choice(len(asins), size=N_QUERIES, replace=False)

    for qi in query_indices:
        query_asin = asins[qi]
        out_path = OUT_DIR / f"query_{query_asin}.png"
        build_comparison_grid(query_asin, qi, raw_sims, modelB_sims, blended_sims,
                               asins, title_map, image_path_map, out_path)

    print(f"\nSaved {N_QUERIES} comparison grids to {OUT_DIR}")


if __name__ == "__main__":
    main()
