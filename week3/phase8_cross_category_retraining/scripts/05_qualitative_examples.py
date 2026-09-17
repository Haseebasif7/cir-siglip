"""
Phase 8, step 5: qualitative check, specifically revisiting phase 7's two
concrete failure cases (Tommy Hilfiger men's watch B005NGRC0W recommending
a bra under phase 7's Model B; pink drawstring laundry bag B01FWDLMYC
recommending kids' cartoon watches and a MAGA cap) plus phase 7's other 6
example queries, reused here (not fresh random ones) so the before/after
comparison is a direct, controlled visual diff rather than a new sample.

Four rows per query: Raw SigLIP / Phase 7 Model B (the original failure) /
Phase 8 Model A (heterogeneous-only, random negs -- the best performer on
the targeted cross-type metric) / Phase 8 Model B (heterogeneous-only, hard
negs -- the direct retest of the exact recipe that failed in phase 7).
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
PHASE7_DIR = BASE_DIR.parent / "phase7_learned_compatibility"
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EVAL_EMBEDDINGS_NPZ = EVAL_DIR / "embeddings" / "siglip_base.npz"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"
OUT_DIR = BASE_DIR / "qualitative_examples"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K = 5

# same 8 queries phase 7 used (seed=42 random choice) -- B005NGRC0W (watch) and
# B01FWDLMYC (laundry bag) are phase 7's two documented failure cases
QUERY_ASINS = [
    "B005NGRC0W", "B00ISL4KMW", "B01AIT77RQ", "B01FWDLMYC",
    "B01B5BIU0E", "B00505DPQG", "B000GB0G1G", "B005PQO6ZO",
]


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


def load_projection(path):
    model = ProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def project(model, embeddings):
    x = torch.tensor(embeddings.astype(np.float32), device=DEVICE)
    return model(x).cpu().numpy()


def top_k_excluding_self(sims, query_idx, k):
    row = sims[query_idx].copy()
    row[query_idx] = -np.inf
    return np.argsort(-row)[:k]


def build_comparison_grid(query_asin, query_idx, rows_config, asins, title_map, image_path_map, out_path):
    fig, axes = plt.subplots(len(rows_config), TOP_K + 1, figsize=(3 * (TOP_K + 1), 3.2 * len(rows_config)))

    for row_i, (label, sims) in enumerate(rows_config):
        ax = axes[row_i, 0]
        img_path = EVAL_DIR / image_path_map[query_asin]
        try:
            ax.imshow(Image.open(img_path).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "[missing]", ha="center", va="center")
        ax.axis("off")
        qtitle = str(title_map.get(query_asin, ""))[:30]
        ax.set_title(f"QUERY\n{query_asin}\n{qtitle}", fontsize=7)
        ax.text(-0.3, 0.5, label, transform=ax.transAxes, fontsize=9, ha="right", va="center", weight="bold")

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

    phase7_b = load_projection(PHASE7_DIR / "models" / "model_b_hard_negs.pt")
    z_p7b = project(phase7_b, raw_embeddings)
    phase7_b_sims = z_p7b @ z_p7b.T

    phase8_a = load_projection(BASE_DIR / "models" / "model_a_random_negs.pt")
    z_p8a = project(phase8_a, raw_embeddings)
    phase8_a_sims = z_p8a @ z_p8a.T

    phase8_b = load_projection(BASE_DIR / "models" / "model_b_hard_negs.pt")
    z_p8b = project(phase8_b, raw_embeddings)
    phase8_b_sims = z_p8b @ z_p8b.T

    rows_config = [
        ("Raw SigLIP", raw_sims),
        ("Phase 7 Model B\n(original failure)", phase7_b_sims),
        ("Phase 8 Model A\n(hetero, random negs)", phase8_a_sims),
        ("Phase 8 Model B\n(hetero, hard negs)", phase8_b_sims),
    ]

    OUT_DIR.mkdir(exist_ok=True)
    asin_to_idx = {a: i for i, a in enumerate(asins)}
    for query_asin in QUERY_ASINS:
        if query_asin not in asin_to_idx:
            print(f"WARNING: {query_asin} not found in eval sample, skipping.")
            continue
        qi = asin_to_idx[query_asin]
        out_path = OUT_DIR / f"query_{query_asin}.png"
        build_comparison_grid(query_asin, qi, rows_config, asins, title_map, image_path_map, out_path)

    print(f"\nSaved {len(QUERY_ASINS)} comparison grids to {OUT_DIR}")


if __name__ == "__main__":
    main()
