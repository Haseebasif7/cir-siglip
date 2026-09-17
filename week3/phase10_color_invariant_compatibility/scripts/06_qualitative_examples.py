"""
Phase 10, step 5: qualitative check. Re-runs phase 8/9's pink drawstring
laundry bag query (B01FWDLMYC, phase 9's clearest color-matching-shortcut
case) plus 5 more queries from the same fixed list phase 8/9 used (for
direct visual comparability across all three learned-compatibility phases),
through: raw SigLIP, phase 8's Amazon-trained Model A, phase 9's naive
Polyvore-trained Model A, and this phase's color-invariant model.

All on the Amazon eval sample's SigLIP embeddings (transfer setting) --
this is the domain the color-matching shortcut misfired on.
"""
import ast
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
PHASE8_DIR = BASE_DIR.parent / "phase8_cross_category_retraining"
PHASE9_DIR = BASE_DIR.parent / "phase9_polyvore_compatibility"
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EVAL_EMBEDDINGS_NPZ = EVAL_DIR / "embeddings" / "siglip_base.npz"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"

OUT_DIR = BASE_DIR / "qualitative_examples"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K = 5

# laundry bag (phase 8/9's clearest color-shortcut case) + 5 more from the
# same fixed 8-asin list phase 8/9 used, for direct visual comparability
QUERY_ASINS = [
    "B01FWDLMYC",  # pink drawstring laundry bag
    "B005NGRC0W",
    "B00ISL4KMW",
    "B01AIT77RQ",
    "B01B5BIU0E",
    "B00505DPQG",
]


def load_projection(path):
    model = ProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def project(model, embeddings):
    x = torch.tensor(embeddings.astype(np.float32), device=DEVICE)
    z = model(x).cpu().numpy()
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def top_k_excluding_self(embeddings, query_idx, k):
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


def main():
    OUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(EVAL_SAMPLE_CSV)
    title_map = dict(zip(df["asin"], df["title"]))
    image_path_map = dict(zip(df["asin"], df["image_path"]))

    data = np.load(EVAL_EMBEDDINGS_NPZ, allow_pickle=True)
    asins = [str(a) for a in data["asins"]]
    embeddings = data["embeddings"]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    idx_of = {a: i for i, a in enumerate(asins)}

    phase8_a = load_projection(PHASE8_DIR / "models" / "model_a_random_negs.pt")
    z_p8a = project(phase8_a, embeddings)

    phase9_a = load_projection(PHASE9_DIR / "models" / "model_a_random_negs.pt")
    z_p9a = project(phase9_a, embeddings)

    phase10 = load_projection(BASE_DIR / "models" / "model_color_invariant.pt")
    z_p10 = project(phase10, embeddings)

    def get_info(j):
        asin_j = asins[j]
        return EVAL_DIR / image_path_map[asin_j], str(title_map.get(asin_j, ""))

    for query_asin in QUERY_ASINS:
        if query_asin not in idx_of:
            print(f"WARNING: {query_asin} not in Amazon eval sample, skipping.")
            continue
        qi = idx_of[query_asin]
        rows_config = [
            ("Raw SigLIP", embeddings, qi),
            ("Phase 8 Model A\n(Amazon-trained)", z_p8a, qi),
            ("Phase 9 Model A\n(Polyvore-trained,\nnaive)", z_p9a, qi),
            ("Phase 10\n(color-invariant)", z_p10, qi),
        ]
        title = str(title_map.get(query_asin, ""))
        out_path = OUT_DIR / f"amazon_query_{query_asin}.png"
        build_grid(f"Amazon query (transfer test): {query_asin}",
                   EVAL_DIR / image_path_map[query_asin], title, rows_config, get_info, out_path)


if __name__ == "__main__":
    main()
