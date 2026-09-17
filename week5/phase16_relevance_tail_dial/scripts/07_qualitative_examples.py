"""A handful of example queries, rendered as image grids: query image on the
left, top-5 retrieved gallery items at alpha=0.0 (tail-exposure), 0.5
(blend), and 1.0 (relevance) in three rows, with each thumbnail captioned by
its catalog-wide ref_count and tier -- makes the (real but structurally
weak, per 04/05/06) shift across alpha directly visible rather than only
described in tables.
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
from model import ControllableProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
PHASE1B_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
TIER_LOOKUP = BASE_DIR.parent.parent / "week2" / "phase3_popularity_eda" / "data" / "popularity_lookup.csv"

CHECKPOINT = BASE_DIR / "models" / "relevance_tail_dial.pt"
GALLERY_NPZ = BASE_DIR / "data" / "candidate_gallery.npz"
QUERY_NPZ = PHASE1B_DIR / "embeddings" / "siglip_base.npz"
QUERY_CSV = PHASE1B_DIR / "data" / "sample_data.csv"
IMAGE_DIRS = [PHASE1B_DIR / "data" / "images", PHASE7_DIR / "data" / "images"]
OUT_DIR = BASE_DIR / "qualitative_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
ALPHAS_SHOWN = [0.0, 0.5, 1.0]
ALPHA_LABELS = {0.0: "tail-exposure (a=0.0)", 0.5: "blend (a=0.5)", 1.0: "relevance (a=1.0)"}
TOP_K = 5
N_EXAMPLES = 6


def find_image(asin):
    for d in IMAGE_DIRS:
        p = d / f"{asin}.jpg"
        if p.exists():
            return p
    return None


def l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


def main():
    model = ControllableProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    g = np.load(GALLERY_NPZ, allow_pickle=True)
    gallery_asins = g["asins"].astype(str)
    gallery_emb = l2norm(g["embeddings"])
    gallery_idx = {a: i for i, a in enumerate(gallery_asins)}

    q = np.load(QUERY_NPZ, allow_pickle=True)
    query_asins = q["asins"].astype(str)
    query_emb = l2norm(q["embeddings"])

    tier_df = pd.read_csv(TIER_LOOKUP, usecols=["asin", "ref_count", "tier"])
    refcount_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["ref_count"].astype(int)))
    tier_lookup = dict(zip(tier_df["asin"].astype(str), tier_df["tier"]))

    query_df = pd.read_csv(QUERY_CSV, usecols=["asin", "category_bucket"])
    cat_lookup = dict(zip(query_df["asin"].astype(str), query_df["category_bucket"]))

    # pick N_EXAMPLES queries that actually have a findable local image, spread across categories seen so far
    rng = np.random.default_rng(42)
    order = rng.permutation(len(query_asins))
    picked = []
    seen_cats = set()
    for i in order:
        asin = query_asins[i]
        if find_image(asin) is None:
            continue
        cat = cat_lookup.get(asin, "unknown")
        if cat in seen_cats:
            continue
        picked.append(i)
        seen_cats.add(cat)
        if len(picked) >= N_EXAMPLES:
            break

    gallery_t = torch.tensor(gallery_emb, device=DEVICE)

    for qi in picked:
        qa = query_asins[qi]
        q_emb_t = torch.tensor(query_emb[qi:qi + 1], device=DEVICE)

        fig, axes = plt.subplots(len(ALPHAS_SHOWN) + 1, TOP_K, figsize=(3 * TOP_K, 3.2 * (len(ALPHAS_SHOWN) + 1)))

        query_img = find_image(qa)
        axes[0, 0].imshow(Image.open(query_img).convert("RGB"))
        axes[0, 0].set_title(f"QUERY: {qa}\n({cat_lookup.get(qa, '?')})", fontsize=9)
        for c in range(TOP_K):
            axes[0, c].axis("off")

        for ri, alpha in enumerate(ALPHAS_SHOWN, start=1):
            with torch.no_grad():
                z_gallery = model(gallery_t, alpha=alpha)
                z_q = model(q_emb_t, alpha=alpha)
                sims = (z_q @ z_gallery.T).squeeze(0)
                if qa in gallery_idx:
                    sims[gallery_idx[qa]] = -float("inf")
                topk_sims, topk_idx = torch.topk(sims, TOP_K)

            for c in range(TOP_K):
                gidx = topk_idx[c].item()
                asin = gallery_asins[gidx]
                img_path = find_image(asin)
                ax = axes[ri, c]
                ax.axis("off")
                if img_path is not None:
                    ax.imshow(Image.open(img_path).convert("RGB"))
                rc = refcount_lookup.get(asin, 0)
                tier = tier_lookup.get(asin, "?")
                ax.set_title(f"{asin}\nref_count={rc} ({tier})\nsim={topk_sims[c].item():.3f}", fontsize=7)
            axes[ri, 0].set_ylabel(ALPHA_LABELS[alpha], fontsize=9)

        fig.suptitle(f"Query {qa} -- top-{TOP_K} retrievals across alpha", fontsize=11)
        fig.tight_layout()
        out_path = OUT_DIR / f"example_{qa}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"Saved {out_path}")

    print(f"Done: {len(picked)} example queries rendered to {OUT_DIR}")


if __name__ == "__main__":
    main()
