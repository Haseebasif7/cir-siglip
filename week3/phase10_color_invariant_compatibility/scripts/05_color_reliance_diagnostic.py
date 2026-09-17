"""
Phase 10, step 4.3: direct color-reliance diagnostic. Phase 9 only showed
the color-matching shortcut qualitatively (a pink laundry bag retrieving
bright-pink items regardless of category); this quantifies it.

For a sample of item pairs drawn from both domains --
  (a) Polyvore: pairs of items that co-occur in a real test-split outfit
      (label=1 lines of compatibility_test.txt), capped at a random sample
      for tractability;
  (b) Amazon: also_buy edges within the untouched eval sample (both
      endpoints present, deduplicated undirected pairs) --
compute, per pair: (1) each model's predicted compatibility score (cosine
similarity of projected embeddings; raw SigLIP uses the raw embedding
cosine directly, no projection), and (2) a raw color-similarity measure
between the two product images (2D hue/saturation histogram intersection in
HSV space, a standard, simple color-similarity convention -- ranges
[0, 1], higher = more similar coloring).

Then reports the Pearson correlation between each model's predicted score
and raw color similarity, across the sampled pairs, for: raw SigLIP,
phase 8's Amazon-trained Model A, phase 9's naive Polyvore-trained Model A,
and this phase's color-invariant model. A lower correlation for this
phase's model than phase 9's Model A would directly confirm the invariance
training reduced reliance on color matching.

All four "models" are frozen-SigLIP-based (raw, or raw + a small MLP
projection), so any model can be applied to either domain's SigLIP
embeddings directly -- no retraining or re-extraction needed here.
"""
import ast
import json
import random
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE8_DIR = BASE_DIR.parent / "phase8_cross_category_retraining"
PHASE9_DIR = BASE_DIR.parent / "phase9_polyvore_compatibility"
EVAL_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"

POLYVORE_IMAGES_DIR = PHASE9_DIR / "data" / "images"
POLYVORE_EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
POLYVORE_RAW_DIR = PHASE9_DIR / "data" / "polyvore_raw"

EVAL_EMBEDDINGS_NPZ = EVAL_DIR / "embeddings" / "siglip_base.npz"
EVAL_SAMPLE_CSV = EVAL_DIR / "data" / "sample_data.csv"

RESULTS_MD = BASE_DIR / "results_table.md"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42
N_POLYVORE_PAIRS = 3000
H_BINS, S_BINS = 16, 4


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


def color_histogram(img_path):
    img = Image.open(img_path).convert("RGB").convert("HSV")
    arr = np.asarray(img, dtype=np.float32)
    h, s = arr[..., 0].ravel(), arr[..., 1].ravel()
    hist, _, _ = np.histogram2d(h, s, bins=[H_BINS, S_BINS], range=[[0, 256], [0, 256]])
    hist = hist / (hist.sum() + 1e-8)
    return hist.ravel()


def hist_intersection(h1, h2):
    return float(np.minimum(h1, h2).sum())


def sample_polyvore_pairs(rng):
    resolver = {}
    with open(POLYVORE_RAW_DIR / "nondisjoint" / "test.json") as f:
        outfits = json.load(f)
    for outfit in outfits:
        set_id = outfit["set_id"]
        for it in outfit["items"]:
            resolver[f"{set_id}_{it['index']}"] = it["item_id"]

    candidate_pairs = []
    with open(POLYVORE_RAW_DIR / "nondisjoint" / "compatibility_test.txt") as f:
        for line in f:
            parts = line.split()
            if int(parts[0]) != 1:
                continue  # real outfits only
            items = [resolver.get(r) for r in parts[1:]]
            items = [i for i in items if i is not None]
            for a, b in combinations(items, 2):
                candidate_pairs.append((a, b))
    rng.shuffle(candidate_pairs)
    return candidate_pairs[:N_POLYVORE_PAIRS]


def sample_amazon_pairs():
    df = pd.read_csv(EVAL_SAMPLE_CSV)
    asin_set = set(df["asin"])
    also_buy = {}
    image_path_map = dict(zip(df["asin"], df["image_path"]))
    for _, row in df.iterrows():
        also_buy[row["asin"]] = set(ast.literal_eval(row["also_buy"])) if pd.notna(row["also_buy"]) else set()

    seen = set()
    pairs = []
    for a, related in also_buy.items():
        for b in related:
            if b not in asin_set or a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs, image_path_map


def compute_domain_results(pairs, item_ids, image_path_fn, embeddings, models):
    idx = {a: i for i, a in enumerate(item_ids)}
    projections = {}
    for name, model in models.items():
        if model is None:
            projections[name] = embeddings  # raw SigLIP, already L2-normalized
        else:
            projections[name] = project(model, embeddings)

    color_sims, model_scores = [], {name: [] for name in models}
    n_skipped = 0
    hist_cache = {}
    for a, b in pairs:
        if a not in idx or b not in idx:
            n_skipped += 1
            continue
        try:
            if a not in hist_cache:
                hist_cache[a] = color_histogram(image_path_fn(a))
            if b not in hist_cache:
                hist_cache[b] = color_histogram(image_path_fn(b))
        except Exception:
            n_skipped += 1
            continue
        color_sims.append(hist_intersection(hist_cache[a], hist_cache[b]))
        for name, proj in projections.items():
            va, vb = proj[idx[a]], proj[idx[b]]
            model_scores[name].append(float(np.dot(va, vb)))

    return color_sims, model_scores, n_skipped


def main():
    rng = random.Random(SEED)

    # --- load all embeddings and models once ---
    poly_data = np.load(POLYVORE_EMBEDDINGS_NPZ, allow_pickle=True)
    poly_item_ids = [str(a) for a in poly_data["item_ids"]]
    poly_embeddings = poly_data["embeddings"]
    poly_embeddings = poly_embeddings / np.linalg.norm(poly_embeddings, axis=1, keepdims=True)

    az_data = np.load(EVAL_EMBEDDINGS_NPZ, allow_pickle=True)
    az_asins = [str(a) for a in az_data["asins"]]
    az_embeddings = az_data["embeddings"]
    az_embeddings = az_embeddings / np.linalg.norm(az_embeddings, axis=1, keepdims=True)

    models = {
        "Raw SigLIP": None,
        "Phase 8 Model A (Amazon-trained)": load_projection(PHASE8_DIR / "models" / "model_a_random_negs.pt"),
        "Phase 9 Model A (Polyvore-trained, naive)": load_projection(PHASE9_DIR / "models" / "model_a_random_negs.pt"),
        "Phase 10 color-invariant model": load_projection(BASE_DIR / "models" / "model_color_invariant.pt"),
    }

    # --- Polyvore domain ---
    poly_pairs = sample_polyvore_pairs(rng)
    poly_color_sims, poly_scores, poly_skipped = compute_domain_results(
        poly_pairs, poly_item_ids, lambda i: POLYVORE_IMAGES_DIR / f"{i}.jpg", poly_embeddings, models)
    print(f"Polyvore: {len(poly_color_sims)} pairs scored ({poly_skipped} skipped).")

    # --- Amazon domain ---
    az_pairs, image_path_map = sample_amazon_pairs()
    az_color_sims, az_scores, az_skipped = compute_domain_results(
        az_pairs, az_asins, lambda i: EVAL_DIR / image_path_map[i], az_embeddings, models)
    print(f"Amazon: {len(az_color_sims)} pairs scored ({az_skipped} skipped).")

    def corr_row(color_sims, scores_dict):
        row = {}
        for name, scores in scores_dict.items():
            if len(scores) < 3 or np.std(color_sims) == 0 or np.std(scores) == 0:
                row[name] = None
            else:
                r, _ = pearsonr(scores, color_sims)
                row[name] = r
        return row

    poly_corr = corr_row(poly_color_sims, poly_scores)
    az_corr = corr_row(az_color_sims, az_scores)

    combined_color = poly_color_sims + az_color_sims
    combined_scores = {name: poly_scores[name] + az_scores[name] for name in models}
    combined_corr = corr_row(combined_color, combined_scores)

    def fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    lines = [
        "",
        "# Phase 10, Step 4.3: Direct Color-Reliance Diagnostic",
        "",
        "Pearson correlation between each model's predicted compatibility score (cosine "
        "similarity, projected for trained models / raw for SigLIP) and raw color similarity "
        "(HSV hue/saturation 2D histogram intersection) across sampled item pairs. Lower = "
        "less reliant on color matching as a proxy for compatibility.",
        "",
        f"- Polyvore domain: {len(poly_color_sims)} pairs (real test-split outfit co-occurrences, "
        f"randomly sampled from all such pairs, capped at {N_POLYVORE_PAIRS}).",
        f"- Amazon domain: {len(az_color_sims)} pairs (all also_buy edges within the untouched "
        f"eval sample, both endpoints present, deduplicated).",
        "",
        "| Configuration | Polyvore-domain r | Amazon-domain r | Combined r |",
        "|---|---|---|---|",
    ]
    for name in models:
        lines.append(f"| {name} | {fmt(poly_corr[name])} | {fmt(az_corr[name])} | {fmt(combined_corr[name])} |")
    lines.append("")

    RESULTS_MD_exists = RESULTS_MD.exists()
    with open(RESULTS_MD, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nAppended color-reliance diagnostic to {RESULTS_MD}")

    # also save raw numbers for the notes file to reference precisely
    out = {
        "polyvore": {"n_pairs": len(poly_color_sims), "correlations": poly_corr},
        "amazon": {"n_pairs": len(az_color_sims), "correlations": az_corr},
        "combined": {"n_pairs": len(combined_color), "correlations": combined_corr},
    }
    with open(BASE_DIR / "data" / "color_reliance_diagnostic.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {BASE_DIR / 'data' / 'color_reliance_diagnostic.json'}")


if __name__ == "__main__":
    main()
