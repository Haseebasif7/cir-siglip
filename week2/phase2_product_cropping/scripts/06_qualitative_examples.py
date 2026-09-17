import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE1_DIR = BASE_DIR.parent / "phase1_frozen_embeddings"
PHASE1B_DIR = BASE_DIR.parent / "phase1b_category_balanced"
RETRIEVAL_JSON = BASE_DIR / "data" / "retrieval_results.json"
CASE_DIR = BASE_DIR / "qualitative_examples" / "case_study_B019SPRFPA"
GENERAL_DIR = BASE_DIR / "qualitative_examples" / "general_examples"
GENERAL_DIR.mkdir(parents=True, exist_ok=True)

ENCODER_LABELS = {"siglip_base": "SigLIP", "fashionclip": "FashionCLIP"}
VARIANT_LABELS = {
    "phase1_original": "Phase 1 original (776-pool)",
    "phase2_uncropped": "Phase 2 uncropped (same pool)",
    "phase2_cropA": "Method A: rembg",
    "phase2_cropB": "Method B: Grounding DINO",
    "baseline": "Uncropped baseline",
    "cropA": "Method A: rembg",
    "cropB": "Method B: Grounding DINO",
}
N_EACH_BUCKET = 2

def safe_open(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=(200, 200, 200))

def render_row(axes, query_img, query_label, retrieved_imgs, retrieved_asins, hit_flags, row_label):
    axes[0].imshow(query_img)
    axes[0].set_title(f"{row_label}\nQUERY", fontsize=8, loc="left")
    axes[0].axis("off")
    for ax, img, asin, hit in zip(axes[1:], retrieved_imgs, retrieved_asins, hit_flags):
        ax.imshow(img)
        marker = "HIT" if hit else "miss"
        ax.set_title(f"{marker}\n{asin}", fontsize=7, color=("green" if hit else "gray"))
        ax.axis("off")

def build_case_study_grids():
    with open(CASE_DIR / "results.json") as f:
        results = json.load(f)

    df_p1 = pd.read_csv(PHASE1_DIR / "data" / "sample_data.csv")
    path_p1 = dict(zip(df_p1["asin"], df_p1["image_path"]))
    df_p1b = pd.read_csv(PHASE1B_DIR / "data" / "sample_data.csv")
    path_p1b = dict(zip(df_p1b["asin"], df_p1b["image_path"]))

    rows_order = ["phase1_original", "phase2_uncropped", "phase2_cropA", "phase2_cropB"]
    query_img_for_row = {
        "phase1_original": safe_open(PHASE1_DIR / "data" / "images" / "B019SPRFPA.jpg"),
        "phase2_uncropped": safe_open(CASE_DIR / "query_original.jpg"),
        "phase2_cropA": safe_open(CASE_DIR / "query_cropA.jpg"),
        "phase2_cropB": safe_open(CASE_DIR / "query_cropB.jpg"),
    }

    for encoder, variants in results.items():
        fig, axes = plt.subplots(4, 6, figsize=(18, 13))
        for r, row_name in enumerate(rows_order):
            data = variants[row_name]
            retrieved = data["retrieved"]
            hits = data["hit_flags"]
            if row_name == "phase1_original":
                imgs = [safe_open(PHASE1_DIR / path_p1.get(a, "")) for a in retrieved]
            else:
                imgs = [safe_open(PHASE1B_DIR / path_p1b.get(a, "")) for a in retrieved]
            render_row(axes[r], query_img_for_row[row_name], row_name, imgs, retrieved, hits, VARIANT_LABELS[row_name])

        fig.suptitle(
            f"{ENCODER_LABELS[encoder]} — case study query B019SPRFPA "
            f"(Marc Jacobs crystal charm bracelet)", fontsize=13
        )
        fig.tight_layout()
        out_path = CASE_DIR / f"grid_{encoder}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"Saved {out_path}")

def pick_general_examples(all_retrieval):
    encoders = ["siglip_base", "fashionclip"]
    variants = ["cropA", "cropB"]

    def hit_rate5(key):
        pq = all_retrieval[key]
        return sum(1 for q in pq if q["n_hits_at_5"] > 0) / len(pq)

    best_pair, best_delta = None, -1
    for encoder in encoders:
        baseline_hr = hit_rate5(f"{encoder}_baseline")
        for variant in variants:
            delta = abs(hit_rate5(f"{encoder}_{variant}") - baseline_hr)
            if delta > best_delta:
                best_delta = delta
                best_pair = (encoder, variant)

    encoder, variant = best_pair
    print(f"Selected pairing for general examples: {encoder} + {variant} (|delta HR@5| = {best_delta:.3f})")

    baseline_by_asin = {q["asin"]: q for q in all_retrieval[f"{encoder}_baseline"]}
    variant_by_asin = {q["asin"]: q for q in all_retrieval[f"{encoder}_{variant}"]}
    common = set(baseline_by_asin) & set(variant_by_asin)

    improved, worsened, unchanged_hit = [], [], []
    for asin in common:
        b, v = baseline_by_asin[asin]["n_hits_at_5"], variant_by_asin[asin]["n_hits_at_5"]
        if b == 0 and v > 0:
            improved.append((asin, b, v))
        elif b > 0 and v == 0:
            worsened.append((asin, b, v))
        elif b > 0 and v > 0 and b == v:
            unchanged_hit.append((asin, b, v))

    improved.sort(key=lambda x: -x[2])
    worsened.sort(key=lambda x: -x[1])
    unchanged_hit.sort(key=lambda x: -x[1])

    selected = []
    for asin, b, v in improved[:N_EACH_BUCKET]:
        selected.append((asin, "improved", b, v))
    for asin, b, v in worsened[:N_EACH_BUCKET]:
        selected.append((asin, "worsened", b, v))
    remaining = 5 - len(selected)
    for asin, b, v in unchanged_hit[:remaining]:
        selected.append((asin, "unchanged", b, v))

    return encoder, variant, selected, baseline_by_asin, variant_by_asin

def build_general_example_grids():
    with open(RETRIEVAL_JSON) as f:
        all_retrieval = json.load(f)

    encoder, variant, selected, baseline_by_asin, variant_by_asin = pick_general_examples(all_retrieval)

    df_p1b = pd.read_csv(PHASE1B_DIR / "data" / "sample_data.csv")
    path_p1b = dict(zip(df_p1b["asin"], df_p1b["image_path"]))
    crop_dir = BASE_DIR / "data" / ("crops_a" if variant == "cropA" else "crops_b")

    summary_lines = [
        "# Phase 2: General Qualitative Examples",
        "",
        f"Selected pairing: **{ENCODER_LABELS[encoder]} + {VARIANT_LABELS[variant]}** "
        f"(largest overall Hit Rate@5 swing vs its uncropped baseline).",
        "",
        "| # | ASIN | Case | Baseline hits@5 | Cropped hits@5 |",
        "|---|---|---|---|---|",
    ]

    for i, (asin, case, b, v) in enumerate(selected):
        fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))

        b_q = safe_open(PHASE1B_DIR / path_p1b.get(asin, ""))
        b_retrieved = baseline_by_asin[asin]["retrieved"][:5]
        b_hits = baseline_by_asin[asin]["hit_flags_at_5"]
        b_imgs = [safe_open(PHASE1B_DIR / path_p1b.get(a, "")) for a in b_retrieved]
        render_row(axes[0], b_q, None, b_imgs, b_retrieved, b_hits, VARIANT_LABELS["baseline"])

        v_q = safe_open(crop_dir / f"{asin}.jpg")
        v_retrieved = variant_by_asin[asin]["retrieved"][:5]
        v_hits = variant_by_asin[asin]["hit_flags_at_5"]
        v_imgs = [safe_open(crop_dir / f"{a}.jpg") for a in v_retrieved]
        render_row(axes[1], v_q, None, v_imgs, v_retrieved, v_hits, VARIANT_LABELS[variant])

        fig.suptitle(f"{ENCODER_LABELS[encoder]} — {case} — query {asin}", fontsize=12)
        fig.tight_layout()
        out_path = GENERAL_DIR / f"example_{i:02d}_{case}_{asin}.png"
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"Saved {out_path}")

        summary_lines.append(f"| {i} | {asin} | {case} | {b} | {v} |")

    (GENERAL_DIR / "summary.md").write_text("\n".join(summary_lines) + "\n")
    print(f"Saved {GENERAL_DIR / 'summary.md'}")

def main():
    build_case_study_grids()
    build_general_example_grids()

if __name__ == "__main__":
    main()
