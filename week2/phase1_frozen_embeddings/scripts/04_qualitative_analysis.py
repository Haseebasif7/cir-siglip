import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "sample_data.csv"
RETRIEVAL_JSON = BASE_DIR / "data" / "retrieval_results.json"
QUAL_DIR = BASE_DIR / "qualitative_examples"

N_HIGH_HIT_EXAMPLES = 4
N_ZERO_HIT_EXAMPLES = 4

def load_title_lookup():
    df = pd.read_csv(SAMPLE_CSV)
    return dict(zip(df["asin"], df["title"])), dict(zip(df["asin"], df["image_path"]))

def pick_examples(per_query):
    sorted_by_hits = sorted(per_query, key=lambda q: q["n_hits_at_5"], reverse=True)
    high = [q for q in sorted_by_hits if q["n_hits_at_5"] > 0][:N_HIGH_HIT_EXAMPLES]
    zero = [q for q in sorted_by_hits if q["n_hits_at_5"] == 0][:N_ZERO_HIT_EXAMPLES]
    return high + zero

def safe_open(path):
    try:
        return Image.open(BASE_DIR / path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=(200, 200, 200))

def make_grid(query_asin, retrieved_asins, hit_flags, title_lookup, path_lookup, out_path, technique):
    n_cols = 1 + len(retrieved_asins)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.5))

    q_img = safe_open(path_lookup.get(query_asin, ""))
    axes[0].imshow(q_img)
    axes[0].set_title(f"QUERY\n{query_asin}", fontsize=8)
    axes[0].axis("off")

    for ax, asin, hit in zip(axes[1:], retrieved_asins, hit_flags):
        img = safe_open(path_lookup.get(asin, ""))
        ax.imshow(img)
        marker = "HIT" if hit else "miss"
        ax.set_title(f"{marker}\n{asin}", fontsize=8, color=("green" if hit else "gray"))
        ax.axis("off")

    fig.suptitle(f"{technique} — query {query_asin}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

def main():
    title_lookup, path_lookup = load_title_lookup()
    with open(RETRIEVAL_JSON) as f:
        all_retrieval = json.load(f)

    for technique, per_query in all_retrieval.items():
        out_dir = QUAL_DIR / technique
        out_dir.mkdir(parents=True, exist_ok=True)
        examples = pick_examples(per_query)
        print(f"{technique}: saving {len(examples)} example grids")
        for i, ex in enumerate(examples):
            retrieved_5 = ex["retrieved"][:5]
            hits_5 = ex["hit_flags_at_5"]
            out_path = out_dir / f"example_{i:02d}_{ex['asin']}.png"
            make_grid(ex["asin"], retrieved_5, hits_5, title_lookup, path_lookup, out_path, technique)

    print("Done. Inspect qualitative_examples/<technique>/ before writing phase1_notes.md")

if __name__ == "__main__":
    main()
