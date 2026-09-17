"""
Phase 6, step 3: pull the 10 highest- and 10 lowest-similarity also_buy edges
and build image grids (query next to its also_buy partner) for visual
inspection -- the key check for whether the similarity distribution's shape
actually corresponds to a real substitute (near-duplicate) vs complement
(different but paired) distinction, or not.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE1B_DIR = BASE_DIR.parent.parent / "week2" / "phase1b_category_balanced"
EDGE_JSON = BASE_DIR / "data" / "edge_similarities_siglip.json"
OUT_DIR = BASE_DIR / "qualitative_examples"


def build_grid(edges, title, out_path, title_map, image_path_map):
    n = len(edges)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    for row, e in enumerate(edges):
        for col, asin in enumerate([e["source"], e["target"]]):
            ax = axes[row, col]
            img_path = PHASE1B_DIR / image_path_map[asin]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception as ex:
                ax.text(0.5, 0.5, f"[missing image]\n{ex}", ha="center", va="center")
            ax.axis("off")
            label = "query" if col == 0 else "also_buy partner"
            name = title_map.get(asin, asin)
            short_name = (name[:40] + "...") if len(name) > 43 else name
            ax.set_title(f"{label}: {asin}\n{short_name}", fontsize=7)
        axes[row, 0].text(-0.15, 0.5, f"sim={e['similarity']:.3f}\n{e['source_category']}->{e['target_category']}",
                           transform=axes[row, 0].transAxes, fontsize=8, ha="right", va="center")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0.05, 0, 1, 0.98])
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"Saved {out_path}")


def main(technique="siglip", edge_json=EDGE_JSON, out_dir=OUT_DIR):
    with open(edge_json) as f:
        d = json.load(f)
    edges = d["edges"]
    title_map = d["title"]
    image_path_map = d["image_path"]

    sorted_edges = sorted(edges, key=lambda e: e["similarity"])
    lowest = sorted_edges[:10]
    highest = sorted_edges[-10:][::-1]

    out_dir.mkdir(exist_ok=True)
    build_grid(highest, f"{technique.upper()}: 10 HIGHEST-similarity also_buy pairs",
               out_dir / f"highest_similarity_{technique}.png", title_map, image_path_map)
    build_grid(lowest, f"{technique.upper()}: 10 LOWEST-similarity also_buy pairs",
               out_dir / f"lowest_similarity_{technique}.png", title_map, image_path_map)

    # also dump a small text summary (asin, titles, sim, category) for reference
    def summarize(edges_list):
        out = []
        for e in edges_list:
            out.append({
                "source": e["source"], "source_title": title_map.get(e["source"], ""),
                "source_category": e["source_category"],
                "target": e["target"], "target_title": title_map.get(e["target"], ""),
                "target_category": e["target_category"],
                "similarity": e["similarity"],
            })
        return out

    summary = {"highest": summarize(highest), "lowest": summarize(lowest)}
    with open(out_dir / f"extremes_summary_{technique}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_dir / f'extremes_summary_{technique}.json'}")

    print("\n=== HIGHEST ===")
    for e in summary["highest"]:
        print(f"{e['similarity']:.4f} | {e['source_category']}->{e['target_category']} | "
              f"{e['source_title'][:40]!r} -> {e['target_title'][:40]!r}")
    print("\n=== LOWEST ===")
    for e in summary["lowest"]:
        print(f"{e['similarity']:.4f} | {e['source_category']}->{e['target_category']} | "
              f"{e['source_title'][:40]!r} -> {e['target_title'][:40]!r}")


if __name__ == "__main__":
    main()
