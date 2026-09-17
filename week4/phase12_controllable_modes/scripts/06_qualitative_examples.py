"""
Phase 12, step 5: qualitative check. For 6 example CIR benchmark queries
(one per a spread of semantic categories, chosen deterministically -- the
first query of that category in the benchmark's own saved order, no
cherry-picking), show top-5 results under substitute mode, complement mode,
and the interpolated blend (alpha=0.5), side by side with the query's own
remaining outfit items shown as context.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cir_eval import load_benchmark, topk_for_query
from model import ControllableProjectionHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
CHECKPOINT = BASE_DIR / "models" / "controllable_modes.pt"
IMAGES_DIR = PHASE9_DIR / "data" / "images"
ITEM_METADATA_JSON = PHASE9_DIR / "data" / "polyvore_raw" / "polyvore_item_metadata.json"
OUT_DIR = BASE_DIR / "qualitative_examples"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K = 5
TARGET_CATEGORIES = ["tops", "bottoms", "shoes", "bags", "jewellery", "outerwear"]


def load_raw_embeddings():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    embeddings = data["embeddings"]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return item_ids, (embeddings / norms).astype(np.float32)


@torch.no_grad()
def project_all(model, embeddings, alpha, batch_size=4096):
    outs = []
    for start in range(0, len(embeddings), batch_size):
        x = torch.tensor(embeddings[start:start + batch_size], device=DEVICE)
        outs.append(model(x, alpha=alpha).cpu().numpy())
    return np.concatenate(outs, axis=0)


def pick_example_queries(queries):
    picked = {}
    for qi, q in enumerate(queries):
        cat = q["category"]
        if cat in TARGET_CATEGORIES and cat not in picked and len(q["query_items"]) <= 4:
            picked[cat] = q
        if len(picked) == len(TARGET_CATEGORIES):
            break
    return picked


def build_grid(query, rows_config, title_map, out_path):
    n_context = len(query["query_items"])
    n_cols = max(n_context, TOP_K) + 1  # +1 label column worth of width via first context slot
    fig, axes = plt.subplots(len(rows_config) + 1, n_cols, figsize=(3 * n_cols, 3.2 * (len(rows_config) + 1)))

    # Row 0: query context items (the outfit's other items) + true target marked
    for c in range(n_cols):
        ax = axes[0, c]
        ax.axis("off")
        if c < n_context:
            iid = query["query_items"][c]
            img_path = IMAGES_DIR / f"{iid}.jpg"
            try:
                ax.imshow(Image.open(img_path).convert("RGB"))
            except Exception:
                ax.text(0.5, 0.5, "[missing]", ha="center", va="center")
            ax.set_title(f"CONTEXT\n{str(title_map.get(iid, ''))[:28]}", fontsize=7)
    axes[0, 0].text(-0.3, 0.5, f"Query outfit\n(target: {query['category']})", transform=axes[0, 0].transAxes,
                     fontsize=9, ha="right", va="center", weight="bold")

    true_target_img = IMAGES_DIR / f"{query['target_item']}.jpg"

    for row_i, (row_label, top_ids, top_sims) in enumerate(rows_config):
        r = row_i + 1
        axes[r, 0].text(-0.3, 0.5, row_label, transform=axes[r, 0].transAxes,
                         fontsize=9, ha="right", va="center", weight="bold")
        for c in range(n_cols):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(top_ids):
                iid = top_ids[c]
                img_path = IMAGES_DIR / f"{iid}.jpg"
                try:
                    ax.imshow(Image.open(img_path).convert("RGB"))
                except Exception:
                    ax.text(0.5, 0.5, "[missing]", ha="center", va="center")
                is_true = " [TRUE TARGET]" if iid == query["target_item"] else ""
                title_j = str(title_map.get(iid, ""))[:24]
                ax.set_title(f"#{c+1} sim={top_sims[c]:.3f}{is_true}\n{title_j}", fontsize=6.5)

    fig.suptitle(f"Polyvore query outfit {query['outfit_id']}, held-out target category: {query['category']} "
                 f"(true target id {query['target_item']})", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    print(f"Saved {out_path}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    pools, queries = load_benchmark()
    item_ids, raw_emb = load_raw_embeddings()

    with open(ITEM_METADATA_JSON) as f:
        meta = json.load(f)
    title_map = {k: v.get("url_name", "") for k, v in meta.items()}

    model = ControllableProjectionHead().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    sub_emb = project_all(model, raw_emb, alpha=1.0)
    comp_emb = project_all(model, raw_emb, alpha=0.0)
    blend_emb = project_all(model, raw_emb, alpha=0.5)

    picked = pick_example_queries(queries)
    print(f"Picked {len(picked)} example queries: {list(picked.keys())}")

    for cat, query in picked.items():
        rows_config = []
        for label, emb in [
            ("Substitute mode\n(alpha=1.0)", sub_emb),
            ("Complement mode\n(alpha=0.0)", comp_emb),
            ("Blend\n(alpha=0.5)", blend_emb),
        ]:
            top_ids, top_sims = topk_for_query(pools, query, item_ids, emb, k=TOP_K)
            rows_config.append((label, top_ids, top_sims))

        out_path = OUT_DIR / f"polyvore_query_{query['outfit_id']}_{cat}.png"
        build_grid(query, rows_config, title_map, out_path)


if __name__ == "__main__":
    main()
