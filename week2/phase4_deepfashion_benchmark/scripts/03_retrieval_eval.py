"""
Phase 4, Step 3: retrieval evaluation using DeepFashion's OFFICIAL query/gallery protocol.

This intentionally does NOT reuse Amazon's query-vs-everything approach (a single
in-memory `embeddings @ embeddings.T` with the diagonal masked out to exclude
self-matches). That protocol only exists because Amazon's phases had no official
query/gallery split -- every sampled product was both a possible query and a possible
candidate. DeepFashion's official split is different and must be preserved: query images
are compared against gallery images ONLY, never against other query images and never
against train images (train is already excluded entirely, from step 1). No diagonal
masking is needed here since query and gallery are disjoint image sets by construction --
there's no self-match to exclude in the first place, which is itself a structural
difference from the Amazon setup worth remembering when comparing the two numbers.

Recall@K = fraction of queries where at least one of the top-K gallery retrievals shares
the query's item_id (the standard formulation in the in-shop retrieval literature, and
the same "at least one hit in top-K" shape as Amazon's Hit Rate@K -- but computed under
a different retrieval universe, see phase4_notes.md).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
QUERY_GALLERY_CSV = BASE_DIR / "data" / "query_gallery.csv"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
RESULTS_MD = BASE_DIR / "results_table.md"
PER_QUERY_JSON = BASE_DIR / "data" / "retrieval_results.json"

K_VALUES = [1, 5, 10, 20]
TECHNIQUES = ["resnet50", "clip_vit_b32", "fashionclip", "siglip_base"]
TECHNIQUE_LABELS = {
    "resnet50": "ResNet50",
    "clip_vit_b32": "CLIP ViT-B/32",
    "fashionclip": "FashionCLIP",
    "siglip_base": "SigLIP",
}


def load_status_and_item_id():
    df = pd.read_csv(QUERY_GALLERY_CSV)
    status = dict(zip(df["image_name"], df["status"]))
    item_id = dict(zip(df["image_name"], df["item_id"]))
    return status, item_id


def evaluate_technique(technique, status, item_id):
    data = np.load(EMBEDDINGS_DIR / f"{technique}.npz", allow_pickle=True)
    all_ids = data["image_ids"]
    all_emb = data["embeddings"]

    is_query = np.array([status[i] == "query" for i in all_ids])
    is_gallery = np.array([status[i] == "gallery" for i in all_ids])

    query_ids = all_ids[is_query]
    query_emb = all_emb[is_query]
    gallery_ids = all_ids[is_gallery]
    gallery_emb = all_emb[is_gallery]

    print(f"  {technique}: {len(query_ids)} query, {len(gallery_ids)} gallery embeddings")

    # Rectangular query-vs-gallery-only similarity matrix. No diagonal masking:
    # query and gallery are disjoint sets, there is no self-match to exclude.
    sims = query_emb @ gallery_emb.T  # (n_query, n_gallery)

    max_k = max(K_VALUES)
    top_idx = np.argpartition(-sims, max_k, axis=1)[:, :max_k]
    # argpartition doesn't guarantee order within the partition -- sort just the top_k slice.
    row_idx = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[row_idx, top_idx], axis=1)
    top_idx = top_idx[row_idx, order]

    hits_at = {k: 0 for k in K_VALUES}
    per_query = []
    n_queries = len(query_ids)

    for i in range(n_queries):
        q_item = item_id[query_ids[i]]
        retrieved_gallery_ids = gallery_ids[top_idx[i]]
        retrieved_item_ids = [item_id[g] for g in retrieved_gallery_ids]
        match_flags = [1 if rid == q_item else 0 for rid in retrieved_item_ids]

        for k in K_VALUES:
            if any(match_flags[:k]):
                hits_at[k] += 1

        per_query.append({
            "query_image": str(query_ids[i]),
            "query_item_id": str(q_item),
            "retrieved_gallery_images": [str(g) for g in retrieved_gallery_ids],
            "match_flags_top20": match_flags,
        })

    recall = {f"recall@{k}": hits_at[k] / n_queries for k in K_VALUES}
    return recall, per_query


def main():
    status, item_id = load_status_and_item_id()

    all_recall = {}
    all_per_query = {}

    for technique in TECHNIQUES:
        npz_path = EMBEDDINGS_DIR / f"{technique}.npz"
        if not npz_path.exists():
            print(f"Skipping {technique}: {npz_path} not found (run step 02 first)")
            continue
        print(f"Evaluating {technique}...")
        recall, per_query = evaluate_technique(technique, status, item_id)
        all_recall[technique] = recall
        all_per_query[technique] = per_query
        print(f"  {recall}")

        # Structural sanity check: Recall@K must be monotonically non-decreasing in K.
        vals = [recall[f"recall@{k}"] for k in K_VALUES]
        if any(vals[i] > vals[i + 1] for i in range(len(vals) - 1)):
            print(f"  WARNING: Recall@K not monotonic for {technique}: {vals} -- "
                  f"check top-K slicing logic for a bug.")

    with open(PER_QUERY_JSON, "w") as f:
        json.dump(all_per_query, f)
    print(f"\nSaved {PER_QUERY_JSON}")

    lines = [
        "# Phase 4 Results: DeepFashion In-shop Retrieval (Query vs. Gallery, Official Split)",
        "",
        "Recall@K = fraction of queries where at least one of the top-K gallery images "
        "shares the query's item_id. Computed strictly query-vs-gallery (never "
        "query-vs-query, never involving train images) per the dataset's official protocol.",
        "",
        "| Technique | Recall@1 | Recall@5 | Recall@10 | Recall@20 |",
        "|---|---|---|---|---|",
    ]
    for technique in TECHNIQUES:
        if technique not in all_recall:
            continue
        r = all_recall[technique]
        lines.append(
            f"| {TECHNIQUE_LABELS[technique]} | {r['recall@1']:.3f} | {r['recall@5']:.3f} "
            f"| {r['recall@10']:.3f} | {r['recall@20']:.3f} |"
        )
    RESULTS_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {RESULTS_MD}")


if __name__ == "__main__":
    main()
