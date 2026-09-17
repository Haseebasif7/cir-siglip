"""
Phase 7, step 2: apply phase 6's data hygiene filters to the new training
pool before building training pairs or extracting embeddings.

Two known contamination sources (found in phase 6, week3/phase6_also_buy_similarity_split):
1. Duplicate images: phase 6 found the 10 highest-similarity also_buy edges
   in the eval sample were ALL byte-identical images (same photo reused across
   variant ASINs) -- doing MD5 dedup here, before any embedding/similarity
   computation, structurally prevents that contamination from recurring
   rather than filtering it out after the fact.
2. Corrupted records: phase 6 found ~0.85% of the eval sample had a scraped
   JS/HTML snippet as the title (e.g. "var aPageStart = ...") paired with a
   sizing-chart image instead of a real product photo.

Duplicate handling is remapped, not just dropped: when N asins share one
image, one representative asin is kept, and every also_buy edge that
touched any of the other N-1 asins is remapped to point at the
representative instead (self-loops from remapping are then dropped). This
preserves real also_buy signal that would otherwise be discarded, rather
than losing every edge that happens to touch a demoted duplicate.
"""
import ast
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_CSV = BASE_DIR / "data" / "sample_data.csv"
CLEANED_CSV = BASE_DIR / "data" / "sample_data_cleaned.csv"
DUP_MAP_JSON = BASE_DIR / "data" / "dup_map.json"
REPORT_MD = BASE_DIR / "training_pool_summary.md"

# Base pattern verified by phase 6 to catch exactly 16/1,872 (0.85%) of the
# existing eval sample. Word-boundary-anchored here (\bvar\s, \bfunction\() --
# a plain "var " substring match false-positived on this new pool's larger
# vocabulary (brand name "Manyavar Men's..." contains "var " as a substring
# of "yavar", 1/188 flags before this fix); \b prevents matching inside a
# larger word since there's no boundary between two word characters (a, v).
CORRUPTED_TITLE_RE = re.compile(r"\bvar\s|\bfunction\(|<script", re.IGNORECASE)


def md5_of(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    df = pd.read_csv(SAMPLE_CSV)
    n_start = len(df)
    print(f"Loaded {n_start} products.")

    # --- Filter 1: corrupted records (title matches scraping-artifact pattern) ---
    is_corrupted = df["title"].astype(str).str.contains(CORRUPTED_TITLE_RE, regex=True)
    n_corrupted = int(is_corrupted.sum())
    corrupted_examples = df.loc[is_corrupted, ["asin", "title"]].head(5).to_dict("records")
    df = df.loc[~is_corrupted].reset_index(drop=True)
    print(f"Corrupted-title filter: removed {n_corrupted} products ({100*n_corrupted/n_start:.2f}%).")
    for ex in corrupted_examples:
        print(f"  example removed: {ex['asin']}: {ex['title'][:80]!r}")

    # --- Filter 2: duplicate images (MD5 hash -> keep one representative) ---
    hash_to_asins = defaultdict(list)
    missing_image = []
    for _, row in df.iterrows():
        img_path = BASE_DIR / row["image_path"]
        if not img_path.exists():
            missing_image.append(row["asin"])
            continue
        h = md5_of(img_path)
        hash_to_asins[h].append(row["asin"])

    dup_map = {}  # asin -> representative asin
    n_dup_groups = 0
    n_dup_demoted = 0
    for h, asins in hash_to_asins.items():
        rep = sorted(asins)[0]  # deterministic representative choice
        for a in asins:
            dup_map[a] = rep
        if len(asins) > 1:
            n_dup_groups += 1
            n_dup_demoted += len(asins) - 1

    print(f"Duplicate-image filter: {n_dup_groups} duplicate groups found, "
          f"{n_dup_demoted} products demoted (kept 1 representative per group).")
    if missing_image:
        print(f"WARNING: {len(missing_image)} products had no local image file, skipped hashing for these.")

    # keep only representative rows for products going forward (demoted duplicates
    # are dropped from the product table but their edges get remapped below)
    df["is_representative"] = df["asin"].apply(lambda a: dup_map.get(a, a) == a)
    df_final = df.loc[df["is_representative"]].drop(columns=["is_representative"]).reset_index(drop=True)

    # --- Remap also_buy edges through dup_map, drop self-loops from remapping ---
    n_edges_before = 0
    n_edges_after = 0
    n_selfloops_dropped = 0
    remapped_also_buy = {}
    final_asins = set(df_final["asin"])
    for _, row in df.iterrows():  # iterate over all surviving (post-title-filter) rows, not just representatives
        src = row["asin"]
        rep_src = dup_map.get(src, src)
        if rep_src not in final_asins:
            continue
        also_buy = ast.literal_eval(row["also_buy"]) if pd.notna(row["also_buy"]) else []
        n_edges_before += len(also_buy)
        remapped = set()
        for tgt in also_buy:
            rep_tgt = dup_map.get(tgt, tgt)
            if rep_tgt == rep_src:
                n_selfloops_dropped += 1
                continue
            remapped.add(rep_tgt)
        remapped_also_buy.setdefault(rep_src, set()).update(remapped)

    for asin, edges in remapped_also_buy.items():
        n_edges_after += len(edges)

    df_final["also_buy"] = df_final["asin"].map(
        lambda a: json.dumps(sorted(remapped_also_buy.get(a, set())))
    )

    df_final.to_csv(CLEANED_CSV, index=False)
    with open(DUP_MAP_JSON, "w") as f:
        json.dump(dup_map, f)

    print(f"\nFinal cleaned pool: {len(df_final)} products (from {n_start} before hygiene filters).")
    print(f"also_buy edges: {n_edges_before} raw -> {n_edges_after} after dedup-remapping "
          f"({n_selfloops_dropped} self-loops dropped from remapping).")
    print(f"Saved {CLEANED_CSV}")
    print(f"Saved {DUP_MAP_JSON}")

    lines = [
        "# Phase 7: Training Pool Summary",
        "",
        "## Data hygiene filtering (step 2)",
        "",
        f"- Starting pool (after step 1 sampling + download): {n_start} products",
        f"- Corrupted-title filter (`\\bvar\\s|\\bfunction\\(|<script`, word-boundary-anchored "
        f"after a plain `var ` substring match false-positived on brand name \"Manyavar\"): "
        f"removed {n_corrupted} products ({100*n_corrupted/n_start:.2f}%)",
        f"- Duplicate-image filter (MD5 hash): {n_dup_groups} duplicate groups found, "
        f"{n_dup_demoted} products demoted to a single representative each",
        f"- **Final cleaned pool: {len(df_final)} products**",
        "",
        f"- also_buy list-total refs (not yet restricted to within-pool targets, same "
        f"convention as every prior phase's raw ref counts): {n_edges_before} raw "
        f"(post title-filter, pre-dedup-remap) -> {n_edges_after} after remapping duplicate "
        f"endpoints to their representative ({n_selfloops_dropped} self-loops dropped, i.e. "
        "edges where both endpoints collapsed to the same representative after remapping). "
        "These still include references to products outside this pool (different category, "
        "filtered out, or never in the qualifying pool at all) -- step 4 restricts to edges "
        "where both endpoints are actually in the cleaned pool with a surviving embedding, "
        "which is the trainable positive-edge count reported there.",
        "",
        "Example corrupted titles removed (up to 5 shown):",
        "",
    ]
    for ex in corrupted_examples:
        lines.append(f"- `{ex['asin']}`: {ex['title'][:100]!r}")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_MD}")


if __name__ == "__main__":
    main()
