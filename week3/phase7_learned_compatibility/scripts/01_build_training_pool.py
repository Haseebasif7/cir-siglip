"""
Phase 7, step 1: build a new, larger category-balanced training pool,
explicitly disjoint from the existing phase 1b/1c evaluation sample
(1,872 products) -- that sample must remain completely untouched, it's the
held-out test set this phase is judged against.

Adapted from week2/phase1b_category_balanced/scripts/01_stratified_sample_and_download.py.
Same BFS/snowball stratified sampling logic, same 11 category_bucket values,
same metadata-streaming/retry logic (dataset-size-agnostic, reused verbatim).

Deviations from the phase 1b script, both deliberate and documented:
1. TARGET_PER_CATEGORY raised from 175 to 3000. Two categories (Traditional &
   Cultural Wear: 997 total qualifying products catalog-wide; Uniforms, Work &
   Safety: 695 total) cannot reach 3000 even before subtracting eval overlap --
   this is expected and reported transparently (same convention phase 1b used
   for its own under-quota categories), not silently patched around.
2. N_INITIAL_SEEDS_PER_CATEGORY raised from 15 to 50: BFS needs more
   independent seeds to reach a much larger quota efficiently. The existing
   add_seed() fallback still handles queue exhaustion correctly regardless.
3. Excluded-asin filtering added directly inside build_seed_pool(): any asin
   already in the existing 1,872-product eval sample is skipped before it
   can ever enter the pool. This transitively excludes it as both a BFS seed
   and a BFS neighbor, since both derive from the filtered pool -- no other
   function needs touching for the exclusion itself.
4. download_images() parallelized with a ThreadPoolExecutor (~20 workers).
   Sequential synchronous downloads at ~27k images would take unreasonably
   long; per-image logic/output format is otherwise identical to phase 1b's
   (15s timeout, reject <500 byte responses, log failures, no retry on
   individual image failures).
"""
import gzip
import json
import random
import csv
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import urllib3
from tqdm import tqdm

META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz"
SEED = 42

TARGET_PER_CATEGORY = 3000
N_INITIAL_SEEDS_PER_CATEGORY = 50
MIN_OVERLAP_REFS_TO_TRUST = 20  # kept for parity with phase 1b's overlap report, not used to gate anything here

EVAL_SAMPLE_CSV = Path(__file__).resolve().parent.parent.parent.parent / \
    "week2" / "phase1b_category_balanced" / "data" / "sample_data.csv"

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "data" / "images"
SAMPLE_CSV = BASE_DIR / "data" / "sample_data.csv"
CATEGORY_INVENTORY_REPORT = BASE_DIR / "data" / "category_pool_inventory.md"
CATEGORY_DIST_REPORT = BASE_DIR / "data" / "category_distribution.md"
EXCLUSION_REPORT = BASE_DIR / "data" / "exclusion_check.md"
FAILED_LOG = BASE_DIR / "logs" / "failed_downloads.log"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)

DOWNLOAD_WORKERS = 20


def load_excluded_asins():
    df = pd.read_csv(EVAL_SAMPLE_CSV)
    excluded = set(df["asin"].astype(str))
    print(f"Loaded {len(excluded)} asins to exclude from the existing eval sample ({EVAL_SAMPLE_CSV}).")
    return excluded


def first_image_url(rec: dict):
    img = rec.get("imageURLHighRes") or rec.get("imageURL")
    if isinstance(img, list):
        return img[0] if img else None
    return img


def category_bucket(categories):
    if len(categories) > 1:
        return categories[1]
    if categories:
        return categories[0]
    return "(none)"


def stream_metadata(desc):
    resp = requests.get(META_URL, stream=True, timeout=(10, 120))
    resp.raise_for_status()
    with gzip.GzipFile(fileobj=resp.raw) as gz:
        for line in tqdm(gz, desc=desc, unit=" lines"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def with_retries(fn, max_attempts=6, backoff_seconds=20):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError,
                TimeoutError, OSError, EOFError) as e:
            last_exc = e
            print(f"Network error on attempt {attempt}/{max_attempts}: {e!r}. "
                  f"Retrying in {backoff_seconds}s...")
            time.sleep(backoff_seconds)
    raise last_exc


def to_record(rec):
    cats = rec.get("category", [])
    return {
        "title": rec.get("title", ""),
        "image_url": first_image_url(rec),
        "categories": cats,
        "category_bucket": category_bucket(cats),
        "also_buy": rec.get("also_buy") or [],
        "also_viewed": rec.get("also_viewed") or [],
    }


def build_seed_pool(excluded_asins):
    pool = {}
    n_excluded_seen = 0
    for rec in stream_metadata("pass 1/1: indexing seed-eligible products"):
        asin = rec.get("asin")
        if not asin:
            continue
        if asin in excluded_asins:
            n_excluded_seen += 1
            continue
        img = first_image_url(rec)
        also_buy = rec.get("also_buy") or []
        also_viewed = rec.get("also_viewed") or []
        if img and (also_buy or also_viewed):
            pool[asin] = to_record(rec)
    print(f"Seed-eligible pool: {len(pool)} products "
          f"(excluded {n_excluded_seen} asins already in the eval sample).")
    return pool, n_excluded_seen


def report_category_inventory(seed_pool):
    counts = defaultdict(int)
    for rec in seed_pool.values():
        counts[rec["category_bucket"]] += 1

    total = sum(counts.values())
    lines = ["# Category Pool Inventory (qualifying pool, post eval-exclusion, pre-sampling)", "",
              "| Category | Qualifying products available | % of pool |", "|---|---|---|"]
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {n} | {100 * n / total:.1f}% |")
    CATEGORY_INVENTORY_REPORT.write_text("\n".join(lines) + "\n")

    print(f"\nCategory pool inventory ({total} total qualifying products, post-exclusion):")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n} ({100 * n / total:.1f}%)")
    print(f"Saved to {CATEGORY_INVENTORY_REPORT}\n")

    return counts


def bfs_within_category(category_asins, seed_pool, target):
    rng = random.Random(SEED)
    eligible = list(category_asins)
    rng.shuffle(eligible)
    ptr = 0

    included = {}
    queue = []

    def add_seed():
        nonlocal ptr
        while ptr < len(eligible):
            asin = eligible[ptr]
            ptr += 1
            if asin not in included:
                included[asin] = seed_pool[asin]
                queue.append(asin)
                return True
        return False

    for _ in range(N_INITIAL_SEEDS_PER_CATEGORY):
        add_seed()

    qi = 0
    while qi < len(queue) and len(included) < target:
        asin = queue[qi]
        qi += 1
        rec = included[asin]
        for nb in (set(rec["also_buy"]) | set(rec["also_viewed"])):
            if len(included) >= target:
                break
            if nb in included or nb not in category_asins:
                continue
            included[nb] = seed_pool[nb]
            queue.append(nb)
        if qi >= len(queue) and len(included) < target:
            if not add_seed():
                break

    return included


def stratified_sample(seed_pool, category_counts):
    by_category = defaultdict(list)
    for asin, rec in seed_pool.items():
        by_category[rec["category_bucket"]].append(asin)

    combined = {}
    quota_report = []
    for cat, asins in sorted(by_category.items(), key=lambda x: -len(x[1])):
        available = len(asins)
        target = min(TARGET_PER_CATEGORY, available)
        under_quota = available < TARGET_PER_CATEGORY

        sampled = bfs_within_category(set(asins), seed_pool, target)

        n_before = len(combined)
        for asin, rec in sampled.items():
            combined[asin] = rec
        n_added = len(combined) - n_before

        quota_report.append({
            "category": cat,
            "available": available,
            "quota": TARGET_PER_CATEGORY,
            "sampled": len(sampled),
            "added_to_combined": n_added,
            "under_quota": under_quota,
        })
        flag = " (BELOW QUOTA - used all available)" if under_quota else ""
        print(f"{cat}: {len(sampled)}/{TARGET_PER_CATEGORY} sampled from {available} available{flag}")

    total_sampled = sum(q["sampled"] for q in quota_report)
    if total_sampled != len(combined):
        print(f"WARNING: {total_sampled - len(combined)} duplicate asin(s) "
              f"appeared across categories -- investigate category_bucket() logic.")

    return combined, quota_report


def _download_one(asin, rec, session):
    local_path = IMAGES_DIR / f"{asin}.jpg"
    try:
        r = session.get(rec["image_url"], timeout=15)
        r.raise_for_status()
        if len(r.content) < 500:
            raise ValueError("response too small to be a real image")
        local_path.write_bytes(r.content)
        return asin, True, None
    except Exception as e:
        return asin, False, str(e)


def download_images(included):
    rows_by_asin = {}
    failed = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research script)"})

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        futures = {
            executor.submit(_download_one, asin, rec, session): (asin, rec)
            for asin, rec in included.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="downloading images"):
            asin, rec = futures[fut]
            asin_result, ok, err = fut.result()
            if ok:
                rows_by_asin[asin] = rec
            else:
                failed.append((asin, rec["image_url"], err))

    rows = []
    for asin, rec in rows_by_asin.items():
        rows.append({
            "asin": asin,
            "title": rec["title"],
            "image_path": str((IMAGES_DIR / f"{asin}.jpg").relative_to(BASE_DIR)),
            "also_buy": json.dumps(rec["also_buy"]),
            "also_viewed": json.dumps(rec["also_viewed"]),
            "category": json.dumps(rec["categories"]),
            "category_bucket": rec["category_bucket"],
        })

    if failed:
        with open(FAILED_LOG, "w") as f:
            for asin, url, err in failed:
                f.write(f"{asin}\t{url}\t{err}\n")

    print(f"Downloaded {len(rows)} images, {len(failed)} failed (see {FAILED_LOG}).")
    return rows


def report_final_category_distribution(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["category_bucket"]] += 1
    total = sum(counts.values())

    lines = ["# Sample Category Distribution (phase 7 training pool, quota=3000)", "",
              "| Category | Count | % of sample |", "|---|---|---|"]
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {n} | {100 * n / total:.1f}% |")
    CATEGORY_DIST_REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nFinal sample category distribution ({total} products):")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n} ({100 * n / total:.1f}%)")
    print(f"Saved to {CATEGORY_DIST_REPORT}")


def verify_no_overlap(rows, excluded_asins):
    new_asins = {row["asin"] for row in rows}
    overlap = new_asins & excluded_asins
    lines = [
        "# Exclusion Check: New Training Pool vs Existing Eval Sample",
        "",
        f"Existing eval sample size (excluded set): {len(excluded_asins)}",
        f"New training pool size: {len(new_asins)}",
        f"**Overlap (should be 0): {len(overlap)}**",
        "",
    ]
    if overlap:
        lines.append(f"WARNING: overlap found! asins: {sorted(overlap)[:20]}")
    else:
        lines.append("Confirmed: zero asin overlap between the new training pool and the existing 1,872-product eval sample.")
    EXCLUSION_REPORT.write_text("\n".join(lines) + "\n")
    print(f"\n{'ERROR' if overlap else 'OK'}: {len(overlap)} overlapping asins. Saved {EXCLUSION_REPORT}")
    return len(overlap) == 0


def main():
    t0 = time.time()
    excluded_asins = load_excluded_asins()
    seed_pool, n_excluded_seen = with_retries(lambda: build_seed_pool(excluded_asins))
    category_counts = report_category_inventory(seed_pool)

    combined, quota_report = stratified_sample(seed_pool, category_counts)
    print(f"\nCombined stratified sample: {len(combined)} products "
          f"(target ~{TARGET_PER_CATEGORY * len(category_counts)} before download losses)")

    rows = download_images(combined)
    report_final_category_distribution(rows)
    ok = verify_no_overlap(rows, excluded_asins)
    if not ok:
        raise RuntimeError("Exclusion check failed -- overlap detected between new pool and eval sample. Stopping.")

    with open(SAMPLE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "asin", "title", "image_path", "also_buy", "also_viewed",
            "category", "category_bucket"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {SAMPLE_CSV}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
