import gzip
import json
import random
import csv
import time
from collections import defaultdict
from pathlib import Path

import requests
import urllib3
from tqdm import tqdm

META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz"
SEED = 42

TARGET_PER_CATEGORY = 175
N_INITIAL_SEEDS_PER_CATEGORY = 15
MIN_OVERLAP_REFS_TO_TRUST = 20

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "data" / "images"
SAMPLE_CSV = BASE_DIR / "data" / "sample_data.csv"
CATEGORY_INVENTORY_REPORT = BASE_DIR / "data" / "category_pool_inventory.md"
CATEGORY_DIST_REPORT = BASE_DIR / "data" / "category_distribution.md"
OVERLAP_REPORT = BASE_DIR / "data" / "relatedness_overlap.md"
FAILED_LOG = BASE_DIR / "logs" / "failed_downloads.log"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)

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

def build_seed_pool():
    pool = {}
    for rec in stream_metadata("pass 1/1: indexing seed-eligible products"):
        asin = rec.get("asin")
        if not asin:
            continue
        img = first_image_url(rec)
        also_buy = rec.get("also_buy") or []
        also_viewed = rec.get("also_viewed") or []
        if img and (also_buy or also_viewed):
            pool[asin] = to_record(rec)
    print(f"Seed-eligible pool: {len(pool)} products.")
    return pool

def report_category_inventory(seed_pool):
    counts = defaultdict(int)
    for rec in seed_pool.values():
        counts[rec["category_bucket"]] += 1

    total = sum(counts.values())
    lines = ["# Category Pool Inventory (full qualifying pool, pre-sampling)", "",
              "| Category | Qualifying products available | % of pool |", "|---|---|---|"]
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {n} | {100 * n / total:.1f}% |")
    CATEGORY_INVENTORY_REPORT.write_text("\n".join(lines) + "\n")

    print(f"\nCategory pool inventory ({total} total qualifying products):")
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

def download_images(included):
    rows = []
    failed = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research script)"})

    for asin, rec in tqdm(included.items(), desc="downloading images"):
        local_path = IMAGES_DIR / f"{asin}.jpg"
        try:
            r = session.get(rec["image_url"], timeout=15)
            r.raise_for_status()
            if len(r.content) < 500:
                raise ValueError("response too small to be a real image")
            local_path.write_bytes(r.content)
        except Exception as e:
            failed.append((asin, rec["image_url"], str(e)))
            continue

        rows.append({
            "asin": asin,
            "title": rec["title"],
            "image_path": str(local_path.relative_to(BASE_DIR)),
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

    lines = ["# Sample Category Distribution (phase 1b, category-balanced, quota=175)", "",
              "| Category | Count | % of sample |", "|---|---|---|"]
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {n} | {100 * n / total:.1f}% |")
    CATEGORY_DIST_REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nFinal sample category distribution ({total} products):")
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n} ({100 * n / total:.1f}%)")
    print(f"Saved to {CATEGORY_DIST_REPORT}")

def verify_relatedness_overlap(rows):
    asins = {row["asin"] for row in rows}
    overall = {"overlap": 0, "total": 0}
    per_cat = defaultdict(lambda: {"overlap": 0, "total": 0})

    for row in rows:
        related = set(json.loads(row["also_buy"])) | set(json.loads(row["also_viewed"]))
        overlap = len(related & asins)
        overall["overlap"] += overlap
        overall["total"] += len(related)
        cat = row["category_bucket"]
        per_cat[cat]["overlap"] += overlap
        per_cat[cat]["total"] += len(related)

    lines = ["# Relatedness Overlap Check (phase 1b, quota=175)", "",
             f"**Overall: {overall['overlap']} of {overall['total']} references "
             f"({100 * overall['overlap'] / overall['total']:.1f}%) point inside the sample.**",
             "", "| Category | Overlap refs | Total refs | Overlap % | Trustworthy? |",
             "|---|---|---|---|---|"]

    print(f"\nOverall relatedness overlap: {overall['overlap']}/{overall['total']} "
          f"({100 * overall['overlap'] / overall['total']:.1f}%)")

    flagged = []
    for cat, d in sorted(per_cat.items(), key=lambda x: -x[1]["total"]):
        pct = 100 * d["overlap"] / d["total"] if d["total"] else 0.0
        trustworthy = d["total"] >= MIN_OVERLAP_REFS_TO_TRUST
        if not trustworthy:
            flagged.append(cat)
        lines.append(f"| {cat} | {d['overlap']} | {d['total']} | {pct:.1f}% | "
                     f"{'yes' if trustworthy else 'NO - too few references'} |")
        print(f"  {cat}: {d['overlap']}/{d['total']} ({pct:.1f}%)"
              f"{' [FLAGGED: too few refs to trust]' if not trustworthy else ''}")

    if flagged:
        lines.append("")
        lines.append(f"**Flagged (fewer than {MIN_OVERLAP_REFS_TO_TRUST} relatedness "
                     f"references, so the overlap %% is not statistically meaningful):** "
                     + ", ".join(flagged))

    OVERLAP_REPORT.write_text("\n".join(lines) + "\n")
    print(f"Saved to {OVERLAP_REPORT}")
    return flagged

def main():
    t0 = time.time()
    seed_pool = with_retries(build_seed_pool)
    category_counts = report_category_inventory(seed_pool)

    combined, quota_report = stratified_sample(seed_pool, category_counts)
    print(f"\nCombined stratified sample: {len(combined)} products "
          f"(target ~{TARGET_PER_CATEGORY * len(category_counts)} before download losses)")

    rows = download_images(combined)
    report_final_category_distribution(rows)
    flagged = verify_relatedness_overlap(rows)

    with open(SAMPLE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "asin", "title", "image_path", "also_buy", "also_viewed",
            "category", "category_bucket"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {SAMPLE_CSV}")
    if flagged:
        print(f"FLAGGED categories (too few relatedness references to trust their "
              f"per-category Hit Rate later): {flagged}")
    print(f"Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
