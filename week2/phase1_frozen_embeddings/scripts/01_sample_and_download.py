import gzip
import json
import random
import csv
import time
from collections import Counter
from pathlib import Path

import requests
import urllib3
from tqdm import tqdm

META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz"
SEED = 42
TARGET_MIN = 500
TARGET_MAX = 800
N_INITIAL_SEEDS = 30

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "data" / "images"
SAMPLE_CSV = BASE_DIR / "data" / "sample_data.csv"
CATEGORY_REPORT = BASE_DIR / "data" / "category_distribution.md"
FAILED_LOG = BASE_DIR / "logs" / "failed_downloads.log"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)

def first_image_url(rec: dict):
    img = rec.get("imageURLHighRes") or rec.get("imageURL")
    if isinstance(img, list):
        return img[0] if img else None
    return img

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
    return {
        "title": rec.get("title", ""),
        "image_url": first_image_url(rec),
        "categories": rec.get("category", []),
        "also_buy": rec.get("also_buy") or [],
        "also_viewed": rec.get("also_viewed") or [],
    }

def build_seed_pool():
    pool = {}
    for rec in stream_metadata("pass 1/N: indexing seed-eligible products"):
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

def targeted_pass(wanted_asins, round_num):
    wanted = set(wanted_asins)
    found = {}
    for rec in stream_metadata(f"pass {round_num}/N: resolving {len(wanted)} neighbor asins"):
        asin = rec.get("asin")
        if asin not in wanted:
            continue
        img = first_image_url(rec)
        if not img:
            continue
        found[asin] = to_record(rec)
    return found

def bfs_sample(seed_pool):
    rng = random.Random(SEED)
    seed_eligible = list(seed_pool.keys())
    rng.shuffle(seed_eligible)
    seed_ptr = 0

    included = {}
    queue = []

    def add_seed():
        nonlocal seed_ptr
        while seed_ptr < len(seed_eligible):
            asin = seed_eligible[seed_ptr]
            seed_ptr += 1
            if asin not in included:
                included[asin] = seed_pool[asin]
                queue.append(asin)
                return True
        return False

    for _ in range(N_INITIAL_SEEDS):
        add_seed()

    leaf_wanted = set()

    qi = 0
    while qi < len(queue) and len(included) < TARGET_MAX:
        asin = queue[qi]
        qi += 1
        rec = included[asin]
        for nb in (set(rec["also_buy"]) | set(rec["also_viewed"])):
            if len(included) >= TARGET_MAX:
                break
            if nb in included:
                continue
            if nb in seed_pool:
                included[nb] = seed_pool[nb]
                queue.append(nb)
            else:
                leaf_wanted.add(nb)

        if qi >= len(queue) and len(included) < TARGET_MIN:
            if not add_seed():
                break

    print(f"In-memory BFS: {len(included)} products (target {TARGET_MIN}-{TARGET_MAX}); "
          f"{len(leaf_wanted)} candidate leaf asins outside the seed-eligible pool.")

    if len(included) < TARGET_MIN and leaf_wanted:
        leaf_wanted -= included.keys()
        print(f"Still short of target; resolving {len(leaf_wanted)} leaf asins via one network pass...")
        found = with_retries(lambda w=leaf_wanted: targeted_pass(w, "leaf"))
        for asin, rec in found.items():
            if asin in included:
                continue
            included[asin] = rec
            if len(included) >= TARGET_MAX:
                break

    print(f"Final BFS sample size: {len(included)} (target {TARGET_MIN}-{TARGET_MAX})")
    return included

def report_category_distribution(included):
    counts = Counter()
    for rec in included.values():
        cats = rec["categories"]
        dept = cats[1] if len(cats) > 1 else (cats[0] if cats else "(none)")
        counts[dept] += 1

    total = sum(counts.values())
    lines = ["# Sample Category Distribution", "", "| Category | Count | % of sample |", "|---|---|---|"]
    for cat, n in counts.most_common():
        lines.append(f"| {cat} | {n} | {100 * n / total:.1f}% |")
    CATEGORY_REPORT.write_text("\n".join(lines) + "\n")
    print(f"Saved category distribution to {CATEGORY_REPORT}")
    for cat, n in counts.most_common(10):
        print(f"  {cat}: {n} ({100 * n / total:.1f}%)")

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
        })

    if failed:
        with open(FAILED_LOG, "w") as f:
            for asin, url, err in failed:
                f.write(f"{asin}\t{url}\t{err}\n")

    print(f"Downloaded {len(rows)} images, {len(failed)} failed (see {FAILED_LOG}).")
    return rows

def verify_relatedness_overlap(rows):
    asins = {row["asin"] for row in rows}
    overlap = 0
    total = 0
    for row in rows:
        related = set(json.loads(row["also_buy"])) | set(json.loads(row["also_viewed"]))
        total += len(related)
        overlap += len(related & asins)
    pct = 100 * overlap / total if total else 0.0
    print(f"Relatedness overlap check: {overlap} of {total} references point inside the sample "
          f"({pct:.1f}%). This must be > 0 for Hit Rate@K to be meaningful.")

def main():
    t0 = time.time()
    seed_pool = with_retries(build_seed_pool)
    included = bfs_sample(seed_pool)
    report_category_distribution(included)
    rows = download_images(included)
    verify_relatedness_overlap(rows)

    with open(SAMPLE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "asin", "title", "image_path", "also_buy", "also_viewed", "category"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {SAMPLE_CSV}")
    print(f"Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
