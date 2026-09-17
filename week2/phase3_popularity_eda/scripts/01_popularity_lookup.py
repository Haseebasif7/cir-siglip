"""
Phase 3, Step 1: catalog-wide popularity lookup.

Single streaming pass over the full metadata file (~2.68M records) to count,
for every asin, how many times it appears inside any OTHER product's
also_buy/also_viewed list. This is a reference-count popularity signal, not
a sample-level one -- must cover the whole catalog per the phase brief.

Design decisions (documented here since they affect every downstream number):
- Within a single referencing product, also_buy and also_viewed are unioned
  before counting, so a neighbor listed in both doesn't get double-counted
  as "2 references" from the same product. This treats "referenced by
  product X" as a single boolean event per X.
- Every asin that ever appears as a record's own `asin` field gets an entry
  in the lookup (defaulting to 0 if never referenced by anyone), not just
  asins that show up inside a some other product's also_buy/also_viewed.
  Otherwise the lookup would silently omit the entire never-referenced
  bottom of the distribution, which is exactly the long tail this phase
  needs to characterize.
- Dangling references (an asin referenced by someone's also_buy/also_viewed
  that never itself appears as a record in this file) still get an entry
  via the increment step. These are real edges in the graph even though we
  never saw that product's own metadata row.
"""
import gzip
import json
import resource
import time
from pathlib import Path

import requests
import urllib3
from tqdm import tqdm

META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOOKUP_CSV = DATA_DIR / "popularity_lookup.csv"
RUN_LOG = LOG_DIR / "step01_run.log"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


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


def build_counts():
    counts = {}
    n_records = 0
    n_dangling_at_end = 0

    def run():
        nonlocal n_records
        for rec in stream_metadata("catalog pass: tallying also_buy/also_viewed references"):
            asin = rec.get("asin")
            if not asin:
                continue
            n_records += 1
            counts.setdefault(asin, 0)

            refs = set(rec.get("also_buy") or []) | set(rec.get("also_viewed") or [])
            for r in refs:
                if not r:
                    continue
                counts[r] = counts.get(r, 0) + 1

    with_retries(run)
    return counts, n_records


def main():
    t0 = time.time()
    counts, n_records = build_counts()
    elapsed = time.time() - t0

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    # macOS reports ru_maxrss in bytes; Linux reports KB. Detect via sane range.
    if peak_rss_mb > 500_000:  # implausible for MB-from-bytes on a 16GB laptop -> was actually KB
        peak_rss_mb = peak_rss_mb / 1024

    n_referenced = sum(1 for c in counts.values() if c > 0)
    n_total = len(counts)

    print(f"Processed {n_records} product records.")
    print(f"Lookup table has {n_total} unique asins ({n_referenced} with ref_count > 0, "
          f"{n_total - n_referenced} with ref_count == 0).")
    print(f"Elapsed: {elapsed:.1f}s. Peak RSS: {peak_rss_mb:.0f} MB.")

    # Sorted descending by ref_count for the long-tail plot / rank-based tiering in step 2.
    with open(LOOKUP_CSV, "w") as f:
        f.write("asin,ref_count\n")
        for asin, c in sorted(counts.items(), key=lambda kv: -kv[1]):
            f.write(f"{asin},{c}\n")
    print(f"Saved {LOOKUP_CSV}")

    with open(RUN_LOG, "w") as f:
        f.write(f"Processed {n_records} product records.\n")
        f.write(f"Lookup table: {n_total} unique asins, {n_referenced} with ref_count > 0.\n")
        f.write(f"Elapsed: {elapsed:.1f}s. Peak RSS: {peak_rss_mb:.0f} MB.\n")


if __name__ == "__main__":
    main()
