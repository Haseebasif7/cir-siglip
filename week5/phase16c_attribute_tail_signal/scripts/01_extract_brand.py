"""Step 1a: extract the `brand` attribute for phase 7's 24,719-item training
pool from the raw Amazon metadata (not present in sample_data_cleaned.csv).
Single streaming pass, retry-wrapped exactly like phase 3's
01_popularity_lookup.py -- same source file, same failure modes to guard
against. Only training-pool asins are kept (a targeted single-pass filter,
not a full second copy of the catalog), since attribute-based training data
only ever needs to be constructed within items that already have an
embedding (phase 7's pool, mirroring how also_buy training was also
restricted to that same pool).
"""
import gzip
import json
import time
from pathlib import Path

import pandas as pd
import requests
import urllib3
from tqdm import tqdm

META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Clothing_Shoes_and_Jewelry.json.gz"

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
POOL_CSV = PHASE7_DIR / "data" / "sample_data_cleaned.csv"

OUT_BRAND_JSON = BASE_DIR / "data" / "brand_lookup.json"
OUT_REPORT = BASE_DIR / "logs" / "brand_extraction_report.md"


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
            print(f"Network error on attempt {attempt}/{max_attempts}: {e!r}. Retrying in {backoff_seconds}s...")
            time.sleep(backoff_seconds)
    raise last_exc


def main():
    pool_df = pd.read_csv(POOL_CSV, usecols=["asin"])
    target_asins = set(pool_df["asin"].astype(str))
    print(f"Target pool: {len(target_asins)} asins")

    brand_lookup = {}
    n_records_seen = 0
    n_matched = 0
    n_matched_nonempty = 0

    def run():
        nonlocal n_records_seen, n_matched, n_matched_nonempty
        for rec in stream_metadata("streaming pass: extracting brand for pool asins"):
            asin = rec.get("asin")
            if not asin:
                continue
            n_records_seen += 1
            if asin not in target_asins or asin in brand_lookup:
                continue
            n_matched += 1
            brand = rec.get("brand")
            if brand and str(brand).strip():
                brand_lookup[asin] = str(brand).strip()
                n_matched_nonempty += 1

    t0 = time.time()
    with_retries(run)
    elapsed = time.time() - t0

    coverage = n_matched_nonempty / len(target_asins)
    print(f"Streamed {n_records_seen} total records in {elapsed:.1f}s. "
          f"Matched {n_matched}/{len(target_asins)} pool asins, "
          f"{n_matched_nonempty} with a nonempty brand ({coverage*100:.2f}% coverage).")

    with open(OUT_BRAND_JSON, "w") as f:
        json.dump(brand_lookup, f)

    lines = [
        "# Phase 16c: Brand Attribute Extraction Report\n",
        f"Source: `{META_URL}` (same raw metadata source used throughout this project), single "
        "retry-wrapped streaming pass, filtered to phase 7's {0}-item training pool.\n".format(len(target_asins)),
        f"- Total records streamed: {n_records_seen}",
        f"- Elapsed: {elapsed:.1f}s",
        f"- Pool asins found as a record in this file: {n_matched}/{len(target_asins)}",
        f"- Pool asins with a nonempty brand value: **{n_matched_nonempty}/{len(target_asins)} "
        f"({coverage*100:.2f}%)**",
        "",
    ]
    if coverage < 0.30:
        lines.append("**Brand coverage is too low to use as a reliable primary attribute** -- flagging this "
                     "directly per the brief's own instruction, rather than proceeding on incomplete metadata "
                     "without reporting it. Fine-grained category (already available via phase 8's "
                     "product_types.json) will carry more of the pair-construction weight in step 1b.")
    else:
        lines.append("Brand coverage is high enough to use as a real, second attribute signal alongside "
                     "fine-grained category for pair construction in step 1b.")
    lines.append("")

    OUT_REPORT.write_text("\n".join(lines))
    print(f"Saved: {OUT_BRAND_JSON}")
    print(f"Report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
