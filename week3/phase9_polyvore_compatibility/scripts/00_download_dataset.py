"""
Phase 9, step 0: download the Polyvore Outfits dataset from Hugging Face.

The dataset is gated (auto-approved, but requires a logged-in HF account and
a valid token) -- confirmed blocked with anonymous access (HTTP 401 on both
`curl` and `hf download --dry-run`), user resolved this by requesting access
and providing a fresh token.

Only the `nondisjoint` split is downloaded (not `disjoint` or
`maryland_polyvore_hardneg`) -- nondisjoint is the standard benchmark split
and this alone saves ~1.5GB vs. pulling everything, relevant given this
machine has only ~11GB free disk.

Staged in two calls, not one `hf download mvasil/polyvore-outfits`:
1. Small metadata/split files first (~40MB) -- fast, validates access works
   end-to-end before committing to the large parquet downloads.
2. The three nondisjoint parquets (~2.6GB total) one at a time via separate
   --include globs, so a mid-download disconnect (this project's internet
   has been dropping frequently) only costs progress on the current file --
   huggingface_hub 1.x resumes each file automatically via HTTP Range
   requests on retry, and re-running this script is safe/cheap since
   already-complete files are skipped by etag.
"""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_DIR = BASE_DIR / "data" / "polyvore_raw"

REPO_ID = "mvasil/polyvore-outfits"

SMALL_FILE_PATTERNS = [
    "categories.csv",
    "polyvore_item_metadata.json",
    "polyvore_outfit_titles.json",
    "nondisjoint/*",
]

PARQUET_FILES = [
    "data/nondisjoint/validation.parquet",  # smallest first
    "data/nondisjoint/test.parquet",
    "data/nondisjoint/train.parquet",       # largest last
]


def run_download(include_patterns, desc):
    cmd = [
        "hf", "download", REPO_ID,
        "--repo-type", "dataset",
        "--local-dir", str(LOCAL_DIR),
    ]
    for p in include_patterns:
        cmd += ["--include", p]
    print(f"\n=== {desc} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"WARNING: download step '{desc}' exited with code {result.returncode}. "
              f"Safe to re-run this script -- already-complete files are skipped.")
        return False
    return True


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    ok = run_download(SMALL_FILE_PATTERNS, "Small metadata/split files (~40MB)")
    if not ok:
        sys.exit(1)

    for pq in PARQUET_FILES:
        ok = run_download([pq], f"Parquet: {pq}")
        if not ok:
            print(f"Stopping after failure on {pq} -- re-run this script to resume.")
            sys.exit(1)

    print("\nAll requested files downloaded successfully.")


if __name__ == "__main__":
    main()
