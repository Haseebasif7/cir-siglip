"""
Phase 18, step 2: the substitute-relevance signal, rebuilt on Amazon. Same
method as phase 12c's `01_build_nn_lookup.py` (unrestricted raw-SigLIP
top-K neighbors, chunked matmul, no column mask) -- that script was only
ever run on Polyvore's 251,008-item pool before. Applied here to phase 7's
24,719-item Amazon pool, the same pool used for every other Amazon signal
in this project (phases 7-9, 16-16d).
"""
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE7_DIR = BASE_DIR.parent.parent / "week3" / "phase7_learned_compatibility"
EMBEDDINGS_NPZ = PHASE7_DIR / "embeddings" / "siglip_base.npz"
OUT_NPZ = BASE_DIR / "data" / "unrestricted_nn_lookup.npz"
SIGNAL_SUMMARY_MD = BASE_DIR / "signal_construction_summary.md"

K = 50
CHUNK_SIZE = 4096
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["asins"]]
    raw = data["embeddings"]
    raw = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
    n_items = raw.shape[0]
    print(f"Loaded {n_items} raw SigLIP embeddings (phase 7 Amazon pool), dim={raw.shape[1]}. Device: {DEVICE}")

    corpus = torch.tensor(raw, device=DEVICE)

    all_indices = np.zeros((n_items, K), dtype=np.int32)
    all_sims = np.zeros((n_items, K), dtype=np.float32)

    t0 = time.time()
    n_chunks = (n_items + CHUNK_SIZE - 1) // CHUNK_SIZE
    for c in range(n_chunks):
        start = c * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_items)
        chunk = corpus[start:end]

        sims = chunk @ corpus.T
        topk_vals, topk_idx = torch.topk(sims, K + 1, dim=1)
        topk_vals = topk_vals.cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()

        for i in range(end - start):
            row_idx = topk_idx[i]
            row_val = topk_vals[i]
            self_pos = start + i
            mask = row_idx != self_pos
            kept_idx = row_idx[mask][:K]
            kept_val = row_val[mask][:K]
            if len(kept_idx) < K:
                kept_idx = np.pad(kept_idx, (0, K - len(kept_idx)), constant_values=kept_idx[-1])
                kept_val = np.pad(kept_val, (0, K - len(kept_val)), constant_values=kept_val[-1])
            all_indices[self_pos] = kept_idx
            all_sims[self_pos] = kept_val

        if (c + 1) % 2 == 0 or c == n_chunks - 1:
            elapsed = time.time() - t0
            rate = end / elapsed
            eta = (n_items - end) / rate if rate > 0 else float("nan")
            print(f"  chunk {c+1}/{n_chunks} ({end}/{n_items} items), elapsed={elapsed:.1f}s, "
                  f"rate={rate:.0f} items/s, ETA={eta:.1f}s")

    total_time = time.time() - t0
    print(f"Done in {total_time:.1f}s ({total_time/60:.2f} min).")

    np.savez(OUT_NPZ, item_ids=np.array(item_ids), indices=all_indices, sims=all_sims)
    out_size_mb = OUT_NPZ.stat().st_size / (1024 * 1024)
    print(f"Saved {OUT_NPZ} ({out_size_mb:.1f} MB)")

    n_self_found = 0
    for i in range(0, n_items, max(1, n_items // 2000)):
        if i in all_indices[i]:
            n_self_found += 1
    sim_min, sim_max, sim_mean = all_sims.min(), all_sims.max(), all_sims.mean()

    lines = [
        "",
        "# Phase 18, Step 2: Unrestricted Nearest-Neighbor Lookup (Substitute-Relevance) -- Coverage Report",
        "",
        f"Precomputed raw SigLIP top-{K} nearest neighbors, unrestricted (phase 12c's original method, "
        f"never previously run on Amazon data), for all {n_items} items in phase 7's Amazon pool. This "
        "is the substitute-relevance corner's ranking-distillation teacher.",
        "",
        f"- Build time: {total_time:.1f}s ({total_time/60:.2f} min), {n_items/total_time:.0f} items/sec, "
        f"device={DEVICE}.",
        f"- Output: `data/unrestricted_nn_lookup.npz` ({out_size_mb:.1f} MB).",
        f"- Self-inclusion check (sampled ~2000 items): {n_self_found}/2000 (expected 0).",
        f"- Neighbor similarity range: min={sim_min:.4f}, max={sim_max:.4f}, mean={sim_mean:.4f}.",
        "",
    ]
    # Append to the same signal_construction_summary.md step 1 wrote, so both new signals live in one file.
    with open(SIGNAL_SUMMARY_MD, "a") as f:
        f.write("\n".join(lines))
    print(f"Appended to {SIGNAL_SUMMARY_MD}")
    print(f"Self-inclusion check: {n_self_found}/2000 (expected 0)")


if __name__ == "__main__":
    main()
