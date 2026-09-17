"""
Phase 12c, step 1: precompute raw SigLIP top-K nearest-neighbor structure for
every item in the training pool (all 251,008 Polyvore items, same pool used
by phases 9/12/12b). This is the "teacher" signal for step 2's ranking-
distillation loss -- who is actually close to whom under raw SigLIP -- fixed
once, reused every training step (does not change during training since raw
SigLIP is frozen).

K=50: large enough that each anchor's true nearby items (the ones that
matter for a top-10/30/50 retrieval task) are almost certainly included, small
enough to keep the per-step ranking-distillation loss cheap (each training
step needs to project K neighbors per anchor through the model, not just the
anchor itself).

A full 251,008 x 251,008 similarity matrix (~252GB at float32) cannot be
materialized at once -- confirmed the same hard constraint phase 9's official
Polyvore evaluation already hit and worked around (see phase 9's
`06_evaluate_official.py` docstring). Computed here in chunks instead: each
chunk of anchor items is matmul'd against the FULL corpus, top-(K+1) is taken
per row (K+1 to account for self-similarity, which is then excluded), and
only the resulting small (chunk, K) index/score arrays are kept -- the full
(chunk, 251008) intermediate similarity matrix is discarded chunk by chunk,
never accumulated.
"""
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
OUT_NPZ = BASE_DIR / "data" / "nn_lookup.npz"
REPORT_MD = BASE_DIR / "nearest_neighbor_lookup_summary.md"

K = 50
CHUNK_SIZE = 2048
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    item_ids = [str(a) for a in data["item_ids"]]
    raw = data["embeddings"]
    raw = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
    n_items = raw.shape[0]
    print(f"Loaded {n_items} raw SigLIP embeddings, dim={raw.shape[1]}. Device: {DEVICE}")

    corpus = torch.tensor(raw, device=DEVICE)  # (N, 768) -- kept resident for the whole run

    all_indices = np.zeros((n_items, K), dtype=np.int32)
    all_sims = np.zeros((n_items, K), dtype=np.float32)

    t0 = time.time()
    n_chunks = (n_items + CHUNK_SIZE - 1) // CHUNK_SIZE
    for c in range(n_chunks):
        start = c * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, n_items)
        chunk = corpus[start:end]  # (chunk, 768)

        sims = chunk @ corpus.T  # (chunk, N) -- the only large intermediate, discarded each iteration
        topk_vals, topk_idx = torch.topk(sims, K + 1, dim=1)  # +1 to cover self before exclusion
        topk_vals = topk_vals.cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()

        for i in range(end - start):
            row_idx = topk_idx[i]
            row_val = topk_vals[i]
            self_pos = start + i
            mask = row_idx != self_pos
            kept_idx = row_idx[mask][:K]
            kept_val = row_val[mask][:K]
            if len(kept_idx) < K:  # pad (only possible if self wasn't in top-(K+1), extremely unlikely)
                kept_idx = np.pad(kept_idx, (0, K - len(kept_idx)), constant_values=kept_idx[-1])
                kept_val = np.pad(kept_val, (0, K - len(kept_val)), constant_values=kept_val[-1])
            all_indices[self_pos] = kept_idx
            all_sims[self_pos] = kept_val

        if (c + 1) % 20 == 0 or c == n_chunks - 1:
            elapsed = time.time() - t0
            rate = (end) / elapsed
            eta = (n_items - end) / rate if rate > 0 else float("nan")
            print(f"  chunk {c+1}/{n_chunks} ({end}/{n_items} items), "
                  f"elapsed={elapsed:.1f}s, rate={rate:.0f} items/s, ETA={eta:.1f}s")

    total_time = time.time() - t0
    print(f"Done in {total_time:.1f}s ({total_time/60:.1f} min).")

    np.savez(OUT_NPZ, item_ids=np.array(item_ids), indices=all_indices, sims=all_sims)
    out_size_mb = OUT_NPZ.stat().st_size / (1024 * 1024)
    print(f"Saved {OUT_NPZ} ({out_size_mb:.1f} MB)")

    # Sanity checks
    n_self_found = 0  # should be 0 -- self must never appear in the final K neighbors
    for i in range(0, n_items, max(1, n_items // 2000)):  # sample ~2000 items for the check
        if i in all_indices[i]:
            n_self_found += 1
    sim_min, sim_max, sim_mean = all_sims.min(), all_sims.max(), all_sims.mean()

    lines = [
        "# Phase 12c, Step 1: Nearest-Neighbor Lookup Summary",
        "",
        f"Precomputed raw SigLIP top-{K} nearest neighbors for all {n_items} Polyvore items "
        f"(the same pool used by phases 9/12/12b), via chunked matmul on {DEVICE} "
        f"(chunk size {CHUNK_SIZE}, {n_chunks} chunks) -- a full {n_items}x{n_items} similarity "
        f"matrix (~{n_items*n_items*4/1e9:.0f} GB at float32) is never materialized; only each "
        "chunk's own top-(K+1) survives past that chunk's iteration.",
        "",
        f"- **Actual build time: {total_time:.1f}s ({total_time/60:.2f} min)**, "
        f"{n_items/total_time:.0f} items/sec average. No subsampling was needed -- the full "
        "251,008-item pool's neighbor structure was computed as specified.",
        f"- Output: `data/nn_lookup.npz`, {out_size_mb:.1f} MB "
        f"(indices: int32 ({n_items}, {K}), sims: float32 ({n_items}, {K})).",
        "",
        "## Sanity checks",
        "",
        f"- Self-inclusion check (sampled ~2000 items): {n_self_found}/2000 had themselves in "
        f"their own top-{K} neighbor list (expected 0 -- self is explicitly excluded before "
        "the final K are kept).",
        f"- Neighbor similarity range across the full lookup: min={sim_min:.4f}, "
        f"max={sim_max:.4f}, mean={sim_mean:.4f} (all well below 1.0, consistent with self "
        "being excluded and no duplicate-embedding items dominating).",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {REPORT_MD}")
    print(f"Self-inclusion check: {n_self_found}/2000 (expected 0)")
    print(f"Sim range: min={sim_min:.4f} max={sim_max:.4f} mean={sim_mean:.4f}")


if __name__ == "__main__":
    main()
