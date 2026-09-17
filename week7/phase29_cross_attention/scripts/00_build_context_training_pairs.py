"""
Phase 29, step 0: build MULTI-ITEM context training examples from the raw
Polyvore outfit files, instead of reusing phase 9/23/25/26/27/28's flattened
pairwise positive_edges.json directly.

Why this is necessary, not a scope-creep add-on
-------------------------------------------------
Candidate-conditioned cross-attention only has something to learn when the
context set it attends over has more than one item: softmax over a single
logit is the constant function 1, so its gradient with respect to that
logit -- and therefore with respect to candidate_key/context_query's
weights -- is exactly zero. Phase 27/28's training data is single-item
anchor->positive pairs (context size always 1). Training this phase's model
on that data as-is would leave candidate_key and context_query at their
random initialization forever: a provable, not speculative, dead-gradient
bug, not a subtle underperformance issue. See architecture_notes.md for the
full writeup of this and the training-time conditioning simplification it
motivates.

The fix keeps "same real Polyvore outfit co-occurrence data" true in
spirit: it uses the exact same official nondisjoint train/valid outfit
files phase 9 built positive_edges.json from (week3/phase9_polyvore_
compatibility/data/polyvore_raw/nondisjoint/{train,valid}.json), just kept
in their natural outfit-grouped form instead of flattened to pairs. For
every outfit, every item is used once as a held-out "target", with every
OTHER item in that same outfit as its "context" -- exactly mirroring the
real eval-time query structure (query_items = context, target_item = held
out), unlike the old anchor->positive pairwise framing. No new leakage
risk: same official split, same item universe (filtered to the same
251,008-item pool phase 27/28's siglip_base.npz/text_embeddings.npz cover).
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_RAW = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility" / "data" / "polyvore_raw"
PHASE27_EMB = BASE_DIR.parent / "phase27_text_and_category" / "scripts"  # not used directly, path kept for reference
ITEM_IDS_NPZ = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility" / "embeddings" / "siglip_base.npz"

OUT_JSON = BASE_DIR / "data" / "context_training_pairs.json"
REPORT_MD = BASE_DIR / "training_pool_summary.md"

MAX_CONTEXT_LEN = 20  # generous cap, well above the observed max outfit size (19); never actually trims


def load_valid_items():
    import numpy as np
    d = np.load(ITEM_IDS_NPZ, allow_pickle=True)
    return set(str(a) for a in d["item_ids"])


def load_outfits(split):
    with open(PHASE9_RAW / "nondisjoint" / f"{split}.json") as f:
        return json.load(f)


def build_examples(outfits, valid_items, split_name):
    examples = []
    n_skipped_single_item = 0
    n_skipped_missing = 0
    for outfit in outfits:
        item_ids = [it["item_id"] for it in outfit["items"] if it["item_id"] in valid_items]
        n_skipped_missing += len(outfit["items"]) - len(item_ids)
        if len(item_ids) < 2:
            n_skipped_single_item += 1
            continue
        for i, target in enumerate(item_ids):
            context = item_ids[:i] + item_ids[i + 1:]
            examples.append({"context_items": context, "target_item": target, "split": split_name})
    return examples, n_skipped_single_item, n_skipped_missing


def main():
    valid_items = load_valid_items()
    print(f"{len(valid_items)} items in the shared 251,008-item embedding universe.")

    train_outfits = load_outfits("train")
    valid_outfits = load_outfits("valid")

    train_ex, n_skip_single_train, n_skip_missing_train = build_examples(train_outfits, valid_items, "train")
    val_ex, n_skip_single_val, n_skip_missing_val = build_examples(valid_outfits, valid_items, "val")

    all_ex = train_ex + val_ex
    with open(OUT_JSON, "w") as f:
        json.dump(all_ex, f)

    ctx_lens = [len(e["context_items"]) for e in all_ex]
    print(f"Train: {len(train_ex)} context examples from {len(train_outfits)} outfits "
          f"({n_skip_single_train} outfits skipped, <2 valid items after filtering; "
          f"{n_skip_missing_train} item-slots dropped, missing from embedding universe).")
    print(f"Val: {len(val_ex)} context examples from {len(valid_outfits)} outfits "
          f"({n_skip_single_val} outfits skipped, <2 valid items; {n_skip_missing_val} item-slots dropped).")
    print(f"Context length: min={min(ctx_lens)} max={max(ctx_lens)} "
          f"mean={sum(ctx_lens)/len(ctx_lens):.2f}")

    with open(REPORT_MD, "w") as f:
        f.write("# Phase 29: Context Training Pairs\n\n")
        f.write("Built from the same official Polyvore nondisjoint train/valid outfit files "
                "phase 9 used, kept in their natural multi-item outfit grouping instead of "
                "flattened to pairwise anchor->positive edges. Necessary so candidate-conditioned "
                "cross-attention has more than one context item to learn to weight during training "
                "-- see architecture_notes.md for why this is required, not optional scope.\n\n")
        f.write(f"- Shared item universe: {len(valid_items)} items (same as phase 27/28's "
                f"siglip_base.npz / text_embeddings.npz).\n")
        f.write(f"- Train: {len(train_ex)} context examples from {len(train_outfits)} outfits "
                f"({n_skip_single_train} outfits skipped for <2 valid items, "
                f"{n_skip_missing_train} item-slots dropped for missing embeddings).\n")
        f.write(f"- Val: {len(val_ex)} context examples from {len(valid_outfits)} outfits "
                f"({n_skip_single_val} outfits skipped, {n_skip_missing_val} item-slots dropped).\n")
        f.write(f"- Context length across all examples: min={min(ctx_lens)}, max={max(ctx_lens)}, "
                f"mean={sum(ctx_lens)/len(ctx_lens):.2f} (mirrors the real CIR benchmark's own "
                f"query_items length distribution, mean 4.86, since both are built from the same "
                f"underlying outfit structure).\n")
        f.write(f"- Output: `data/context_training_pairs.json` ({len(all_ex)} total examples), "
                f"to be uploaded to the Modal volume alongside phase 27/28's existing data.\n")

    print(f"\nWrote {OUT_JSON} and {REPORT_MD}")


if __name__ == "__main__":
    main()
