"""
Phase 14b, step 1: verify the category-match rate of the new negative
sampling, before trusting it enough to train against. Per the brief: don't
assume the fix worked without checking.

"Before" figure: phase 14's own directly-measured 13.1% (`phase14_notes.md`,
200 random 96-outfit training batches, ALL possible in-batch negative
pairs) -- cited here, not recomputed, since it's already an established
number from this exact project on this exact data and the mechanism it
describes (unrestricted in-batch pairing) no longer exists in this phase's
training loop to re-measure.

"After" figure: computed fresh here, same 200-batch / 96-outfit-per-batch
convention as phase 14's measurement, over the NEW sample_negatives-based
negatives this phase's train_core.py actually produces.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_core import TrainState, NUM_NEGATIVES

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
PHASE13_DIR = BASE_DIR.parent / "phase13_csa_net_baseline"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
TRAINING_DATA = PHASE13_DIR / "data" / "training_data.json"
NEGATIVE_CANDIDATES = PHASE13_DIR / "data" / "negative_candidates.json"
OUT_MD = BASE_DIR / "negative_sampling_fix_summary.md"

N_BATCHES = 200
BATCH_SIZE = 96
BEFORE_MATCH_RATE = 0.131  # week4/phase14_outfittransformer_siglip/phase14_notes.md, line 29


def main():
    state = TrainState(EMBEDDINGS_NPZ, TRAINING_DATA, NEGATIVE_CANDIDATES, seed=0)
    train_outfits = [o for o in state.train_outfits if len(o["items"]) >= 2]

    total_negatives = 0
    matched_negatives = 0
    fallback_used = 0
    short_rows = 0
    negatives_per_sample = []

    for b in range(N_BATCHES):
        b_idx = [state.rng.randrange(len(train_outfits)) for _ in range(BATCH_SIZE)]
        for i in b_idx:
            s = state.make_sample(train_outfits[i], "train", NUM_NEGATIVES)
            negatives_per_sample.append(len(s["negatives"]))
            if len(s["negatives"]) < NUM_NEGATIVES:
                short_rows += 1
            for neg_id in s["negatives"]:
                total_negatives += 1
                if state.item_cat[neg_id] == s["target_cat"]:
                    matched_negatives += 1

    match_rate = matched_negatives / total_negatives if total_negatives else 0.0
    mean_negs = sum(negatives_per_sample) / len(negatives_per_sample)

    lines = [
        "# Phase 14b, Step 1: Negative Sampling Category-Match Rate, Before and After",
        "",
        f"Measured the same way as phase 14's own diagnostic "
        f"(`week4/phase14_outfittransformer_siglip/phase14_notes.md`): "
        f"{N_BATCHES} random {BATCH_SIZE}-outfit batches from the training split.",
        "",
        "| | Category-match rate | Source |",
        "|---|---|---|",
        f"| Before (phase 14, in-batch negatives, no category restriction) | {BEFORE_MATCH_RATE:.1%} "
        f"| `week4/phase14_outfittransformer_siglip/phase14_notes.md`, line 29 (cited, not recomputed -- "
        f"the unrestricted in-batch mechanism it measured no longer exists in this phase's training loop) |",
        f"| **After (phase 14b, same-category candidate pool)** | **{match_rate:.4f} ({match_rate:.1%})** "
        f"| computed fresh here, {total_negatives} total sampled negatives across "
        f"{len(negatives_per_sample)} samples |",
        "",
        f"Mean negatives actually returned per sample: {mean_negs:.2f} / {NUM_NEGATIVES} target. "
        f"Samples that fell short of {NUM_NEGATIVES} negatives (target category's own pool, "
        f"plus mined candidates, together held fewer than {NUM_NEGATIVES} eligible items): "
        f"{short_rows} / {len(negatives_per_sample)} ({short_rows/len(negatives_per_sample):.2%}).",
        "",
        "Every negative returned by `_sample_negatives` is drawn either from the mined "
        "same-category candidate list (`week4/phase13_csa_net_baseline/data/negative_candidates.json`) "
        "or, as a top-up, from the target's own category pool "
        "(`train_items_by_category[target_cat]`) directly -- both sources are restricted to "
        "`target_cat` by construction, so any match rate below 100% would indicate a bug in "
        "`item_cat` lookup consistency, not a sampling choice.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Category-match rate after fix: {match_rate:.4f} ({total_negatives} negatives checked)")
    print(f"Mean negatives per sample: {mean_negs:.2f}/{NUM_NEGATIVES}, short rows: {short_rows}")
    print(f"Saved {OUT_MD}")

    assert match_rate > 0.99, f"Category-match rate {match_rate:.4f} is not close to 100% -- stop and investigate before training."


if __name__ == "__main__":
    main()
