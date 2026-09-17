"""
Phase 27, step 3 (user-requested addition, not in the original brief): after
training, verify the learned category embedding table (text_category_learned
variant) actually contains meaningful, differentiated vectors rather than
having collapsed toward a single point.

Why this matters: if the 11 category vectors collapse to near-identical
directions (pairwise cosine -> ~1), the "category conditioning didn't help"
finding from 02_train_variants.py would be an ARCHITECTURAL FAILURE (the
signal never actually reached the network in a usable form), not a real
"category conditioning doesn't help retrieval" finding -- these are very
different conclusions and must not be conflated. This check exists to tell
them apart before phase27_notes.md draws any conclusion from the learned-
table variant's numbers.
"""
import json
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
CATEGORY_LIST = [
    "accessories", "all-body", "bags", "bottoms", "hats", "jewellery",
    "outerwear", "scarves", "shoes", "sunglasses", "tops",
]

CKPT_PATH = BASE_DIR / "models" / "text_category_learned_category_table.pt"
OUT_MD = BASE_DIR / "category_embedding_check.md"

COLLAPSE_THRESHOLD = 0.95  # mean off-diagonal cosine at/above this -> treat as collapsed


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"{CKPT_PATH} not found -- run 02_train_variants.py first "
            f"(text_category_learned must have save_checkpoint=True, which it does by default)."
        )

    state = torch.load(CKPT_PATH, map_location="cpu")
    # nn.Embedding's state_dict has a single "weight" tensor, (11, 768).
    weight = state["weight"].numpy().astype(np.float32)
    assert weight.shape == (len(CATEGORY_LIST), 768), f"unexpected shape {weight.shape}"

    norms = np.linalg.norm(weight, axis=1, keepdims=True)
    unit = weight / norms
    sims = unit @ unit.T  # (11, 11) cosine similarity matrix

    n = len(CATEGORY_LIST)
    off_diag = sims[~np.eye(n, dtype=bool)]
    mean_off_diag = float(off_diag.mean())
    max_off_diag = float(off_diag.max())
    min_off_diag = float(off_diag.min())

    # Most-confusable pair (highest cosine, i.e. least differentiated).
    tmp = sims.copy()
    np.fill_diagonal(tmp, -2.0)
    i, j = np.unravel_index(np.argmax(tmp), tmp.shape)
    most_similar_pair = (CATEGORY_LIST[i], CATEGORY_LIST[j], float(tmp[i, j]))
    tmp2 = sims.copy()
    np.fill_diagonal(tmp2, 2.0)
    i2, j2 = np.unravel_index(np.argmin(tmp2), tmp2.shape)
    least_similar_pair = (CATEGORY_LIST[i2], CATEGORY_LIST[j2], float(tmp2[i2, j2]))

    collapsed = mean_off_diag >= COLLAPSE_THRESHOLD

    print(f"Mean off-diagonal cosine: {mean_off_diag:.4f}")
    print(f"Range: [{min_off_diag:.4f}, {max_off_diag:.4f}]")
    print(f"Most similar pair: {most_similar_pair}")
    print(f"Least similar pair: {least_similar_pair}")
    print(f"Collapsed (>= {COLLAPSE_THRESHOLD})? {collapsed}")

    lines = [
        "# Category Embedding Collapse Check\n",
        "Requested addition: after training the `text_category_learned` variant, verify the "
        "learned 11-category embedding table actually contains meaningful, differentiated vectors "
        "rather than having collapsed toward a single direction. A collapsed table would mean "
        "category conditioning never actually engaged the network, which would make any "
        "\"category doesn't help\" finding an architectural failure, not a real result about "
        "whether category conditioning helps retrieval -- these must not be conflated.\n",
        f"**Verdict: {'COLLAPSED' if collapsed else 'NOT collapsed'}** "
        f"(mean off-diagonal pairwise cosine = {mean_off_diag:.4f}, threshold = {COLLAPSE_THRESHOLD}).\n",
        "## Pairwise cosine similarity matrix\n",
        "| | " + " | ".join(CATEGORY_LIST) + " |",
        "|---|" + "|".join(["---"] * n) + "|",
    ]
    for r, cat in enumerate(CATEGORY_LIST):
        row = " | ".join(f"{sims[r, c]:.3f}" for c in range(n))
        lines.append(f"| **{cat}** | {row} |")
    lines += [
        "",
        f"Mean off-diagonal cosine: **{mean_off_diag:.4f}**  (range [{min_off_diag:.4f}, {max_off_diag:.4f}])",
        f"Most-confusable pair (highest cosine): **{most_similar_pair[0]}** / **{most_similar_pair[1]}** "
        f"(cosine={most_similar_pair[2]:.4f})",
        f"Most-differentiated pair (lowest cosine): **{least_similar_pair[0]}** / **{least_similar_pair[1]}** "
        f"(cosine={least_similar_pair[2]:.4f})",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"Saved {OUT_MD}")


if __name__ == "__main__":
    main()
