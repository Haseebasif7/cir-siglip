"""
Phase 17: adapted mechanistic diagnostic, phase 16d's own head-similarity
check (`week5/phase16d_dedicated_capacity/scripts/06_head_similarity_diagnostic.py`),
retrofit onto the substitute/complement heads. For a sample of items, compute
both heads' own (independently normalized) OUTPUT embeddings and measure
their mean cosine similarity -- the adapted equivalent of phase 12/12b/12c's
per-item cosine(z_sub, z_comp) check on the shared-trunk architecture (their
own diagnostic used the SAME blended-output notion but on a fully shared
projection; here the two heads are the "identity" being compared, matching
phase 16d's own reframing).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DedicatedCapacityHead

BASE_DIR = Path(__file__).resolve().parent.parent
PHASE9_DIR = BASE_DIR.parent.parent / "week3" / "phase9_polyvore_compatibility"
EMBEDDINGS_NPZ = PHASE9_DIR / "embeddings" / "siglip_base.npz"
CHECKPOINT = BASE_DIR / "models" / "dedicated_capacity_substitute_complement.pt"
OUT_MD = BASE_DIR / "logs" / "head_similarity_diagnostic.md"

DEVICE = "cpu"

# Phase 12/12b/12c's per-item cosine(z_sub, z_comp) on their shared-trunk architecture (5,000 items)
PHASE12_PER_ITEM_COS = 0.8288
PHASE12B_PER_ITEM_COS = 0.5844
PHASE12C_PER_ITEM_COS = 0.3000
# Phase 16d's own head-output cosine on the relevance/tail axis, same architecture family
PHASE16D_HEAD_OUTPUT_COS = -0.0147


def main():
    m = DedicatedCapacityHead().to(DEVICE)
    m.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    m.eval()

    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    embeddings = data["embeddings"][:5000]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = (embeddings / norms).astype("float32")
    x = torch.tensor(embeddings)

    with torch.no_grad():
        z_sub, z_comp = m.head_outputs(x)
        per_item_cosine = torch.nn.functional.cosine_similarity(z_sub, z_comp, dim=-1)
    mean_output_cosine = per_item_cosine.mean().item()
    std_output_cosine = per_item_cosine.std().item()

    w_sub = m.substitute_head.weight.detach().flatten()
    w_comp = m.complement_head.weight.detach().flatten()
    weight_cosine = torch.nn.functional.cosine_similarity(w_sub.unsqueeze(0), w_comp.unsqueeze(0)).item()

    lines = ["# Phase 17: Head Similarity Diagnostic\n"]
    lines.append("Adapted equivalent of phases 12/12b/12c's per-item cosine(z_sub, z_comp) check on "
                 "their shared-trunk architecture -- here there is no single shared projection output, "
                 f"so this compares the two dedicated heads' own OUTPUT embeddings for the same "
                 f"{len(embeddings)} sample items directly (same method phase 16d used on the "
                 "relevance/tail axis).\n")
    lines.append(f"**Mean per-item cosine similarity between substitute_head's and complement_head's "
                 f"outputs, same items: {mean_output_cosine:.4f} (std {std_output_cosine:.4f})**\n")
    lines.append("## Comparison across the sequence\n")
    lines.append("| | Phase 12 | Phase 12b | Phase 12c | Phase 16d (relevance/tail heads) | Phase 17 (substitute/complement heads) |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| Cosine similarity | {PHASE12_PER_ITEM_COS:.4f} | {PHASE12B_PER_ITEM_COS:.4f} | "
                 f"{PHASE12C_PER_ITEM_COS:.4f} | {PHASE16D_HEAD_OUTPUT_COS:.4f} | **{mean_output_cosine:.4f}** |")
    lines.append("")
    lines.append("(Phases 12/12b/12c measure cosine between the shared-trunk-derived substitute and "
                 "complement PROJECTIONS of the same items; phase 16d and this phase measure cosine "
                 "between two structurally independent HEADS' outputs -- both are the same underlying "
                 "question (how differentiated are the two modes' representations of the same item), "
                 "adapted to whichever architecture is being diagnosed.)\n")
    lines.append("## Secondary, structural comparison: the two heads' weight matrices\n")
    lines.append(f"- cosine(substitute_head.weight, complement_head.weight), flattened: {weight_cosine:.4f}\n")

    OUT_MD.write_text("\n".join(lines))
    print(f"Mean output cosine (substitute_head vs complement_head, same items): {mean_output_cosine:.4f}")
    print(f"Weight matrix cosine: {weight_cosine:.4f}")
    print(f"Written: {OUT_MD}")


if __name__ == "__main__":
    main()
