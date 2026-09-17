"""Adapted mechanistic diagnostic for the dedicated-capacity architecture.
Phases 16/16c compared two fixed 128-d mode-vector PARAMETERS directly
(cosine similarity between them). There's no single mode vector here --
each mode's identity is the full relevance_head/tail_head transformation.
The direct equivalent: for the SAME sample of items, compute both heads'
own (independently normalized) OUTPUT embeddings and measure their mean
cosine similarity -- how similar the two heads' representations of the
same items are, the same question the mode-vector check was answering,
adapted to a full-head architecture. Also reports the two heads' weight
matrices' own cosine similarity (flattened) as a secondary, purely
structural comparison.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DedicatedCapacityHead

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT = BASE_DIR / "models" / "dedicated_capacity.pt"
GALLERY_NPZ = BASE_DIR.parent / "phase16_relevance_tail_dial" / "data" / "candidate_gallery.npz"
OUT_MD = BASE_DIR / "logs" / "head_similarity_diagnostic.md"

DEVICE = "cpu"

# phase 16's own mode-vector cosine (16c's dropped this to 0.059), for narrative continuity
PHASE16_MODE_VECTOR_COSINE = 0.5525
PHASE16C_MODE_VECTOR_COSINE = 0.0590


def main():
    m = DedicatedCapacityHead().to(DEVICE)
    m.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    m.eval()

    d = np.load(GALLERY_NPZ, allow_pickle=True)
    emb = d["embeddings"][:2000]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb = (emb / norms).astype("float32")
    x = torch.tensor(emb)

    with torch.no_grad():
        z_rel, z_tail = m.head_outputs(x)
        per_item_cosine = torch.nn.functional.cosine_similarity(z_rel, z_tail, dim=-1)
    mean_output_cosine = per_item_cosine.mean().item()
    std_output_cosine = per_item_cosine.std().item()

    # secondary, structural comparison: the two heads' weight matrices themselves
    w_rel = m.relevance_head.weight.detach().flatten()
    w_tail = m.tail_head.weight.detach().flatten()
    weight_cosine = torch.nn.functional.cosine_similarity(w_rel.unsqueeze(0), w_tail.unsqueeze(0)).item()
    w_rel_norm = m.relevance_head.weight.norm().item()
    w_tail_norm = m.tail_head.weight.norm().item()

    lines = ["# Phase 16d: Head Similarity Diagnostic\n"]
    lines.append("Adapted equivalent of phases 16/16c's mode-vector cosine similarity check -- there's no "
                 "single mode-vector parameter here, so this compares the two dedicated heads' own OUTPUT "
                 f"embeddings for the same {len(emb)} sample items directly.\n")
    lines.append(f"**Mean per-item cosine similarity between relevance_head's and tail_head's outputs, same "
                 f"items: {mean_output_cosine:.4f} (std {std_output_cosine:.4f})**\n")
    lines.append("## Comparison across the sequence\n")
    lines.append("| | Phase 16 (mode vectors) | Phase 16c (mode vectors) | Phase 16d (head outputs) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Cosine similarity | {PHASE16_MODE_VECTOR_COSINE:.4f} | {PHASE16C_MODE_VECTOR_COSINE:.4f} | "
                 f"{mean_output_cosine:.4f} |")
    lines.append("")
    lines.append("## Secondary, structural comparison: the two heads' weight matrices\n")
    lines.append(f"- cosine(relevance_head.weight, tail_head.weight), flattened: {weight_cosine:.4f}")
    lines.append(f"- relevance_head weight norm: {w_rel_norm:.4f}, tail_head weight norm: {w_tail_norm:.4f}\n")

    OUT_MD.write_text("\n".join(lines))
    print(f"Mean output cosine (relevance_head vs tail_head, same items): {mean_output_cosine:.4f}")
    print(f"Weight matrix cosine: {weight_cosine:.4f}")
    print(f"Written: {OUT_MD}")


if __name__ == "__main__":
    main()
