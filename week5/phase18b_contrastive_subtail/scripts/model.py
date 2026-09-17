"""Phase 18b: UNCHANGED copy of phase 18's model.py, byte-for-byte -- per
this phase's own explicit instruction, no architecture change (widening the
shared layer is a separate, untested hypothesis, not conflated with this
phase's training-method fix). Only `sub_tail`'s training SIGNAL and the
four-way loss weighting change this phase (see `train_core.py`).

Original docstring: the same minimal-shared-trunk, dedicated-capacity
pattern phases 16d and 17 both independently confirmed improves a
controllable dual-mode mechanism, extended here from 2 heads to 4 -- one
per corner of the substitute/complement x relevance/tail-exposure grid.

Shared: Linear(768,256)->ReLU->Dropout (dimensionality reduction only, same
as 16d/17). Four fully independent Linear(256,128) heads from there:
`sub_rel_head`, `sub_tail_head`, `comp_rel_head`, `comp_tail_head`.

Blending is BILINEAR across two independent alpha signals (alpha1:
substitute/complement, 1.0=pure substitute, 0.0=pure complement; alpha2:
relevance/tail-exposure, 1.0=pure relevance, 0.0=pure tail-exposure),
matching phases 12/12c/16d/17's alpha convention on each axis individually:

    w_sr = alpha1 * alpha2            (substitute-relevance)
    w_st = alpha1 * (1 - alpha2)      (substitute-tail-exposure)
    w_cr = (1 - alpha1) * alpha2      (complement-relevance)
    w_ct = (1 - alpha1) * (1 - alpha2)  (complement-tail-exposure)

At any of the 4 grid corners (alpha1, alpha2 in {0,1}), exactly one weight
equals 1 and the other three equal 0 -- the blend reduces to exactly that
corner's own head output (already L2-normalized), and the final
renormalization is then a no-op. This is the 2D generalization of phase
16d/17's verified "endpoint identity" property, and it's what makes step
4's training (each corner head trained only on its own fixed-extreme data,
never on an interpolated blend -- per phase 15's finding that continuous
training dilutes per-corner signal) produce exact per-corner gradient
isolation: at a corner, the three inactive heads receive exactly zero
gradient, since their contribution to `blend` is multiplied by a weight of
0.0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CORNERS = ("sub_rel", "sub_tail", "comp_rel", "comp_tail")
# (alpha1, alpha2) for each corner, matching the axis conventions above
CORNER_ALPHAS = {
    "sub_rel": (1.0, 1.0),
    "sub_tail": (1.0, 0.0),
    "comp_rel": (0.0, 1.0),
    "comp_tail": (0.0, 0.0),
}


class FourHeadDedicatedCapacity(nn.Module):
    def __init__(self, in_dim=768, shared_dim=256, out_dim=128, dropout=0.1):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, shared_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sub_rel_head = nn.Linear(shared_dim, out_dim)
        self.sub_tail_head = nn.Linear(shared_dim, out_dim)
        self.comp_rel_head = nn.Linear(shared_dim, out_dim)
        self.comp_tail_head = nn.Linear(shared_dim, out_dim)

    def head_outputs(self, x):
        """All four heads' own normalized outputs, unblended -- (dict of
        corner name -> (B, out_dim) tensor). Used for training (isolate a
        single corner's loss) and for the head-similarity / corner-sanity
        diagnostics."""
        h = self.shared(x)
        return {
            "sub_rel": F.normalize(self.sub_rel_head(h), p=2, dim=-1),
            "sub_tail": F.normalize(self.sub_tail_head(h), p=2, dim=-1),
            "comp_rel": F.normalize(self.comp_rel_head(h), p=2, dim=-1),
            "comp_tail": F.normalize(self.comp_tail_head(h), p=2, dim=-1),
        }

    def forward(self, x, alpha1, alpha2):
        """alpha1: substitute(1.0)/complement(0.0). alpha2: relevance(1.0)/
        tail-exposure(0.0). Both may be python floats or (B,) tensors for
        per-example mixing. Bilinear blend of all four heads, renormalized."""
        outs = self.head_outputs(x)
        if isinstance(alpha1, torch.Tensor) or isinstance(alpha2, torch.Tensor):
            a1 = alpha1 if isinstance(alpha1, torch.Tensor) else torch.full((x.shape[0],), alpha1, device=x.device)
            a2 = alpha2 if isinstance(alpha2, torch.Tensor) else torch.full((x.shape[0],), alpha2, device=x.device)
            a1 = a1.unsqueeze(-1)
            a2 = a2.unsqueeze(-1)
        else:
            a1, a2 = alpha1, alpha2
        w_sr = a1 * a2
        w_st = a1 * (1 - a2)
        w_cr = (1 - a1) * a2
        w_ct = (1 - a1) * (1 - a2)
        blend = w_sr * outs["sub_rel"] + w_st * outs["sub_tail"] + w_cr * outs["comp_rel"] + w_ct * outs["comp_tail"]
        return F.normalize(blend, p=2, dim=-1)

    def forward_at_corner(self, x, corner):
        """Convenience: forward at one of the 4 exact grid corners by name.
        Equivalent to `forward(x, *CORNER_ALPHAS[corner])`, kept as a
        separate entry point for training-loop readability."""
        alpha1, alpha2 = CORNER_ALPHAS[corner]
        return self.forward(x, alpha1, alpha2)

    def single_head_output(self, x, corner):
        """Compute only ONE head's normalized output (skip the other 3) --
        numerically identical to `forward_at_corner(x, corner)` (per the
        endpoint-identity property), used during training so each corner's
        loss doesn't pay for the other three heads' forward pass every step."""
        h = self.shared(x)
        head = getattr(self, f"{corner}_head")
        return F.normalize(head(h), p=2, dim=-1)


def mnrl_loss(z_a, z_p, z_extra, inbatch_false_neg_mask, tau=0.07):
    """Unchanged from phases 9/12c/16/16c/16d/17 -- used for both
    complement-mode corners (comp_rel, comp_tail)."""
    B = z_a.shape[0]
    logits_inbatch = (z_a @ z_p.T) / tau
    logits_inbatch = logits_inbatch.masked_fill(inbatch_false_neg_mask, float("-inf"))
    logits_extra = torch.einsum("bd,bkd->bk", z_a, z_extra) / tau
    logits = torch.cat([logits_inbatch, logits_extra], dim=1)
    labels = torch.arange(B, device=z_a.device)
    return F.cross_entropy(logits, labels)


def kl_ranking_distillation_loss(z_anchor, z_neighbors, teacher_sims, tau=0.07):
    """Unchanged from phase 12c/17 -- used for both substitute-mode corners
    (sub_rel, sub_tail). teacher_sims: (B, K) raw-SigLIP cosine similarity
    to each anchor's precomputed top-K neighbors (the ranking-distillation
    teacher, fixed, not model-dependent)."""
    teacher_dist = F.softmax(teacher_sims / tau, dim=-1)
    student_sims = torch.einsum("bd,bkd->bk", z_anchor, z_neighbors)
    student_log_probs = F.log_softmax(student_sims / tau, dim=-1)
    return F.kl_div(student_log_probs, teacher_dist, reduction="batchmean")


def mean_pairwise_cosine(z):
    """Collapse-guard diagnostic, unchanged."""
    sims = z @ z.T
    n = sims.shape[0]
    off_diag_sum = sims.sum() - torch.diagonal(sims).sum()
    return (off_diag_sum / (n * (n - 1))).item()
