# Phase 18, Step 3: Architecture -- Four Dedicated Heads, Bilinear Interpolation

## Design

Direct extension of phase 16d/17's dedicated-capacity pattern (twice independently confirmed to fix
capacity competition in a controllable dual-mode mechanism) from 2 heads to 4 -- one per corner of the
substitute/complement x relevance/tail-exposure grid:

- Shared: `Linear(768,256) -> ReLU -> Dropout(0.1)` (dimensionality reduction only, same as 16d/17 --
  not a shared transformation).
- Four fully independent `Linear(256,128)` heads: `sub_rel_head`, `sub_tail_head`, `comp_rel_head`,
  `comp_tail_head`, each L2-normalized independently before blending.

## Blending: bilinear interpolation across two independent alpha signals

`alpha1` (substitute=1.0 / complement=0.0), `alpha2` (relevance=1.0 / tail-exposure=0.0), matching this
project's existing per-axis alpha conventions (phase 12/12c/17 for alpha1, phase 16/16d for alpha2)
exactly, so a 1D slice through this grid at a fixed alpha2 endpoint reduces to the familiar single-axis
dial.

```
w_sr = alpha1 * alpha2            (substitute-relevance)
w_st = alpha1 * (1 - alpha2)      (substitute-tail-exposure)
w_cr = (1 - alpha1) * alpha2      (complement-relevance)
w_ct = (1 - alpha1) * (1 - alpha2)  (complement-tail-exposure)

blend = w_sr*z_sr + w_st*z_st + w_cr*z_cr + w_ct*z_ct
output = normalize(blend)
```

## Verification (direct Python test, `model.py`'s docstring claims checked, not assumed)

**Endpoint identity**: at each of the 4 grid corners (alpha1, alpha2 in {0,1}^2), the bilinear blend
collapses to exactly one weight of 1.0 and the other three at 0.0 -- confirmed the blended output
matches that corner's own `head_outputs()` entry to float32 precision (max abs diff <= 2.98e-08 across
all 4 corners, `model.eval()`, dropout off).

**Gradient isolation**: at corner `sub_rel` (alpha1=1.0, alpha2=1.0), backpropagating a loss on the
model's output produces exactly zero gradient into `sub_tail_head`, `comp_rel_head`, and
`comp_tail_head`'s parameters (grad abs sum = 0.000000 for all three), while `sub_rel_head` and the
shared layer both receive real, nonzero gradient. This is the 2D generalization of phase 16d/17's own
verified endpoint-gradient-isolation property -- confirms training step 4's design (each corner's loss,
computed via `forward_at_corner`, only ever updates its own head plus the shared trunk) is architecturally
exact, not approximate.

## Why bilinear, and why this collapses correctly

Bilinear interpolation is the natural generalization of the project's existing linear single-axis blend
(`alpha*z_A + (1-alpha)*z_B`) to two independent axes: it is linear in `alpha1` for any fixed `alpha2`
and linear in `alpha2` for any fixed `alpha1`, and the four corner weights always sum to 1 (a proper
convex combination) for any `(alpha1, alpha2)` in `[0,1]^2`. At intermediate values, all four heads
contribute proportionally to their nearness (in each axis) to that corner -- e.g. `(alpha1=0.5,
alpha2=0.5)` weights all four heads equally at 0.25 each.
