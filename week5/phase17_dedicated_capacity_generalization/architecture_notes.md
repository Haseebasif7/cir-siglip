# Phase 17: Architecture Notes

## Confirmation this matches phase 16d's design, retrofit onto phase 12c's mechanism

`scripts/model.py`'s `DedicatedCapacityHead` is structurally identical to
`week5/phase16d_dedicated_capacity/scripts/model.py`'s `DedicatedCapacityHead`:

```
shared: Linear(768, 256) -> ReLU -> Dropout(0.1)
substitute_head: Linear(256, 128)   [phase 16d: relevance_head]
complement_head: Linear(256, 128)   [phase 16d: tail_head]
```

Both heads are independently L2-normalized, then blended by `alpha` and
renormalized -- byte-for-byte the same `forward()`/`head_outputs()` logic as
phase 16d, only the two head attributes are renamed to match this
mechanism's own vocabulary (`substitute_head`/`complement_head` instead of
`relevance_head`/`tail_head`). Convention carried over from phase 12/12b/12c
unchanged: `alpha=1.0` = pure substitute mode, `alpha=0.0` = pure complement
mode.

The single architectural difference from phase 12/12b/12c's
`ControllableProjectionHead` (`week4/phase12c_ranking_distillation/scripts/model.py`)
is exactly the one under test: that mechanism uses one fully shared
`768->256->128` trunk (`self.net`) plus two small learnable 128-d additive
correction vectors (`mode_substitute`, `mode_complement`), a design with far
less independent capacity per mode than this phase's minimal-shared-layer
architecture. Loss formulations, data, hyperparameters, and training
procedure are otherwise unchanged from phase 12c (see `loss_balancing_check.md`
and `02_train.py`) -- architecture is the only variable under test, per the
brief.

## Verification checks (before committing to training)

Run directly against a freshly-initialized model (see session transcript,
also reproducible via `python3 -c` using `scripts/model.py`):

- **Endpoint identity**: `model(x, alpha=1.0)` exactly matches
  `head_outputs(x)[0]` (the substitute head's own output), and
  `model(x, alpha=0.0)` exactly matches `head_outputs(x)[1]` (complement),
  confirmed with `model.eval()` (dropout disabled) -- exact match, as
  expected for a pure linear blend at the endpoints.
- **Gradient isolation**: at `alpha=1.0` exactly, `complement_head`'s
  parameters receive **zero** gradient (confirmed: gradient sum = 0.0),
  since its output is multiplied by `(1-alpha)=0` in the blend; `substitute_head`
  receives a large nonzero gradient. This confirms the two heads are
  genuinely independent capacity, not just nominally separate parameters that
  still get entangled through the blend's backward pass.

Both checks passed cleanly, matching phase 16d's own verification pattern.
