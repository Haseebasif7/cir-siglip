# Phase 21: Architecture Confirmation -- Fully Independent, Zero Shared Capacity

## What every prior dial in this project had, and what this one deliberately removes

Every controllable-mode mechanism this project has built, from phase 12's original shared-trunk-plus-additive-mode-vector design through phase 17/18/18b's dedicated-capacity architecture, kept at least one layer in common between the two modes:

- Phase 12/12b/12c/12d: `Linear(768,256) -> ReLU -> Dropout -> Linear(256,128)`, one full shared trunk, with a small learned additive vector per mode blended in before the final L2-normalization.
- Phase 16d/17/18/18b (the "dedicated capacity" architecture): a smaller shared `Linear(768,256)` dimensionality-reduction step, but still shared, followed by fully independent per-mode `Linear(256,128)` heads.

Phase 20 found phase 17's own complement mode -- trained on the exact same objective phase 9 uses, on this dedicated-capacity architecture, the best dial design this project has built -- reached `Recall@10=0.1202`, below phase 9's standalone `0.1317`. Even with only one small shared layer left, and even with a loss-balancing weight specifically calibrated to protect it, the complement side still paid a real, measurable cost. This is the clue this phase's design directly targets.

## This phase's design: two complete networks, zero shared layers, zero shared parameters

`scripts/model.py` defines a single `ProjectionHead` class (`768 -> 256 -> 128`, identical shape to phase 9's own architecture, L2-normalized output) -- and this phase instantiates it TWICE, as two entirely separate `nn.Module` objects:

- `models/complement_head.pt`: trained by `01_train_complement_head.py`, an exact replica of phase 9's own `05_train_projection.py` (Model A specifically: R=8 random negatives, H=0 hard negatives, seed=42, identical data, identical hyperparameters, identical training-loop code).
- `models/substitute_head.pt`: trained by `02_train_substitute_head.py`, using phase 12c's proven ranking-distillation objective (KL divergence between each anchor's raw-SigLIP-neighbor teacher distribution and this head's own student distribution over the same neighbors) as its SOLE loss -- no complement loss, no shared trunk to balance a weight against, so `weight_sub` is not even a concept here; the objective simply gets the network's full, undivided capacity.

**Confirmed directly, not just by construction**: at no point does any code path let a gradient from one head's loss touch the other head's parameters. Each `torch.optim.Adam` optimizer in `01_train_complement_head.py` / `02_train_substitute_head.py` is constructed from `model.parameters()` of its own, separately-instantiated `ProjectionHead` -- there is no shared object, no shared `nn.Module` reference, no module passed between the two training scripts at all. The two checkpoints are produced by two fully separate Python processes (or sequential runs), each with its own model instance created from scratch (`ProjectionHead().to(DEVICE)`), with `torch.manual_seed(SEED)` called immediately before each model's own instantiation -- matching phase 9's own script's exact call order, so weight initialization is reproduced under the identical seeding sequence phase 9 used.

## Where the two heads' outputs meet for the first time: inference, not training

The only place this phase's two networks ever interact is in `04_alpha_sweep.py`, at inference time, on their already-trained, frozen outputs:

```
z_blend = normalize((1 - alpha) * z_complement + alpha * z_substitute)
```

This is a pure post-hoc combination of two independently-computed 128-d vectors -- there is no learned blending mechanism, no additional training step for the blend itself. At `alpha=0.0` this is exactly `z_complement` (already unit-norm, so normalizing it again is a no-op); at `alpha=1.0` this is exactly `z_substitute` -- the same endpoint-identity property every prior dial in this project has had, here holding by simple algebra rather than by a zero-initialized additive correction.

This is precisely the design the brief calls for: if there is truly zero shared capacity, the complement head's own number should be untouched by anything happening to the substitute head -- confirmed directly in `complement_endpoint_verification.md`, not assumed from the architecture description alone.
