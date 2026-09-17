# Phase 21, Step 4: Complement Head Endpoint Verification -- The Central Check

Phase 21's complement head, trained with a fully independent set of parameters (zero shared layer, zero shared capacity with the substitute head), evaluated ALONE (no blending) on the identical CIR benchmark used throughout this project, compared directly against phase 9's own confirmed Recall@10/30/50.

| | Recall@10 | Recall@30 | Recall@50 |
|---|---|---|---|
| Phase 9 (confirmed, phase 20) | 0.1317 | 0.2464 | 0.3216 |
| Phase 21 complement head (this phase) | 0.1317 | 0.2464 | 0.3216 |
| Delta | -0.0000 | -0.0000 | -0.0000 |

n_total=29681, n_skipped=0.

**Verdict: EXACT MATCH (within floating-point noise, <0.0001 at every K).**

This is the central check the entire zero-shared-capacity premise rests on: if there is truly no other objective touching these parameters at any point during training, this endpoint should reproduce phase 9's own number exactly or within ordinary run-to-run training noise (this project has repeatedly observed such noise even for identical setups, e.g. MPS's non-deterministic parallel float reductions across separate runs). A larger gap than that would mean something in the setup diverged from phase 9's own -- an accidental difference in data loading, hyperparameters, or the RNG seeding sequence -- and would need direct investigation before trusting this phase's design at all, per the brief's own explicit instruction.
