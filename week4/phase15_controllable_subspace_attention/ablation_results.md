# Phase 15, Step 7: Continuous-Alpha vs. Discrete-Alpha Training Ablation

## Pre-committed interpretation framework (written before either run, per the plan)

**Recall axis**: if discrete-trained matches or beats continuous-trained on
Recall@K across most of the sweep, that's evidence the richer
attention-conditioning doesn't need dense alpha exposure during training to
work at inference -- a genuine simplification finding, reported as such even
though it undercuts the more elaborate training recipe. If continuous
clearly wins, that's evidence the richer mechanism needs the denser training
signal to actually use its extra capacity.

**Smoothness axis**: if discrete-trained's alpha-sweep still passes phase
12d's adjacent-vs-distant overlap smoothness check about as well as
continuous-trained, that specifically means the richer attention mechanism
is not buying any continuous controllability that the simple phase 12c
mechanism (already shown in phase 12d to interpolate smoothly) didn't
already have -- stated plainly as a limiting finding, not folded into a
generic "training was cheaper" framing. If discrete-trained's sweep is
visibly rougher than continuous-trained's, that's the scenario that would
justify continuous training as necessary.

## Result

### Recall axis: discrete WINS at every alpha, at every K

| alpha | Continuous R@10 | Discrete R@10 | Continuous R@30 | Discrete R@30 | Continuous R@50 | Discrete R@50 |
|---|---|---|---|---|---|---|
| 0.0 | 0.0611 | **0.0657** | 0.1251 | **0.1326** | 0.1720 | **0.1792** |
| 0.2 | 0.0608 | **0.0655** | 0.1251 | **0.1319** | 0.1712 | **0.1791** |
| 0.4 | 0.0604 | **0.0652** | 0.1251 | **0.1315** | 0.1708 | **0.1784** |
| 0.5 | 0.0601 | **0.0651** | 0.1249 | **0.1311** | 0.1707 | **0.1781** |
| 0.6 | 0.0602 | **0.0647** | 0.1244 | **0.1310** | 0.1705 | **0.1778** |
| 0.8 | 0.0598 | **0.0642** | 0.1243 | **0.1299** | 0.1695 | **0.1762** |
| 1.0 | 0.0596 | **0.0634** | 0.1236 | **0.1294** | 0.1693 | **0.1754** |

(Full 11-point sweep in `data/cir_sweep_continuous.json` / `data/cir_sweep_discrete.json`.)

**Discrete training beats continuous training at every single alpha value
sampled, at every K.** This is not a close or mixed result -- the margin is
consistent (roughly 5-8% relative at R@10, growing slightly at higher K)
across the entire sweep. Per the pre-committed framework: **this is evidence
the richer attention-conditioning mechanism does not need dense
alpha exposure during training to work well at inference** -- the cheaper
discrete-then-interpolate recipe is not just adequate, it is measurably
better here.

### Smoothness axis: discrete is ALSO smoother, not just comparably smooth

| Checkpoint | Adjacent overlap (delta=0.1) | Distant overlap (delta=1.0) | Gap | Verdict |
|---|---|---|---|---|
| Continuous | 0.9930 | 0.9422 | 0.0508 | Partial confirmation |
| Discrete | 0.9846 | 0.8722 | 0.1124 | Partial confirmation |

Both land in phase 12d's "partial confirmation" band (0.05 < gap < 0.15),
neither reaches phase 12c/12d's own "real, smooth relationship confirmed"
threshold (gap > 0.15) that the simpler additive mechanism cleared by more
than 3x (gap 0.475, see `week4/phase12d_alpha_sweep/results_table.md`). But
**discrete's gap (0.1124) is more than double continuous's (0.0508)** --
discrete training did not just fail to hurt smoothness, it produced a
noticeably stronger, more real alpha-dependence in the retrieval sets.

This is corroborated directly by the attention-weight-shift probe
(`architecture_notes.md`'s named-risk section): mean L1 shift (alpha=0 vs.
alpha=1, all 121 category pairs) is **0.486 for the discrete checkpoint vs.
0.188 for the continuous checkpoint** -- discrete's `attn_net` learned more
than 2.5x the conditioning response, with its maximum per-pair shift (1.909)
sitting near the theoretical ceiling of 2.0 for a 5-way softmax pair,
compared to continuous's maximum of 0.871.

## Interpretation (stated plainly, per the plan's explicit instruction not to spin this as a convenience win)

Per the pre-committed framework's own terms, this is the **"discrete matches
or beats continuous"** scenario on the Recall axis, and importantly NOT the
"discrete visibly rougher" scenario on the smoothness axis that would have
justified continuous training's added complexity -- if anything the reverse
happened. **The continuous-alpha training recipe conferred no benefit for
this architecture, and measurably underperformed the simpler discrete-then-
interpolate approach on every metric checked.**

A plausible mechanism, consistent with `architecture_notes.md`'s named
alpha double-duty confound: continuous training samples a fresh, different
alpha almost every step (`alpha ~ Uniform(0,1)`), so `attn_net` rarely sees
the SAME alpha value repeated across consecutive steps, diluting the
per-alpha gradient signal it receives at any specific point in the [0,1]
range. Discrete training instead concentrates all training signal at exactly
two alpha values (0 and 1), letting `attn_net` learn a sharper, more
decisive response at those two points -- which then interpolates
surprisingly well at intermediate alphas it never saw during training
(matching phase 12d's own finding for the simpler mechanism, that
discrete-trained endpoints interpolate smoothly without needing dense
training exposure). This was not tested further (e.g. a middle-ground
sampling schedule concentrated near the endpoints) -- out of scope for this
phase, a reasonable follow-up if this mechanism is revisited.

**Both checkpoints share the same underlying limitation documented in
`architecture_notes.md`: even the better (discrete) checkpoint's aggregate
Recall@K sweep only moves modestly across the full alpha range (R@10:
0.0657 -> 0.0634, a ~3.5% relative swing) compared to phase 12c/12d's
simple additive mechanism on the same backbone (R@10: 0.0971 -> 0.0667, a
~31% relative swing).** Discrete training measurably improved this
architectural extension's realized conditioning strength, but did not close
the gap to the simpler mechanism's own, much larger controllable range.

The discrete checkpoint is used as this phase's best configuration for
`results_table.md`, `qualitative_examples/`, and the final verdict in
`phase15_notes.md`.
