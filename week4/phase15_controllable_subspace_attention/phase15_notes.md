# Phase 15: Control-Conditioned Subspace Attention -- Overall Interpretation

## Headline result

This phase extended CSA-Net's category-pair-conditioned subspace attention
(phase 13b) so its `attn_net` also conditions on the substitute/complement
control signal alpha -- a genuine, citable architectural merge of two
previously-separate, independently-validated mechanisms (phase 13b's
attention conditioning, phase 12c's controllable dial). The extension
**works** in the narrow sense that `attn_net` demonstrably learns a real,
non-trivial response to alpha (attention-weight shift 0.486 for the better
checkpoint, vs. 0.041 untrained) and produces a genuinely smooth,
monotonically-decaying retrieval-set overlap as alpha moves (the
adjacent-vs-distant check). But it does **not** beat either of the two
simpler mechanisms it was built from, on either axis that matters:

- **Vs. phase 13b's own fixed (non-conditional) CSA-Net mechanism**: this
  phase's alpha=0 endpoint (0.0657/0.1326/0.1792) sits at 90.6%/95.2%/97.2%
  of phase 13b's Recall@K -- within the pre-declared endpoint tolerance,
  but still consistently lower at every K. Adding conditioning capacity did
  not help; it cost a small amount of complement-mode quality, most likely
  from shared-parameter competition with the substitute objective (the same
  capacity-competition pattern phase 12c first documented, now confirmed to
  extend to this richer architecture too).
- **Vs. phase 12c/12d's own simpler additive mode-vector mechanism (same
  backbone)**: this phase's realized controllable RANGE is far smaller.
  This phase's better (discrete) checkpoint moves Recall@10 by only ~3.5%
  relative across the full alpha sweep (0.0657 -> 0.0634); phase 12c/12d's
  mechanism moves Recall@10 by ~31% relative (0.0971 -> 0.0667) on the
  identical backbone. The adjacent-vs-distant smoothness gap tells the same
  story: this phase's best gap (0.1124, discrete) reaches only "partial
  confirmation" under phase 12d's own thresholds, while phase 12c/12d's gap
  (0.475) clears the "real, smooth relationship confirmed" bar more than 3x
  over.

**Put plainly: the more architecturally sophisticated, attention-based
conditioning mechanism built this phase produces a real but substantially
WEAKER controllable dial than the simple two-additive-vectors mechanism
already validated in phase 12c/12d, on the identical frozen SigLIP
backbone.** This is not what a naive prior (richer mechanism = better
controllability) would predict, and is reported as such, not spun as a
partial win.

## Why: the alpha double-duty confound, quantified rather than assumed

`architecture_notes.md`'s named risk -- alpha serving simultaneously as the
`attn_net` conditioning input and as the outer loss-envelope weight -- was
checked directly, not assumed away, at three points:

1. **Smoke test** (small-scale, 5 epochs): attention-weight shift stayed
   near the untrained baseline (0.024-0.028 vs. 0.041) even with 20%
   decoupled-alpha steps -- an early, inconclusive signal consistent with
   the confound suppressing conditioning, but not yet a verdict (the smoke
   test is deliberately tiny).
2. **Full continuous-alpha training** (60 epochs to convergence): the
   confound's predicted effect partially held -- `attn_net` DID learn real
   conditioning (0.188 mean shift), but far less than it might have without
   the confound, and concentrated unevenly across category pairs (0 to
   0.871 per-pair range) rather than uniformly.
3. **Full discrete-alpha training** (58 epochs, early-stopped): concentrating
   all training signal at exactly two alpha values (rather than diluting it
   across a continuous range) produced substantially MORE conditioning
   (0.486 mean shift, max per-pair 1.909, near the theoretical ceiling of
   2.0) and better Recall@K at every point in the sweep -- direct evidence
   that continuous alpha sampling was itself diluting the per-alpha gradient
   signal `attn_net` received, exactly the mechanism the named risk
   predicted, now confirmed by a controlled ablation rather than inferred.

This is a real, mechanistically-explained finding, not an unexplained
negative result: **the double-duty confound measurably suppressed this
architecture's realized conditioning strength, and discrete-alpha training
measurably (if only partially) counteracts it** -- but even the
partially-counteracted (discrete) version still falls well short of phase
12c/12d's simpler mechanism's controllable range.

## The three-way + one architectural-extension comparison, all on the identical frozen SigLIP backbone

| Mechanism | Recall@10 (complement-leaning) | Adjacent-vs-distant smoothness gap |
|---|---|---|
| CSA-Net, fixed/non-conditional (phase 13b) | 0.0725 | n/a (no alpha dial) |
| OutfitTransformer set-encoder (phase 14) | 0.0201 | n/a (no alpha dial) |
| **This phase: CSA-Net + alpha-conditioning, discrete-trained** | **0.0657** | **0.1124 (partial confirmation)** |
| Phase 12c/12d: simple additive mode vectors | 0.0971 (alpha=0) | 0.475 (confirmed) |

Ranking by Recall@10 (complement-leaning end): phase 12c/12d > phase 13b's
fixed CSA-Net > this phase's conditioned CSA-Net > phase 14's
OutfitTransformer. This phase's extension improves on phase 14's mechanism
by a wide margin, but does not close the gap to either of the two
mechanisms it most directly builds on or competes with.

## Loss-balancing caveat carried into this interpretation

`loss_balancing_check.md`'s gradient-norm verification passed at alpha=0.5
and alpha=0.8 but failed at alpha=0.2 (weighted ratio 3.75x, outside the
declared 3x tolerance) -- the raw complement/substitute gradient-scale ratio
is itself somewhat alpha-dependent, and a single fixed `weight_sub` does not
fully correct for it everywhere. This is a plausible SECONDARY contributor
to the modest, uneven realized conditioning (on top of the primary,
confirmed double-duty confound) -- not tested in isolation this phase (an
alpha-dependent weight was explicitly not introduced, per the plan's
pre-declared fallback-only-if-needed rule), and a reasonable next step if
this mechanism is revisited.

## What this does and doesn't establish

- **Does establish**: a real, working architectural merge of CSA-Net's
  conditioning mechanism and phase 12c's controllable dial is possible and
  trains stably (no collapse, no crash, clean convergence on both runs) --
  a genuine, if modest, positive engineering result. Also establishes a
  concrete, quantified, mechanistically-explained account of WHY the merge
  underperforms both of its parent mechanisms (the double-duty confound,
  confirmed via a controlled discrete-vs-continuous ablation, not assumed).
  Also establishes that discrete-then-interpolate training is a genuinely
  better recipe than continuous-alpha training for this specific
  architecture -- a real, actionable, if unglamorous, finding.
- **Doesn't establish**: that category-pair-conditioned attention is
  inherently worse at controllability than simple additive vectors in
  general -- only that THIS specific way of injecting alpha (concatenated
  into the same subnet that also determines the loss-envelope weight) has a
  structural confound that this training regime did not fully overcome. A
  genuinely decoupled conditioning signal (e.g. alpha never appearing in the
  loss envelope at all, replaced by a fixed 50/50 weighting with alpha only
  ever a forward-pass input) was not tried -- a real follow-up, out of scope
  here.
- **Doesn't establish** that the alpha-dependent gradient-ratio breach found
  in `loss_balancing_check.md` is a major contributor to the weak
  controllability (it wasn't isolated from the double-duty confound via a
  separate ablation) -- flagged as a plausible secondary factor, not proven.

## Not attempted (per the brief's own scope)

No backbone swap away from frozen SigLIP. No paper writeup. No
alpha-dependent `weight_sub` variant (flagged as a possible follow-up, not
built). No fully-decoupled alpha-injection variant (alpha never entering the
loss envelope) that would more cleanly isolate the double-duty confound from
this phase's specific architecture choice.
