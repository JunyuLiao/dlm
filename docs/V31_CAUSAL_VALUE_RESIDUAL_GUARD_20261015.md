# Causal value-residual trust: definition, evidence, and limits

This is a refinement of directions 2 and 3 in the 2026-10-15 plan. The
previous-query extrapolator failed its real-state screen; unconditional cached
correction improved mean error but introduced harmful row tails. This guard
uses a cached value-error direction and the current selected output. It keeps
the attention map and per-query-tile K unchanged.

## Rule

All vectors below are for one query row after the full attention O projection,
with contributions from every head summed before computing norms. At an
observation, cache full output D0, selected output S0, and residual r=D0-S0.
At a held call, the current sparse output St is available. Apply

    alpha = clip(1 - <St-S0, r> / ||r||², 0, 1)
    corrected = St + alpha*r

For a zero residual use alpha=0. At an unchanged state alpha=1 reconstructs
the observed full output. If the selected output has already moved all the
way toward the old dense anchor along r, alpha becomes zero and avoids adding
the same correction twice. Orthogonal movement does not automatically erase
the useful cached direction. A norm-only shrinkage rule is a separate ablation.

The rule is the exact constrained minimizer of
`||St + alpha*r - D0||²`. It optimizes a causal historical surrogate. It does
not have access to the current omitted-key output or the current dense output.

## A precise approximation statement

Let Dt be the unavailable current dense output and deltaD=Dt-D0. The oracle
coefficient along the same residual is

    alpha_star = clip(1 - <(St-S0)-deltaD, r> / ||r||², 0, 1).

Projection onto [0,1] is nonexpansive, giving

    |alpha-alpha_star| <= |<deltaD,r>| / ||r||².

For the actual squared-error objective f(a)=||St+a*r-Dt||², the stronger
one-dimensional quadratic regret bound is

    0 <= f(alpha)-f(alpha_star) <= <deltaD,r>² / ||r||² <= ||deltaD||².

To see the middle inequality, write f(a) as a constant plus
`||r||²*(a-c)²`, where c is the unconstrained current optimum. The implemented
coefficient is the interval projection of c+e, with
`e=-<deltaD,r>/||r||²`. For an interval, moving the quadratic center by e
increases the constrained minimum's evaluated objective by at most e².
Checking the interior and the two clipped boundary cases proves the result.

Thus the relevant uncertainty is movement of the dense output *along the
cached value-error direction*. Large orthogonal drift can be harmless to the
coefficient. This is a relative-to-oracle statement, not a certificate that
correction beats no correction: the unknown dense movement can still make
the implemented coefficient worse than alpha=0. It is also not a guarantee
about logits, sampler acceptance, final accuracy, or denoising count.

## Observed same-state evidence

Source: `results/v31_value_decision_reuse_20261015/reuse_guard/`. All 16 heads,
both Q128 tiles and five GLOBAL layers; frozen Q/K/V; identical K; no model
generation in this replay. The following includes sparse held calls only.

| Previously examined trajectory | Guard mean error / V31 | Unguarded frames with >2% worse max row | Guard frames with >2% worse max row |
|---|---:|---:|---:|
| LongBench request001, 220 frames | 0.6444 | 26 | 0 |
| LongBench request002, 100 frames | 0.7753 | 25 | 3 |
| Late AIME request004, 30 frames | 0.8291 | 11 | 2 |

For layer29 specifically, the pooled LongBench mean-error ratio is 0.7526
across 64 sparse held frames, with no >2% maximum-row regression; AIME's six
layer29 frames give 0.6673 and no such regression. The layer29 model probe was
chosen for an isolated suffix intervention before these layer-specific
results were calculated. These small, previously examined trajectories are
exploratory and do not establish general safety or a native accuracy gain.

The norm-only guard removes the observed >2% maximum-row regressions but
reduces the mean gain to 17.2% / 3.0% / 1.8% on these three trajectories. Its
mean coefficient is only 0.076 / 0.014 / 0.017. Retain it as a conservative
ablation instead of interpreting suppression of nearly all correction as
successful reconstruction.

## Implementation and next gates

Reference: `experiments/numerical_qk_reuse/v31_value_reuse_guard.py`.
The independent native implementation is opt-in, final-GLOBAL-layer-only,
and rejects launch without a passed decision-gate receipt. It counts extra
O projections and observation work. It does not alter native stopping,
acceptance, noise, output limits, or K. All-kept states bypass reconstruction.
The current implementation explicitly computes a current O projection for
the guard; folding it into the existing projection is future engineering,
not a measured saving. Cached omitted-moment reconstruction has a separate
fused selected-output/LSE merge kernel, with full-V CUDA parity checked.

First qualify same-state logits/acceptance/stopping and observation-only
native invariance. Then evaluate official native accuracy and N/N-C/output
length/tails under fresh matched controls. Dense-output fidelity is a proxy;
V31's accuracy already exceeds dense on some prior panels, so fidelity alone
cannot select a winner. Clean timing remains paused. Cached residual and
compensated sparse attention have prior art; no first-of-its-kind claim is
made for residual caching alone.

## Exact adjoint implementation

Let pre-projection residual be r and output projection be W. At observation,
cache u = W^T W r and d = ||W r||^2, preserving signed contributions across
all heads. A held call computes alpha = clip(1 - <St-S0,u>/d,0,1) and returns
St + alpha r. This is algebraically the same whole-layer post-O coefficient.
It needs two projection GEMMs at the observation, then only a row dot product
and vector update per held call. It avoids a full W projection at every held
call; it does not avoid the model's normal output projection.

The reference cache is `ResidualDirectionCache`; one Triton kernel fuses the
held-step coefficient reduction and correction. Both the row reduction and
correction cover all concatenated heads. Cache identity guards canvas/support
changes. Memory stores three pre-O vectors per row (anchor selected, residual,
adjoint residual), plus one scalar; this trades extra cache memory for fewer
held-step arithmetic operations. CPU double-precision tests verify equality to
the explicit post-O calculation, including zero residual. H100 parity is
required before native integration. No clean timing claim is made.
