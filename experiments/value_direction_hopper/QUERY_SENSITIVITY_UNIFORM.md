# Uniform query-sensitivity calibration

## Problem

The historical temporal coefficient is

\[
 s_i^t = 1 + \beta z_i^t,
\]

where `z` is an EMA of top-1 flips. The first call has no previous logits and
therefore has `s=1`; the second call has an empty flip history and is also close
to one. A single threshold pair consequently routes the first calls sparsely,
even though denoising errors made there can propagate through the whole
trajectory. The guardrail runner addresses this with separate early and late
thresholds.

## Proposal: a causal uncertainty completion

`T_prior` keeps the value-direction-aware temporal signal, but completes its
missing early history with the sampler signal that is available after each
completed call. For query position `i`, let `r_i^{t-1}` be one when the previous
sampler call renoised the position and zero when it accepted it. Maintain

\[
 q_i^1=1,\qquad
 q_i^t=\gamma q_i^{t-1}+(1-\gamma)r_i^{t-1}.
\]

Let `z_i^t` be the existing flip EMA. The completed hazard and coefficient are

\[
 h_i^t=q_i^t+(1-q_i^t)z_i^t,\qquad
 s_i^t=\operatorname{clip}(1+\beta h_i^t,1,1+\beta).
\]

This has three useful boundary cases:

* At the first call, `q=1`, so every query receives the maximum protection
  `1+beta` without looking at current logits or current routing.
* A query that was repeatedly renoised remains high-sensitivity even if its
  argmax has not flipped yet.
* A query that is repeatedly accepted and has no flips decays toward one, so
  stable positions become cheaper under the same threshold pair.

`State.observe_logits` updates `q` only after the attention/sampler call. The
next `State.begin` consumes it, so the coefficient is causal. The old `T`
method remains unchanged for comparison.

## Uniform threshold adapter

`UniformThresholdState` in
[`query_sensitivity_uniform.py`](query_sensitivity_uniform.py) installs a
single `{local, global}` log-threshold pair on every call. It does not have an
early/late branch and does not disable first-call sensitivity. A runner can use
it as follows:

```python
from experiments.value_direction_hopper.query_sensitivity_uniform import (
    UniformThresholdState, uniform_policy,
)

state = UniformThresholdState(
    'T_prior', router, m_ref=cfg['m_ref'], beta=cfg['beta'], gamma=cfg['gamma'],
    thresholds=uniform_policy(local_log_threshold, global_log_threshold),
    seed=row['seed'], diagnostics=True,
)
```

The AIME runner in [`aime_query_sensitivity_uniform.py`](aime_query_sensitivity_uniform.py)
supports the historical priors, causal gates, and the exploratory
entropy/confidence variants with this same two-threshold protocol:

* `M_prior` applies the prior to margin uncertainty;
* `C_prior` applies the prior to confidence uncertainty;
* `T_prior` applies the prior to temporal flip uncertainty;
* `T_smooth` is `T_prior` with a slower trajectory EMA (default 0.8);
* `T_hybrid` adds top-1 confidence drift through a noisy-OR hazard.
* `T_run` decays uncertainty with consecutive accepted calls and resets it
  when a query is renoised or flips.
* `H_anchor` blends the completed-call trajectory prior with processed entropy;
* `C_soft` uses a tempered confidence prior that does not let `q=1` erase
  per-query confidence;
* `C_tail` adds causal protection only to the high-confidence-uncertainty tail;
* `M_gate`, `M_gate_norm`, `C_gate`, and `T_gate` add stable-run hysteresis.
* `C_gate_soft` softens the rejected-query floor while retaining `C_gate`'s
  trajectory term.

## C_gate: causal confidence prior with stable-run hysteresis

`C_gate` remains the strongest current method. It starts from
the confidence uncertainty used by `C_prior`,

\[
 u_i^t=\sqrt{\max(1-p_i^t,0)},
\]

and combines it with the previous renoising trajectory `q` and a causal
stable-run count `r`. Define

\[
 a_i^t=1-\exp(-r_i^t/\tau),\qquad
 h_i^t=1-a_i^t(1-q_i^t)(1-u_i^t),
\]

\[
 s_i^t=\operatorname{clip}(1+\beta h_i^t,1,1+\beta).
\]

The first call has `r=0` and therefore `h=1` for every query. A query remains
high-sensitivity for the first few accepted calls, then relaxes toward the
ordinary confidence/trajectory prior. A rejection or top-1 flip resets the
stable run for that query. All state used by the next call is produced after
the previous sampler result, so the method is causal. `tau=2.5`,
`beta=3`, and trajectory EMA `gamma=0.65` were used in the finalist run.

The uniform adapter still installs exactly one local/global threshold pair on
call 1, call 2, and every later call. The gate changes only the per-query
sensitivity values; it does not add an early/late threshold schedule.

### Observed AIME26 result

On 30 prompts with seeds 42, 43, and 44 at approximately 50% physical
sparsity, `C_gate` reached **54.44% pooled accuracy** with
**50.06%/49.78%/50.13% overall/global/local sparsity**. Mean canvas calls were
**18.63**, compared with 19.12 for Gaussian32 s50 and 14.97 for dense. Fresh
seed-42 request timing was 9.035 seconds on average, versus 9.125 for
Gaussian32 and 7.069 for dense. The seed accuracies were 60.00%, 50.00%, and
53.33% (5.09 percentage-point sample SD).

The phase profile was 48.20% sparsity on call 1, 46.41% on call 2, and 50.37%
on later calls. This confirms that the gate naturally protects early calls
under one threshold pair. It does not yet reach the desired 17-call target.
The complete measurements are in
[`aime_query_sensitivity_gated_v2/report.md`](/home/exouser/aime_query_sensitivity_gated_v2/report.md).

## Signal-validity audit completed (2026-10-01)

[Full report](/home/exouser/aime_query_signal_audit_v1/analysis/report.md).
All 270 traces (dense/C_gate/T_prior, 30 questions, seeds 42--44) reproduce
archived predictions and native denoising counts exactly. The six existing
calibration IDs are 2, 8, 14, 20, 23, 30; established formula diagnostics use
24 other question IDs for a predictive holdout. All 30 were already exposed
in task-quality evaluation. Future flips/renoising are instability proxies,
not causal labels for the benefit of extra attention.

The within-call AUROC for the next-call renoise-or-flip label is:

| Formula/signal | Dense | C_gate trajectory | T_prior trajectory |
|---|---:|---:|---:|
| Processed confidence uncertainty | 0.990 | 0.964 | 0.926 |
| Processed entropy H/(1+H) | 0.991 | 0.967 | 0.931 |
| Margin uncertainty | 0.981 | 0.960 | 0.925 |
| C_gate | 0.886 | 0.866 | 0.841 |
| T_prior | 0.893 | 0.873 | 0.853 |

The loss is structural: at q=1, the prior formula h=q+(1-q)u erases u;
at stable_run=0, the gate forces h=1. The latter has exactly 0.5 AUROC within
currently rejected queries, versus entropy's 0.951/0.940/0.921 on the three
trajectory conditions. On C_gate traces, the accepted-query AUROCs are 0.636
for C_gate and 0.768 for entropy. C_gate saturation (h>=0.995) is 95.9% on
routing call2, 77.9% on calls3--8, and 35.1% on calls9+. Phase statistics use
complete one-call future windows, and later calls have a survivor bias.

Margin is strongest for predicting flips alone. Confidence/entropy rank
future renoising better. The ranking advantage persists across three seeds,
two horizons and 128-query groups. These checks do not prove an improvement
in accuracy, denoising steps, latency, or accuracy variance. Query ranking
budgets are also not physical tile sparsity budgets.

Earlier text attributed excess calls to slow relaxation. That causal claim
was unsupported: faster relaxation means less attention, which could worsen
convergence. The audit replaces that claim with the measured loss of query
ranking above.

## Exploratory H_anchor pilot

A separate candidate now uses a convex prior:

    u_H = H/(1+H)
    h = 0.1*q + 0.9*u_H
    s = clip(1+3*h, 1, 4)

q starts at 1 and uses the previous sampler's renoising mask (gamma=0.65).
Call1 is exactly s=4. Only completed-call entropy enters the next call.
Every call uses one frozen local/global log-threshold pair.

H_anchor and its 0.90 mixture were explored after inspecting the diagnostic
holdout; its offline results are exploratory development evidence. No hazard
weights were fitted exclusively on the six IDs. The pilot output root is
`/home/exouser/aime_query_sensitivity_anchor_v1`; it first evaluates seed42
before considering any extension. The sparse gate/prior controls remain
archived comparisons. Improved predictive AUROC alone cannot justify more
expensive evaluation if denoising steps regress.

A conservative next formula, if needed, is a confidence EMA initialized at
one: e0=1; e_next=gamma*e_previous+(1-gamma)*sqrt(1-p). Unlike hard rejection
or stable-run floors, every completed confidence observation affects the
next coefficient. It preserves causal early protection without learning a
large multi-signal model. This variant remains a proposal; C_gate, M_prior,
and T_prior remain controls.

## Selective causal refinements evaluated

The two exploratory pilots do not meet the denoising-call target. At matched
calibrated sparsity, `H_anchor` gave 60.00% on seed42 with 24.54 mean canvas
steps; `C_soft` gave 43.33% with 23.73 mean canvas steps. Their call-1
sparsities were 33.49% and 36.92%, while later calls were 52.49% and 52.26%.
The early protection therefore moved the fixed sparsity budget to later calls
without reducing the number of calls. This is an observed trajectory effect,
not a calibration defect.

The next candidate should keep the absolute uncertainty level, and use rank
only to protect the high-risk tail. For the 256 query positions in one call,
let `u_C=sqrt(max(1-p_top1,0))`, let `r_C` be its tie-aware percentile rank in
`[0,1]`, and let `tail=((r_C-tau)/(1-tau))_+` with `tau=0.75`. Use

    h_Ctail = clip(u_C + lambda*q*tail*(1-u_C), 0, 1),  lambda=0.5
    s_Ctail = clip(1 + beta*h_Ctail, 1, 1+beta)

The first call still overrides this expression with `h=1`. Later `q` comes
only from the preceding completed renoising mask. This keeps stable queries
cheap as their absolute confidence uncertainty falls, while a causal prior
adds sensitivity only to the top quarter of the current uncertainty ranking.
It also avoids both failure modes in the audit: `q=1` no longer erases the
base signal, and a rejection no longer forces every query to `h=1`. On the
held-out traces this tail lift preserves direct-confidence ranking (C_gate
either-event AUROC 0.9636 versus 0.9637 for `u_C`; T_prior 0.9262 versus
0.9260), while leaving the phase means close to the raw uncertainty level.
Pure percentile normalization is not the primary choice because its mean
does not decay when a trajectory becomes stable.

The frozen `C_tail` pair was evaluated on all 30 prompts with seed42. It
reached 40.00% accuracy, 49.89% overall sparsity, 23.93 mean calls, P90 48,
and 20 capped canvases. Its phase sparsity was 36.25% / 39.63% / 50.89%.
The improved ranking did not improve the generation trajectory, so it was not
extended to seeds43--44.

As a narrower gate-preserving test, `C_gate_soft` used

    h = 1 - (lambda + (1-lambda)*a(r)*(1-q))*(1-u_C), lambda=0.2

where `a(r)=1-exp(-stable_run/gate_tau)`. This removes the exact rejected-query
saturation while retaining the `C_gate` trajectory term. Its seed42 pilot
reached 40.00% accuracy, 50.26% overall sparsity, 20.11 mean calls, P90 35,
and 6 capped canvases. Phase sparsity was 46.07% / 45.31% / 50.73%.
It was also stopped before seeds43--44 because of the accuracy loss and the
failure to reach 17 calls.

These negative results narrow the next decision: keep `C_gate_tau2p5` as the
primary method. The two refinements improve coefficient selectivity in replay,
but neither improves task quality or denoising cost under one uniform
threshold pair. Do not run another full GPU sweep until a new formula changes
the phase allocation or has a direct counterfactual attention-value signal.

## General confidence-memory refinement

HumanEval exposes a failure mode that is independent of programming syntax:
the first sparse call can move a query onto an incorrect trajectory, and a
later call can assign high confidence to that trajectory. The native stopping
rule then observes a stable answer but has no correctness signal. On the
completed 100-question run, confidence and entropy still predict the next
call's renoising well, with C_gate confidence uncertainty at AUROC about
0.964, so the useful intervention is to preserve early uncertainty briefly.

The benchmark-agnostic candidate is `C_ema`. Let

    u_t = sqrt(max(1 - p_top1,t, 0))
    e_t = rho e_{t-1} + (1-rho) u_t,   e_0 = 1
    h_t = max(q_t, e_t)
    s_t = clip(1 + beta h_t, 1, 1 + beta)

Here `q_t` is the EMA of the previous completed renoising mask, and `e_t` is
an uncertainty debt that decays after confident calls. The first call remains
at `s=1+beta`; the first observed confidence only affects the next call. A
query that becomes confident immediately still receives a short causal
protection window, while a query that stays accepted and confident relaxes
geometrically. No task type, token class, syntax rule, or correctness label
enters the method.

The first implementation uses `rho=0.65` and keeps C_gate as the control. The
30-question balanced HumanEval development run is in
`humaneval_confidence_v1.py`; it calibrates one local/global pair for each
sparse method using tile counts only, then evaluates dense, Gaussian32, C_gate,
and C_ema at matched 50% physical sparsity. The result root is
`results/humaneval_query_sensitivity_confidence_v1/`.

If C_ema improves accuracy but exceeds the call target, the next refinement is
a gated debt update that decays faster for accepted queries and retains the
slow memory only when `q_t` or current uncertainty is high. If it reduces
accuracy, the evidence will distinguish two cases: early uncertainty was
insufficiently protected, or the sparse value operator changes the trajectory
before confidence can observe it. The latter requires an operator-level
intervention, not a more elaborate temporal formula.

The original `T` remains the control. The default AIME targets are 40%, 50%,
and 60% physical sparsity, where the archived temporal method underperformed
Gaussian32. Set `AIME_TARGETS` to change the target set. Calibration searches
complete six-question trajectories, then freezes thresholds for all 30
questions and seeds 42--44.

Calibration should search only these two values on a disjoint calibration
cohort, freeze them, then evaluate quality, executed sparsity, denoising-call
counts, and confirmation seeds on untouched prompts. Because the native kernel
uses the sensitivity in its row-risk product, the coefficient is squared in
the risk calculation; the larger early coefficients therefore have a strong
but explicit effect on retention.

## What can be checked without a GPU

`query_sensitivity_uniform_analysis.py` audits the coefficient schedule from
archived aggregate diagnostics. It is deliberately labelled an expectation
proxy: aggregate renoised and flip counts cannot reconstruct per-query masks,
so it cannot establish sparsity or output quality. On the archived `T_s70`
trace (`beta=3`, `gamma=0.5`) it predicts mean coefficients of approximately
4.00, 3.56, and 3.19 on calls 1--3, versus 1.00, 1.00, and 1.16 for raw `T`.
The proposed value then declines as the renoised fraction falls, while the
flip term keeps unstable positions protected. This is the intended direction,
not a quality result.

The AIME-specific audit in
[`aime_query_sensitivity_offline.py`](aime_query_sensitivity_offline.py) shows
that archived 50% AIME temporal trajectories have approximately 3.95 mean
`T_prior` coefficient on call 2 and 3.89 on call 3 because roughly 90% of
positions are still being renoised. This warns that AIME may require a much
lower uniform threshold than RULER. The run-length estimate is particularly
noisy when only aggregate flip counts are available.

The focused CPU tests cover the coefficient bounds, causal update ordering,
canvas reset, invariant threshold selection, and all causal variants. The
latest run in the `ljy_dlm` environment passed 45 tests in 4.79 seconds:

```text
45 passed in 4.79s
```

The completed H100 AIME26 evaluation is recorded in
`/home/exouser/aime_query_sensitivity_uniform_causal_v1/report.md`. It covers
all six causal methods, 30 prompts, and seeds 42--44 at a 50% target, with one
frozen local/global threshold pair shared by every call. `M_prior` reached
52.22% pooled accuracy at 50.09% overall sparsity, exceeding the archived
Gaussian32 s50 control (51.11% at 47.60% sparsity) by 1.11 percentage points.
`T_prior` matched Gaussian32 at 51.11%; `C_prior`, `T_hybrid`, `T_smooth`, and
`T_run` were lower. Seed standard deviations were 5.09, 3.33, 8.39, 11.71,
and 5.09 percentage points for M_prior, C_prior, T_hybrid, T_smooth, and
T_run, respectively; therefore the causal variants did not uniformly reduce
seed variance. The calibration IDs overlap the historical AIME manifest, so
these numbers are development evidence rather than held-out confirmation.

## Related signals from prior work

Several recent approaches support the same design direction: protect early or
unstable positions, then stop spending compute as stability is established.
SparseD reports that early diffusion steps are critical and uses dense early
attention before reusing sparse patterns ([paper](https://arxiv.org/abs/2509.24014),
[code](https://github.com/horseee/SparseD)). *Just on Time* uses causal
top-1 stability and confidence ratios for token-level early stopping
([paper](https://arxiv.org/abs/2602.11133)). Prophet uses an early-converging
answer signal and the top-two confidence gap ([paper](https://arxiv.org/abs/2508.19982)).
KLASS and Stability-Weighted Decoding likewise turn token stability into a
compute or decoding weight ([KLASS](https://arxiv.org/abs/2511.05664),
[Stability-Weighted Decoding](https://arxiv.org/abs/2604.17068)). These works
do not validate this exact value-direction-aware kernel coefficient, but they
motivate replacing an artificial step-index calibration with a causal
uncertainty/stability state.

## HumanEval confidence follow-up (full cohort)

The full seed-42 HumanEval cohort distinguishes early uncertainty from final
correctness. C_gate failures are long-tailed (13% of questions, 32.0% of all
calls) and 12/13 finish with zero top-1 flips, native stopping, and entropy
below 0.01. Thus stable late confidence describes convergence to the sparse
trajectory, not correctness. A short early uncertainty signal is useful but
only moderately predictive: first-call question confidence has AUROC about
0.68 for dense-pass/sparse-fail cases.

Two general causal controls were evaluated at matched 50% sparsity with one
local/global threshold pair: corrected `C_selective` (EMA initialized from the
first completed confidence) reached 84% with 11.93 mean canvas calls and 8
caps; `C_soft` (alpha=.25 confidence/trajectory blend) reached 80% with 15.17
mean canvas calls and 15 caps. Both are rejected. A weak `C_gate_rank` residual
reached 88% with 9.38 mean canvas calls, still below C_gate's 87%/8.45 and far
above the <7 target. These results reject both persistent uncertainty debt and
un-gated confidence ranking as the next method.

The first candidate was a bounded, causal early risk pulse: robust confidence
excess from the first completed calls, with a fixed short decay, added as a
floor to the existing C_gate hazard. It retained uniform thresholds and native
stopping, but its full-cohort result was negative. The first-call tail pulse was
also evaluated. It was 93.33% with 6.50 mean
total calls on the balanced 30-question development sample, but 87% with 12.09
mean total calls on all 100 questions. It gained three C_gate failures and lost
three C_gate successes, with the losses producing 58--61-call tails. The next
confidence candidate should therefore use causal uncertainty progress: lift
only when confidence stalls or worsens, and remove the lift when confidence
recovers.

## Next iteration: kind-conditioned confidence progress

The full HumanEval split gives a more actionable failure signature. C_gate
failures have lower confidence and higher entropy at both early calls, but also
receive higher physical sparsity (call 1: 49.2% versus 45.5% for passes; call
2: 45.4% versus 41.6%; late: 50.5% versus 42.8%). Therefore increasing the
scalar sensitivity for an uncertain query is not sufficient: the current row
signal is not reliably preserving the tiles that repair a hard trajectory.

The independent attention audit points to a possible source. Gaussian32's
sampled local output error is 0.257, 0.424, and 0.442 on calls 1--3, versus
global error 0.232, 0.191, and 0.197. BLASST shows the reverse local/global
pattern, so the statistic must be measured for C_gate rather than assumed.

The proposed next method, `C_kind_progress`, keeps the C_gate hazard and adds a
bounded prior-call term per attention kind:

```text
u_t(q) = sqrt(1 - p_t(q))
g_t(q) = EMA(clip((u_t(q)-u_{t-1}(q))/(scale_kind+eps), 0, 1))
a_t,k(q) = robust EMA/percentile of rejected or near-threshold KV mass
h_t,k(q) = h_C_gate,t(q) + lambda*g_t(q)*a_t,k(q)*(1-h_C_gate,t(q))
s_t,k(q) = 1 + beta*h_t,k(q)
```

`a_t,k` is updated only after the completed attention/sampler call and is
normalized separately for local and global layers. Call 1 remains at maximum
sensitivity. The adapter still exposes exactly one frozen local and one frozen
global log threshold for every call; the kind state only changes row ranking.
First instrument C_gate on a balanced HumanEval slice and a small RULER slice,
then implement the term only if the kind-specific statistic predicts the next
call's confidence deficit. A row skip fraction is the initial proxy; a
near-threshold rejected contribution is preferable if the kernel can expose it.
