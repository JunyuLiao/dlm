# Value direction, denoising decisions, and reuse: implementation plan

Status: authorized implementation and evaluation; 2026-10-15 (Hong Kong).
Starting source: `fc19cceb`, branch `value-aware_cross-step-reuse_v2`.
Study root: `results/v31_value_decision_reuse_20261015/`.

## Objective and constraints

Improve official accuracy over the inherited Yuhan V31 control and reduce
native denoising work. Fewer calls are a potential speed benefit, not measured
speedup. Clean timing remains paused throughout this study. Previously paused
campaigns are not resumed. Exact greedy remains excluded.

Every sparse consumer uses exactly the same number K of eligible prefix KV64
blocks per query head and Q128 tile, with mandatory boundary/canvas blocks kept
separately. K is fixed across tiles in each workload: LongBench/RULER K=128;
AIME/MATH K=27, matching the recent short-workload control. Fewer available
prefix blocks means keep all of them. C_gate changes row weights before
aggregation, never K. Dense bootstrap, prefill and commit phases are accounted
separately and shared with the control. No hidden model forwards, sampler edits,
extra acceptance, altered stopping, or output-limit changes enter native arms.

## Direction 1: bounded, mass-neighbor value selection

1. Start from the actual V31 map, including its named sticky/progress rules.
   Maintain a separate fresh mass-only map for factorial diagnostics.
2. Construct candidates by global attention-mass rank, not physical position:
   the anchor union the best K+32 prefix blocks. Diagnose additions from V2's
   online risk within this mass-qualified pool. Keep an explicit ablation using
   online ranking versus true full-output ranking; both use full-dimensional V.
3. Restrict removal to the lowest-mass 16 retained blocks and addition to the
   best 16 eligible outside blocks. Evaluate bounded simultaneous proposals
   with swap caps 2 and 8. This is not exact greedy deletion. Re-evaluate the
   combined proposed support, including its denominator, rather than summing
   singleton errors. A no-op is valid and preserves K.
4. Primary score is signed full-V output error, transformed by the real O
   projection; pre-projection error is an ablation. Account for cross-head
   cancellation when moving from single-head screening to whole-layer replay.
5. Admit a proposal only if its current weighted squared error improves and its
   worst-row error and retained row mass remain within predeclared limits.
   Limits: maximum row-mass loss 0.025; worst-row error no more than 1.02 times
   the anchor, with a numerical tolerance of 1e-6. Preserve the stronger
   full-information guard for the online proposal variant.
6. C_gate uses only previous native sampler statistics (unaccepted rows,
   argmax changes, acceptance reversals); clip weights to [1,4]. Test neutral
   weights and shuffled row weights. Missing sampler statistics are not
   replaced by fabricated sensitivity or future labels.

## Direction 2: value selection over the lifetime of a reused map

At an authorized observation, use the previous query displacement to estimate
the effect of one further step. For fixed prefix K,V and scale s:

`dO = sum_k p_k (v_k - O) * (s * dQ dot k_k)`.

The directional derivative needs two extra moment reductions, not a full
Jacobian. Fit neither directions nor parameters using future frames. Compare
current-only selection with a two-state objective (current plus first-order
predicted next output, equal weights). Cap extrapolation to the previous query
displacement norm; reset history at canvas changes. Future actual queries and
outputs are used only to score prediction, never to select the map.

Evaluate reuse ages 1,2,4,8 and all available held calls. Compare with simple
age, query-change norm, and mass drift. Include an exact future-state oracle
only as a diagnostic ceiling. Separately test a causal reuse/repair guard at
fixed K: a failed guard may trigger the named existing observation path, but
all extra observation work must be counted. Retained-only statistics cannot
certify omitted blocks; any outside probe has a fixed, separately counted
budget and must demonstrate useful detection before native promotion.

## Direction 3: reconstruction of omitted mass and value

This direction changes the output consumer and has separate receipts and
controls. It must not be called selector-only or unchanged FA4 output.

Test two explicitly distinct reconstructions on the same fixed-K support:

* Dense-anchor residual: `O_sparse(t) + O_dense(a) - O_sparse(a)`.
* Cached omitted numerator and denominator: combine current selected
  unnormalized numerator/mass with anchor omitted numerator/mass, using a
  common softmax scale. A first-order omitted-moment update is a separate arm.

Always reconstruct the entire selected-plus-mandatory distribution. Maintain
consistent log-sum-exp scaling, including when maxima change. Reset at canvas
and mask changes unless a validated remapping explicitly reconstructs the new
complement. Compare fixed, causal risk-gated and zero correction. Exact
fresh omitted moments are only an oracle, since computing them removes the
intended savings. Existing residual-cache and compensation literature prevents
claiming novelty for correction alone; the hypothesis is decision-safe reuse.

## Ordered validation and promotion

### A. Independent mathematical and implementation checks

Implement new isolated modules; do not borrow the predecessor selector as an
oracle. Check against direct float64 softmax and finite differences. Cover
noncontiguous mass ranks, exact K, all-kept, short/partial tiles, mandatory
tails, ties, no candidates, signed cancellation, simultaneous swaps, online
order dependence, GQA mapping, changing normalization, cache reset, derivative
sign and extrapolation. CPU tests precede pinned-H100 parity. Optimize by
batched GEMM/moment reductions and bounded candidate evaluation; do not add
timing benchmarks while timing is paused.

### B. Existing same-state operator screen

Verify the immutable 670-file Q/K/V inventory from the stable-refresh study.
Use its 620 frames across all five GLOBAL layers and both workloads. First
screen selected heads/tiles, then broaden promising variants to all heads.
These are four previously examined trajectories, not new task accuracy.
Freeze trajectory splits before calculation and publish results per trajectory,
layer, workload, and reuse age. K=27 for short-task replay is a new operator
budget and does not relabel the source K=128 trajectory.

Report true full-V and post-O errors (mean, p95, maximum), retained mass,
IoU, swaps, prediction residual, sign of next-state improvement, correction
harm rate, memory, and counted discovery/probe/consumer work. A mean reduction
alone is insufficient. Persist one complete group at a time with immutable
attempts, source/input hashes and completion markers. No output tokens or raw
tensors are published.

### C. Same-state model decision diagnostics

Only operator-qualified variants proceed. Capture additional LongBench,
AIME/MATH and RULER (especially variable tracking) states, using frozen
prompt/seed selections independent of correctness. Verify observation-only
invariance against a source-matched control twice. At identical model/sampler
states, measure next-logit KL, top-1 changes, margin-normalized errors,
acceptance-mask changes, entropy and stopping changes. Reuse exact RNG draws
for stochastic comparisons. Extra suffix/full forwards are offline diagnostic
work and excluded from native N and latency claims. Validate all KV/RNG/cache
restoration before trusting intervention results.

Direction 1 must reduce decision disagreement relative to the mass anchor;
direction 2 must improve next-state risk prediction beyond age/query-norm
baselines and improve held-call decision errors; direction 3 must avoid stale
correction tails. Select at most one candidate per direction plus a combined
1+2 arm. Freeze choices before official accuracy. If only operator data are
available, mark this gate incomplete and do not claim a passed decision gate.

### D. Native development, then full accuracy

New source-frozen deployments, common inputs/seeds/runtime and official
scorers for every arm. Retain FULL fix51994, PIECEWISE dense, all-kept FA4,
V31, and the relevant ablation. Use 64 LongBench items, 30 AIME x 3 seeds,
and 104 RULER prompts; use MATH100 for short-task transfer. These reused
benchmarks are exploratory, not held-out confirmation. Rotate arm order in
resumable shards. Validate actual selector/budget/consumer receipts first.

Promote only a candidate with no observed accuracy loss on each tested
workload, at least one strict accuracy gain, aggregate N no larger than
control on each workload, no material cap/p95 regression (>5%), and complete
fairness/work accounting. These are screening criteria, not noninferiority
tests. If none pass, retain the negative results and do not spend a full panel
on a failed candidate. Full LongBench503 is authorized for survivors, followed
by independent seeds; disclose previously examined prompts and multiplicity.

Report official accuracy and paired/item-clustered intervals, N/problem, N/C,
canvas count, output length, tail/cap behavior, absolute GLOBAL/LOCAL eligible
and kept tiles, and all observation/selection/correction work. Instrumented
wall time is labeled audit time. Do not report clean S/N or speedup while
timing is paused.

## Execution and recovery

One worker on the idle H100; pinned interpreter/model/runtime and user-owned
caches. Check progress every 30 minutes once a long experiment is launched.
Write durable state, failure reasons and source hashes. Partial attempts remain
immutable; resumption writes a new attempt and reuses only validated completed
groups/shards. No asynchronous queue may bypass a diagnostic/promotion gate.
Update HANDOFF and decisions as verified stages finish. Ledger entries require
scoped completion and source/protocol identity. No claims of success before
measured evidence, and no whole-study completion marker for partial work.

## Evidence-driven amendment after the first operator screens

The previous-query tangent is rejected: it worsened age-1 output-error
prediction on all 100 screened sparse units. Its derivative implementation
passed central finite differences; the failure is the extrapolation assumption.
Direction 2 therefore proceeds with causal trust in the cached value residual,
coupled to direction 3, rather than promoting a failed query predictor.

For r=D0-S0 and m=St-S0, use alpha=clip(1-<m,r>/||r||^2,0,1).
The norm-only coefficient clip(1-||m||/||r||,0,1) is the ablation. Both use
whole-layer post-O vectors, preserve K, and require no current omitted-key
output. They can still fail when the dense target drifts. The exact conditional
regret bound and its assumptions are documented separately; no current-error
improvement or task-accuracy guarantee is asserted.

On the two LongBench source trajectories, the direction guard's mean-error
ratio grows from 0.557 at reuse age1 to 0.717 at age9+, retaining an observed
benefit beyond fresh calls. Those correlated age buckets are descriptive, not
independent samples. The bounded full/online swap improvements are smaller
and diminish from about 2% at age1 to 1.3% at age9+.

The final-layer decision probe uses privately coupled diagnostic Gumbel draws,
not the compiled sampler's actual noise. Native RNG is preserved and checked.
Its first phase qualifies instrumentation on one previously captured LongBench
item; this is not the multi-workload promotion gate. Two uninstrumented/observer-
disabled/observer-active repetitions are now required after detecting an
observer allocation bug in an immutable failed attempt. Native accuracy is
not launched until valid same-state model measurements support promotion.
