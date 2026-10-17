# Full503 end-to-end evaluation of value swaps and directional correction

Authorized 2026-10-17 (Hong Kong). Starting source78e8eb9f, branch
value-aware_cross-step-reuse_v2. New study root:
results/v31_value_e2e_longbench503_20261017/.

The user explicitly requests clean end-to-end performance on all503 LongBench-v2
questions at panel seed42, for full-value swaps and direction-guarded residual
against V31 and dense. This is a new authorized experiment despite the previous
all-workload development screen having no survivors. That screen and its results
remain immutable. Clean timing is enabled for this study only; other campaigns
are not resumed. Previously examined questions remain identified as development
overlap, not called held-out confirmation.

## Fixed method and execution contract

Keep inherited Yuhan V31 selection/progress/carry/sticky machinery, native
DiffusionGemma sampling, model/runtime pins, thinking template and output limits.
Value changes remain at GLOBAL layer29; all other GLOBAL layers use V31, LOCAL
remains dense. Every head/Q128 tile keeps K=128 eligible prefix KV64 blocks
(all available if fewer), plus the inherited common mandatory/carry additions.
No change to refresh frequency, candidate pool, swap cap, mass/worst-row guards,
correction coefficient, sampler decisions or stopping is an optimization.

Primary arms: current_v31_control, full_swap8, residual_direction_guard,
dense_full_fix51994. Retain dense_piecewise and allkept_fa4 controls to expose
execution-mode/consumer effects. Material optimized implementations receive
explicit backend names and source-bound receipts. Reference implementation
comparisons are diagnostic, not an extra full503 arm unless a correctness
ambiguity makes them necessary.

Use all503 original LongBench-v2 question identities, same frozen prompt/token
manifests and native budgets, no correctness-based filtering or truncation.
Set panel_seed42 for every question and preserve the established request-seed
derivation (base31, dataset, index, panel seed42, repeat0); record actual derived
RNG seeds. This keeps comparisons matched to the repository's seed42 convention.
Check full manifest coverage, context capacity and storage before freezing.

## 1. Measure overhead before selecting optimizations

Use an idle H100 and existing immutable Q/K/V states plus synthetic boundary
geometries. Freeze the diagnostic input inventory independently of accuracy.
Measure warmed synchronized CUDA-event and host-wall times separately for:

* QK/normalization and block masses; full-V block moments;
* candidate construction, pair scoring, disjoint ranking and joint admission;
* support/list rebuild, O-projection/adjoint cache construction;
* held-call directional correction;
* complete observation and held attention-module paths.

Profile only diagnostic executions. Record allocations, host scalar reads,
kernel launches, peak memory and geometry. Compare the actual reference code,
not an artificially slowed control. Kernel measurements are not end-to-end
speedup and profiler overhead is excluded from clean performance.

## 2. Optimize within the same mathematical method

Prioritize observed bottlenecks, testing the following implementations in order:

1. Reuse FP32 K/V conversion and static O-derived data within observations;
   avoid recomputing unused moments and allocating unused correction arrays.
2. Batch independent heads/query tiles where memory allows. Keep all selection
   state on device; replace dynamic nonzero/scalar reads with fixed-shape
   candidate lists using known K and bounded pool sizes.
3. Factor the quadratic pair objective into cached transformed value vectors
   if projection GEMMs dominate. Re-evaluate exact combined-support guards;
   use an explicit numerical-ambiguity fallback where needed. This preserves
   the formula but changes rounding and must pass support/decision checks.
4. For correction, compute only the dense/selected aggregates actually needed
   and fuse anchor reductions where justified. Preserve normalization and full
   cross-head O geometry. Keep the existing one-kernel held correction; assess
   output-buffer reuse and omitted alpha stores without changing its arithmetic.
5. Evaluate overlap on a separate stream only if independent work exists before
   the next consumer. Establish event dependencies and tensor lifetimes. Never
   delay an update by one call or reuse an old correction merely to hide cost.

Do not implement every possible optimization blindly. Keep only measured wins
that pass correctness and memory checks. Full-dimensional V and exact fixed K
remain mandatory. Do not change shared installed environments or peer sources.

## 3. Correctness and method-equivalence gates

Run focused CPU/H100 tests against the independent existing FP32/FP64 reference.
Cover noncontiguous masks, ties/no candidates, short/partial tiles, mandatory
tails, normalization, all-kept, exact K, disjoint swaps, combined guards and
cache identity/reset. Require finite outputs and no out-of-bounds accesses.

Replay the saved same-state inventory at layer29. Report candidate/pair ranking,
mask equality, swap counts, guard outcomes, output/correction error and alpha
differences against the reference. Summarize every mismatch, including near ties.
Do not silently call a changed selector identical. Reject/fallback on material
changes; if only rounding differs, name the backend and run paired suffix/native
qualification before freeze. Preserve failed attempts.

Run native smoke requests spanning context lengths and long trajectories, with
both control bookends and both requested methods. Verify actual FULL/PIECEWISE
dispatch, selector/backend identity, layer scope, fixed K, native settings,
zero measured graph captures, all-kept/dense controls and official scoring.
Record control drift; retain structurally valid outcomes, never retry for an
unfavorable answer. No hidden model forwards enter production requests.

## 4. Clean timing and audit passes

Build a clean worker with request-boundary synchronization and one explicit
prefill/decode boundary for W and S. Remove the inherited synchronization and
timing events after every engine step. Disable routing accounting kernels,
profilers, diagnostic tensor transfers, per-layer events and scoring during
clean generation. Keep actual CUDA capture counting and minimal native N/C/T
counters. Retrieve receipts and decode/write completions after the timed region.

Warm compile/capture paths and candidate-specific buffers before measurement,
using a fixed common warmup inventory. Require zero new CUDA captures in timed
requests; predeclare any failed warmup/relaunch handling. Include all production
routing, observation, correction, allocations and required CPU/GPU transfers in
W and S. Model loading, JIT warmup and offline scoring are excluded and reported
separately. Report the exact timer boundaries.

The clean pass covers all503 questions for every primary/reference arm and is
officially scored. A separate instrumented audit covers the same503 questions
for physical-work/phase/selector accounting and is also scored independently.
Do not transfer accuracy or N from the audit to the clean run if trajectories
differ. Publish cross-pass hash/N/C/T/score agreement and differences. Only
clean-pass records determine clean latency and its associated quality.

Sparse candidates and V31 use a common warmed PIECEWISE engine with per-request
state reset and rotating order; V31 before/after bookends expose drift. Native
FULL dense uses its own engine. Retain matched PIECEWISE dense/all-kept controls.
Balance pass/arm order across resumable shards. Freeze schedules before any
full503 generation. Do not select a faster repetition or drop long/capped cases.

## 5. Statistics, decisions and artifacts

For every arm and each pass report official accuracy, N/problem, N/C, canvases C,
output tokens T, cap/finish rates and p50/p95 tails. Clean performance includes
prefill, generation S, request W, S/N, totals, paired per-question ratios and
speedup against both V31 and FULL dense. S/N is an amortized cost. Report both
ratio-of-total time and paired geometric mean, with20,000-draw fixed-seed
question-cluster intervals. Include correctness-conditioned and output-length
diagnostics as descriptive analyses, never alternative primary estimands.

Audit physical GLOBAL/LOCAL eligible/kept/skipped rectangles, stated overall
denominator, phases, observations, swaps/no-ops, held calls, correction exposure,
extra QK/PV/projection work, peak memory and GPU reservation seconds. Separate
kernel, module and end-to-end measurements. Decompose S=N*(S/N) and N=C*(N/C)
without claiming causal attribution. Quantify whether per-step overhead consumes
the call saving, and whether any speedup relies on worse accuracy or shorter
failed answers. Do not infer speed from tile counts or N alone.

Freeze spec → optimized source/tests → protocol/input/runtime pins → bind/smoke →
clean/audit generation → official scoring → independent aggregation audit.
Use one worker per GPU, immutable attempts, completion markers and resumable
shards. Monitor long runs about every30 minutes. Publish sanitized numeric
records, source fingerprints, receipts and plots under the new study root;
keep prompts, gold, tokens and raw completions private. Update HANDOFF,
DECISIONS and RESULTS_LEDGER only with verified state and completed evidence.
