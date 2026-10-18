# Bounded refresh diagnosis: completed execution and research decision

The full 2×2 panel and clean runtime comparisons are finished. **The strongest
provisional mechanism signal is preserving useful future support, particularly
when the current output is sparse. Neither a dense-output-preservation claim nor
a deployable cheap-refresh improvement is established.** The next investment
should resolve numerical repeatability and qualify a regime with enough measured
runtime headroom before another method sweep.

This is an exploratory, single-model, batch-one diagnostic. It does not establish
benchmark superiority, noninferiority, or an MLSys contribution. Yuhan's inherited
attention-mass selection/cross-step reuse remains the reference; these experiments
introduce diagnostic interventions, not a replacement production method.

## Frozen design and actual coverage

The plan is `docs/REFRESH_DIAGNOSTIC_PLAN_20261018.md`, with the explicit recovery
and external-trace addenda. All runs preserve the native sampler, output limits,
256-token canvas and dense prefill/commit. V31, fixed16 and factorial sparse-method
arms use Q128/KV64 selection, K8192 eligible prefix tokens and dense LOCAL
attention; dense/all-kept controls do not apply this sparse budget. This V31
configuration retains sticky=1.386;
it is distinct from the Slack stickiness-off comparison. Fixed16 changes only the
progress refresh clock, retaining initialization, carry and stickiness.

At the first genuine mid-canvas progress refresh, all five GLOBAL layers receive
one of four interventions:

| Arm | Current attention output | Future support |
|---|---|---|
| dense_fresh | Exact dense | Exact fresh mask |
| dense_hold | Exact dense | Pre-refresh mask |
| sparse_fresh | Native sparse on the pre-refresh mask | Exact fresh mask |
| sparse_hold | Native sparse on the pre-refresh mask | Pre-refresh mask |

All four compute both attention alternatives as diagnostic oracles, without an
extra model forward, sampler call or RNG draw. Original V31 runs before and after
every group; dense_fresh is the no-op treatment. Treatment order is rotated and
reversed across independent engines. Later-layer inputs are treatment mediators;
the experiment does not force identical fresh masks across different current-
output branches.

- Factorial: eight LongBench-v2 questions across four length bins, four RULER128K
  task-family questions, seeds 1/42 and two engine repetitions. All 288 measured
  requests / 48 six-arm groups completed. Inputs were selected from metadata,
  independently of outcomes, but were previously examined: this is not held-out
  confirmation. Technical repeats are not independent answer observations.
- Headroom: two selected LongBench inputs, 41,189 and 111,616 tokens, two clean
  repetitions of V31, fixed16, corrected FULL dense, PIECEWISE dense and all-kept
  FA4. All 20 clean requests completed. Separate shared/FULL profiles completed;
  the all-kept profile was skipped unstarted under its phase allocation.
- External check: the original clean worker, copied private cache, unchanged
  schedule and audit-disabled flags, traced once with Nsight Systems. Both
  warmups and all eight measured requests reproduced the original clean outputs,
  N/C/T, input/configuration bindings and adapter counters exactly.

The original source is 6aab8632; the recovery source is db755c43, with byte-identical
native/factorial execution code and explicit references to completed original
leaves. External runner 2f5687d4 executes the original 6aab8632 worker. Protocols and
receipts retain full hashes. The profiling startup failure and CPU-only recovery
preparation failure are preserved; neither is silently overwritten.

## GPU budget and completion boundary

| Phase | Final allocation | Actual reservation |
|---|---:|---:|
| Qualification |15 min|7.87 min|
| Headroom, including failed worker |30 min|28.23 min|
| Factorial |65 min|48.86 min|
| One external trace and processing |10 min|5.44 min|
| Total |120 min|90.40 min|

Actual charged reservation is 5423.862923 seconds, including startup, warmups,
failure and external export/analysis. No unresolved reservation or GPU worker
remains. The older full503 campaign remains paused.

The recovery explicitly reassigned five reserve minutes to headroom; there was
no automatic borrowing between phases. The missing all-kept profile leaves
attempt003's overall state `budget_limited_partial`, despite complete factorial
coverage. Its global `final_complete.json` is absent. This report completes the
bounded analysis, not the missing profile or a full benchmark campaign.

## Mechanism findings

Every group reached the intended event. All-six recorded prefix fingerprints
match; factor separation, exact K, five-layer scope, unchanged LOCAL/RNG, zero
added model forwards or sampler calls, and zero request-time CUDA captures pass
independent verification.
Future support is actually consumed: 800 treated layer events use same-canvas
held sparse execution, 160 use next-canvas carried sparse execution.

LongBench means over 32 paired groups (eight question clusters):

| Current output / future mask | Official accuracy | Mean denoising calls N |
|---|---:|---:|
| Dense / fresh |50.00%|126.375|
| Dense / held |56.25%|142.813|
| Sparse / fresh |62.50%|133.531|
| Sparse / held |50.00%|150.563|

These small-cohort point estimates are not a ranking of deployable methods.
All four arms still pay for the dense observation in this diagnostic.

**Future support is the stronger provisional lead.** Holding the mask increases
the main-effect mean N by 16.734 calls (+12.88%; descriptive question-cluster 95%
interval −2.859 to 36.969). With sparse current output specifically, holding the
mask adds 17.031 calls (+12.75%; interval 6.031 to 29.126), with four repeated adverse
question/seed answer changes across three questions, and two beneficial changes
across two questions. Leave-one-question-out N differences stay positive for this
conditional contrast. This supports studying support quality and lifetime; it
does not validate the progress clock or a cheap mask-construction algorithm.

**Dense current output is not established as the protective mechanism.** Its
removal with a fresh mask changes mean N by +7.156 (+5.66%; interval −17.000 to 36.034),
but the sign reverses across seeds and when question 77 is omitted. Sparse/fresh
has three repeated beneficial versus one adverse question/seed answer changes.
The extra calls in the beneficial cases average 63.83; unchanged-score cases
average 10 fewer calls. These outcome strata are post-hoc explanations, not
filtered estimators. Every case remains in headline results. Fewer calls cannot
be treated as improvement when answer quality changes.

RULER's mean official task score is 0.875 for dense/fresh, dense/held and
sparse/fresh, and 0.8875 for sparse/held; this is a partial-credit task metric,
not a fraction of fully correct answers. Mean N is 5.5/5.75/5.875/6.125. All 16
groups are exposed, and both engines reproduce exactly. Small relative N changes
here represent less than one call in absolute terms; the sparse/held score
change is a single repeated partial-credit case.

## Repeatability limits and frozen gates

Two LongBench V31 bookends differ in output bytes and N (9 and 4 calls in absolute
value), with no answer-score differences. Mean absolute A/A N variation is 0.406,
maximum 9. One sparse/fresh technical repeat differs by 33 calls and output hash,
also without a score change. No cases were excluded or retried.

The qualification-only question 422 also differs from the main panel's trajectory;
it is not pooled into the estimator. Within-group prefix matching and no-op
checks therefore cannot be presented as full-engine cloning. Fingerprints cover
recorded sampler quantities and tensor bytes, not every layout, auxiliary tensor,
resolved FA4 configuration or compiled artifact. Full self-conditioning is hashed
only at the first post-event state. The earliest unobserved divergence and its
cause remain unresolved.

The frozen triage required at least six matched/exposed LongBench questions with
seed/engine coverage, effects assessed against A/A variation and tails, a 5% N
screen or repeated adverse answer changes, and a plausible ≥10% incremental clean
request benefit before investing in a deployable refresh method. Coverage and
structural checks pass. **Overall causal promotion remains unresolved under the
A/A rule.** The support result is a mechanism lead; the current-output result is
too conditional to justify a general dense-output requirement. Neither result
establishes a latency gain.

## Runtime headroom

Clean request times (two repetitions per selected input):

| Arm |41K request W|112K request W|41K /112K scores |
|---|---:|---:|---|
| V31 dynamic |8.628s|10.875s|1 /0|
| Fixed16 |4.433s|11.600s|0 /0|
| Corrected native FULL |6.901s|13.561s|1 /0|
| Matched PIECEWISE dense |7.108s|16.419s|1 /0|
| All-kept FA4 consumer |7.480s|19.006s|0 /0|

Fixed16's short-input timing advantage accompanies an incorrect answer. FULL,
PIECEWISE and all-kept controls also have different native trajectories. These
two inputs cannot establish benchmark accuracy or overturn the broader Slack
fixed/dynamic comparison. They demonstrate why clean timing, quality and consumer
controls must be retained together.

The Python-instrumented profiles changed all six matched trajectories. Their
progress-refresh spans are about 1.5% of profiled request time; even the generous
distinct bootstrap/initial/refresh/commit parent-span sum is about 5.2–5.5% on
dynamic V31. These are inclusive diagnostic spans, not transferable clean bounds;
discovery is a nested child and is not added again. FULL replayed decode scopes
were incompletely observed. Missing scope is unknown, not zero.

The external trace repairs trajectory and request-window measurement: all eight
measured requests match, there is one unique source-consistent 33-sync alignment
for 11 requests including warmups, and no GPU activity crosses accepted boundaries.
No monitored JIT warning or graph capture occurs during measurement. Original caches remain unchanged.

| Dynamic V31 trace |41K input|112K input|
|---|---:|---:|
| Traced W |8.914s|11.092s|
| Prefill boundary time /traced W |12.25%|44.27%|
| Attention-named decode GPU activity |1.043s|1.099s|
| That activity /traced W |11.70%|9.91%|
| Trace W increase over clean W |3.31%|1.99%|

Kernel-name categories are descriptive and retain unknown activity. They do not
identify GLOBAL refresh, selection, commit or removable critical-path costs.
GPU gaps do not establish host causality. The external trace is not clean timing;
its exact trajectory match does not make its durations a deployable speedup.
**The ≥10% incremental refresh-latency gate is not established; its strict
critical-path bound remains unresolved.** Batch size >1 and alternate models were
not tested, so no larger-batch benefit follows.

## Research decision and next bounded direction

1. **Measurement gate first.** Reproduce the observed same-recorded-input dense
   return difference in a bounded native attention operator check. Record full
   tensor layout/dtype/scale/auxiliary metadata, resolved native SplitKV settings,
   compiled identities, partial outputs/LSE and final output error magnitudes.
   Separate legitimate numerical variation from a defect before using small
   trajectory differences as an algorithmic contribution. Source review identifies
   these dependencies but does not diagnose the cause.
2. **Qualify the runtime opportunity.** Build on the successful external trace
   using lightweight phase attribution that itself passes trajectory parity.
   Measure initialization, progress updates, commit, prefill and selection on
   useful workloads. Then qualify actual batching or a model with a larger
   attention share; changing batch/stride is not evidence of increased headroom.
   Continue only when measured removable cost, minus proposed replacement cost,
   plausibly supports the frozen ≥10% incremental request benefit at matched
   quality. Stop a named regime if a valid optimistic bound is below that bar.
3. **Conditional algorithm lead: support maintenance tied to native decisions.**
   If the runtime gate passes, preserve effective future support while reducing
   observation cost. Admission and lifetime should be judged by native acceptance,
   self-conditioning/trajectory stability, answer quality and actual physical work,
   rather than retained mass or operator error alone. Use this one-event oracle
   as a diagnostic reference, then test a concrete cheap selector across repeated
   refreshes; one-event results cannot certify replacing every refresh.

No further GPU experiment is queued by this report. V31 and fixed scheduling
remain controls; dynamic timing gains no renewed contribution claim from these
results. The current evidence supports a targeted measurement/support-maintenance
research agenda, not another broad value-score sweep or a paper-level novelty
claim.

## Evidence map

- Preregistered plan and addenda: `docs/REFRESH_DIAGNOSTIC_PLAN_20261018.md`,
  `docs/REFRESH_DIAGNOSTIC_RECOVERY_20261018.md`,
  `docs/REFRESH_EXTERNAL_PROFILE_20261018.md`.
- Factorial estimates, all cases and reproducible analysis:
  `attempt003/factorial_analysis/attempt001/{summary.json,mechanism_review.md,mechanism_review.json}`.
  The supplement qualifies the frozen summary's overly broad "full-state" label.
- Independent verification: `attempt003/receipts/factorial_verification.{md,json}`.
- Headroom: `attempt003/headroom_analysis/attempt001/trajectory_and_scope_review.md`.
- Source reviews: `attempt003/headroom_analysis/{source_path_review,numerical_dependency_review}/attempt001/`;
  read the source-path review's status-only correction sidecar with its original receipt.
- External parity, activity windows and reservation:
  `external_headroom/attempt001/{receipt.json,parity.json,activity_summary.json,verification.md,verification.json}`.
- Campaign accounting: `gpu_budget.json`; execution status: `attempt003/state.json`.

All public evidence is sanitized. Raw generations, prompts, pools, caches and
profiler traces remain in ignored private directories. Main diagnostic CPU tests
(67) and external-runner tests (20) passed; independent analysis recomputation and
artifact/native receipt checks passed.
