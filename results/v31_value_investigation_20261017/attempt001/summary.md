# Decision: do not resume full503 for either current method

The bounded investigation is complete. Full-value swaps and direction-guarded
residual do not currently demonstrate a better accuracy/latency tradeoff than
inherited Yuhan V31. Full503 remains paused at17 completed engines,64 matched
questions per pass, with one extra32-question audit PIECEWISE shard preserved.
No further GPU worker is active. Resume still requires explicit user
authorization through the guarded procedure in the parent panel's
[PAUSE_AND_RESUME.md](../../v31_value_e2e_longbench503_20261017/native_attempt001/PAUSE_AND_RESUME.md).

This is a completed investigation of partial benchmark results, not a completed
503-question benchmark or held-out confirmation. The original frozen protocol,
sourceec87d397, completed/partial artifacts and private generations are retained.
Only the authorized bounded diagnostics ran after the pause.

## Verified performance

Officially rescored clean64, seed42; V31 averages matched control bookends.

| Method | Accuracy | Calls/problem | Tokens/problem | Generation S | Request W | W versus V31 |
|---|---:|---:|---:|---:|---:|---:|
| V31 | 46.875% (30/64) | 240.31 | 3229.67 | 5.869s | 8.914s | — |
| Full-value swaps | 45.313% (29/64) | 277.83 | 3336.42 | 11.196s | 14.242s | +59.8% |
| Directional residual | 46.875% (30/64) | 290.91 | 3628.95 | 8.375s | 11.420s | +28.1% |
| Dense FULL, fix51994 | 46.875% (30/64) | 252.13 | 3298.75 | 7.587s | 10.643s | +19.4% |

Accuracy differences have wide paired intervals spanning zero. Neither
superiority nor noninferiority is established. Request penalties remain
positive in both shards and in both-correct subsets. See
[CPU_FINDINGS.md](CPU_FINDINGS.md) for exact N/C decomposition, capped-output
rates, dense PIECEWISE/all-kept controls, paired intervals and outcome/tail
sensitivity. W is the defined native engine request interval; it excludes model
startup, warmup, offline tokenization and scoring. S/N is an amortized ratio.

## Why the earlier result reversed

There is no single established cause. The previous23.51% swap call saving was
real within its seed1/cohort:41/64 inputs improved, including correct answers.
Five inputs nevertheless explained89.69% of net savings, and four are absent
from the current cohort. The current negative mean is also tail-sensitive:
question144 alone explains82.47% of swap's net call penalty. Both sides of this
sensitivity matter; neither trimming exercise replaces the primary estimate.

Question composition is not enough. The eight common inputs reverse their
relative mean call effect. The predeclared96-request cross now separates some
factors on two of those inputs: swaps match reference/device in16/16 pairs,
and clean means change from V31280→swap176 calls at seed1 to224.125→233 at
seed42. Thus seed-dependent method behavior exists under a common implementation
and engine design. The optimized backend is not the explanation on these two
cases. This does not identify the cause on the remaining501 inputs.

Execution variation is also observed: one unchanged V31 bookend pair changes
100→241 calls without a correctness change. Residual has two audit backend
discrepancies, −11 and+47 calls, with unchanged correctness and no consistent
direction. The final trace matches both residual implementations exactly,
including26 correction caches, so those intermittent discrepancies remain
unlocalized. See [CROSSED_FINDINGS.md](CROSSED_FINDINGS.md).

## Why attention fidelity is not a convergence objective

The native trace supplies a concrete mechanism. On question0/seed1, residual
first changes attention/logits/self-conditioning at call2 while support,
sampled tokens and RNG still agree with V31. Q diverges at call3, acceptance
and actual canvas at call6, refreshed support at call8, and commit scheduling
at call18. RNG states first diverge at call19, after different commit schedules.
V31 finishes correctly in218 calls; both residual implementations finish
incorrectly in321. See [TRACE_FINDINGS.md](TRACE_FINDINGS.md).

Therefore the initial perturbation can propagate through self-conditioning
before token or mask overlap exposes it. Native confidence, entropy-ranked
acceptance and argmax-history stability are nonlinear, thresholded decisions.
The swap objective targets observation-time headwise post-O error; the residual
guard targets a stale dense anchor. Neither bounds future sampler decisions or
answer error. Residual's much better saved-state operator error does not rescue
its native result. The trace is one exploratory case, not a population theorem,
and does not measure current dense error on those held calls.

Current swap's+37.52 calls decompose into+8.49 from canvas count and+29.03 from
calls/canvas; residual's+50.59 split into+29.50 and+21.10. Even current
both-correct swap savings come from shorter output: the canvas term is−21.07
calls, while within-canvas convergence contributes+3.78. The old both-correct
subset benefited from both terms. All such outcome-conditioned comparisons
are descriptive rather than causal.

## Cost and realistic optimization limits

| Method | Observations/problem | Audit ms/observation | Audit seconds/problem | Denoising calls/observation |
|---|---:|---:|---:|---:|
| Swaps | 26.36 | 181.83 | 4.793 | 10.58 |
| Residual | 29.11 | 52.71 | 1.534 | 10.05 |

These module times include launch gaps. Current costs come from duplicated
QK/normalization, full-support block PV, and (for swaps) pair scoring and
combined admission. At120K the numerator alone is491.52MB per head/query tile.
The selector launches404 kernels; measured GPU activity in pair scoring is
about1.30ms per unit. The former26 scalar reads and10 nonzero calls are already
removed and do not explain the current cost. No measured-request JIT warnings
occur in the completed full-panel clean shards or the crossed diagnostic.

A narrowly scoped fixed-shape graph prototype cuts warmed selection from
4.074→2.176ms at120K and5.467→2.160ms at40K. It matches masks, pair losses,
swap counts and budgets on64 saved units. No production kernel is promoted.
Moment construction, input-buffer management, shape/capture preparation and
native integration remain outside that speedup. Profiler activity is separated
from annotation spans in [cost_analysis/attempt001/summary.json](cost_analysis/attempt001/summary.json).

At observed amortized costs, break-even would require39.4% fewer swap calls or
15.2% fewer residual calls; current calls increase15.6%/21.1%. Granting the old
59-call saving at current V31 amortized cost allows only70.25ms of extra work
per old observation. Current swap construction costs182ms. Even the favorable
two-question clean seed1 diagnostic saves37.1% calls but takes10.31s/request
versus9.37s V31. Fewer calls have therefore failed to yield speed in actual
optimized execution, not just in a cost projection.

Removable costs include launch gaps, redundant materialization/gathers and
workspace allocation. Some full-support information and objective work remain
necessary for the current mathematics. Existing native dense outputs/LSE and
retained/candidate-only PV could remove much of the duplication, but FA4/BF16
and reconstructed FP32 targets are not numerically interchangeable. Streaming
omitted aggregates or changing reduction order also needs fresh qualification.
Overlap must preserve input-buffer lifetime and finish before the selected map
is consumed; there is no evidence that it can hide the entire observation.

## Go/no-go and smallest next experiments

**Full-value swaps: no-go for the current full503 continuation.** Preserve the
implementation as a diagnostic, and avoid a broad tuning or kernel campaign.
It has the stronger historical trajectory signal of the two methods, but
neither consistent accuracy gain nor sufficient amortization. The smallest
engineering result that would justify revisiting it is a source-qualified
observation implementation below roughly70ms on the existing representative
states, including all moment/selection/list work. That threshold is an
optimistic cross-study gate, not a promised speedup. Only after that gate should
a predeclared small multi-seed panel test unconditional accuracy and request
latency, retain tails/bookends, and use the same fixed budgets. A positive
two-case or trimmed result does not justify full503.

**Directional residual: no-go in its current form; redesign the decision
objective before further accuracy runs.** It has strong operator-fidelity
evidence but weak or negative native evidence, and the trace shows how it can
delay convergence and worsen the answer. A cheap native-anchor construction
could remove duplicate full PV, but cost removal alone does not fix the observed
trajectory. The smallest useful follow-up is a predeclared same-state/native
decision screen on early reused calls that tracks self-conditioning, actual
acceptance/stability, and later commit behavior; compare a new numerical target
against both V31 and the unchanged guard. An entropy decrease or output-error
decrease alone must not be its admission criterion. No such redesigned method
has been evaluated here, and the intermittent execution discrepancy must remain
visible rather than being filtered away.

The recommended immediate action is to stop both current full-evaluation arms
and retain the pause. Full503 still requires new explicit authorization;
there is no automatic promotion or full503 resume.

## Integrity and accounting

Four CPU tests pass. Independent analysis binds1696 retained trials and verifies
official scores, frozen input hashes, derived RNG seeds and paired settings.
The new cross has96 measured requests; the trace adds4. There are25 native
warmups including implicit warmups. Cross and trace receipts verify fixed
budgets, scope, native counts and zero measured graph captures. Private payloads
remain private. Two failed profiler harness attempts are retained.

Successful profile, native cross and trace account for2516.79s of measured
diagnostic reservation; including conservative upper bounds for failed profiler
attempts and launch overhead gives2732.85s, or45.55 minutes, below the60-minute
target. Generation/request timings exclude those startup/profile reservations.
The final completion marker belongs to this investigation only. The parent
full503 completion marker remains absent, and the H100 is idle.
