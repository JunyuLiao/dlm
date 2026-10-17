# Completed artifact analysis: partial LongBench-v2 cohorts

This is a completed analysis of partial experiments, not a completed full503
benchmark. The parent queue is paused with17 completed engines,64 matched
questions in both passes, and32 additional audit PIECEWISE questions excluded
from the matched comparison. Native decoding, fixed prefix K128/Q128/KV64,
GLOBAL-layer29 value scope, and dense LOCAL routing are unchanged. V31 is the
inherited Yuhan cross-step-reuse baseline.

The analysis independently applies the official LongBench-v2 extraction and
judging rules to retained private completions. Completion-marker, source,
protocol, public/private input bindings, canvas/sample counters, graph captures,
audit scope and budget checks pass. Previous scores are independently reproduced.
No answer text or token IDs enter this report. Numeric evidence and per-question
descriptive comparisons are in
[summary.json](cpu_analysis/attempt001/summary.json), with source/receipt bindings
in [config.json](cpu_analysis/attempt001/config.json).

## Current matched64, seed42

Clean execution supplies the quality/latency comparison. V31 is the paired
mean of its two control bookends; each separately scores30/64.

| Method | Correct | Accuracy | N/problem | C/problem | N/C | T/problem | S seconds | W seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V31 bookend mean | 30/64 | 46.875% | 240.3125 | 13.1250 | 18.31 | 3229.67 | 5.8686 | 8.9137 |
| Full-value swaps | 29/64 | 45.3125% | 277.8281 | 13.5625 | 20.49 | 3336.42 | 11.1955 | 14.2424 |
| Directional residual | 30/64 | 46.875% | 290.9063 | 14.6719 | 19.83 | 3628.95 | 8.3751 | 11.4195 |
| Dense FULL, mask fix51994 | 30/64 | 46.875% | 252.1250 | 13.4219 | 18.78 | 3298.75 | 7.5873 | 10.6434 |
| Dense PIECEWISE | 30/64 | 46.875% | 252.1250 | 13.4219 | 18.78 | 3298.75 | 7.6435 | 10.6923 |
| All-kept FA4 consumer | 31/64 | 48.4375% | 283.3438 | 14.4063 | 19.67 | 3550.81 | 8.9732 | 12.0232 |

Swaps add15.61% calls and59.78% request latency versus V31; residual adds21.05%
calls and28.11% request latency. Paired question-bootstrap95% intervals for the
accuracy difference are −12.50 to+7.81 percentage points and −7.81 to+7.81pp.
Those intervals do not establish noninferiority. Request-time penalties have
conditional intervals +3.47 to+7.67s and +0.89 to+4.56s. There are only two
engine-shards, so these are not robust population intervals over engine
variation. Both shards separately have positive mean request penalties:
swaps+5.46/+5.20s; residual+3.24/+1.77s.

The clean bookends differ in three output hashes, zero correctness decisions,
and at most88 calls on an item. Audit bookends differ in five hashes and zero
correctness decisions, but one item differs by1155 calls. Audit means therefore
must not substitute for clean trajectories. Clean versus audit changes five
swap hashes without score changes, and two residual hashes with one score
change. Audit residual scores31/64; clean residual remains30/64. Dense FULL,
PIECEWISE and all-kept outputs match between passes on these64 inputs.

## Why the earlier favorable result does not transfer

The previous study uses indices0–63, seed1 and two engine repeats. The current
subset uses two index-strided shards, seed42 and one engine per shard. Model,
manifest, runtime sources, GPU, memory fraction, block/chunk geometry and native
sampler match; execution implementation, schedule, warmup context, cohort and
seed differ. The eight common indices are0,1,16,17,32,33,48,49. Aggregate domain
composition is similar, but current prompts average73,332 tokens versus83,449;
there are29 versus22 short inputs. These are descriptive cohort differences,
not a causal explanation of the reversal.

Old swaps reduced N from251.039 to192.023 (−59.016,−23.51%) and changed accuracy
from45.703% to46.875%. Both engine repeats show call savings (−61.867/−56.164).
Calls fall on41/64 questions, including a38.23-call mean saving among26
always-correct-both questions. This is a real within-study result.

It is also tail-sensitive. Five questions account for89.69% of net savings;
question5 alone accounts for58.56%, with N2549→337 and T16384→3026, both wrong.
Four of the five largest-saving questions are absent from the current cohort.
Removing all five descriptively leaves6.602 fewer calls/problem. The old
residual direction has almost no unconditional call gain (−1.523) and lower
accuracy (43.750% versus45.703%).

Apply the same scrutiny to the current negative result. Swaps still reduce N
on38/64 questions and their median paired change is−7.5 calls. Their worst
regression, question144, explains82.47% of the net call penalty: V31 N436,
T5589; swaps N2416,T16384; residual N2749,T16384; all wrong. That question
explains71.43% of residual's net call penalty. Excluding each method's five
largest regressions produces mean N changes of−25.24/−11.97, respectively,
but request time remains worse by3.36/0.99s. Neither favorable trimming nor
unfavorable trimming replaces the unconditional comparison.

Cohort composition is insufficient: on the same eight questions the old
control/swap means are365.9375/243.0000; current means296.7500/313.6250.
Existing saved-state equality is not a proof of full trajectory equivalence.
The separately frozen two-question seed/backend cross is required to resolve
some of the remaining ambiguity; its results are reported separately.

## What changes in the trajectories

Using the exact symmetric decomposition of N=C×(N/C), current swap's+37.516
calls split into+8.486 from canvas count and+29.029 from calls per canvas.
Residual's+50.594 split into+29.497 and+21.097. These are algebraic terms,
not causal attributions. Old swap's−59.016 splits into−25.376 and−33.640.
Thus the old benefit involved both shorter output and faster within-canvas
convergence; the current regression also involves both, especially within-
canvas convergence for swaps.

Current caps occur in1/64 V31 questions,2/64 swaps,3/64 residual, and2/64 dense
FULL. On the24 current both-correct questions, swaps save17.29 calls but add
2.28s/request. On the26 residual both-correct questions, residual adds7.77
calls and1.07s/request. These post-treatment groups are descriptions only.

The native sampler uses temperature-scaled logits for Gumbel sampling, entropy-
ranked acceptance/re-noising, self-conditioning, confidence and multi-step
argmax stability. Minimizing an observation-time attention error does not
minimize any of those discontinuous decisions, nor answer error. Equal RNG
states can produce different decisions when logits change; later differing
numbers of denoise and commit calls consume different parts of the RNG stream.
This is verified in source; the direction and timing of the present value
interventions' first divergence require native traces.

Prior matched-state LongBench evidence is consistent with that limitation.
Swap's whole-layer post-O error ratio changes from0.9797 at reuse age1 to0.9863
at age9+; the benefit is small and decays. Directional residual improves the
same ratios much more (0.5575→0.7174), but neither implies native convergence
or correctness. Those frames come from two trajectories and are correlated.
On eight same-state decision inputs, swaps improve top1 disagreement on5 items
and worsen2; directional residual improves8, yet its native benchmark quality
is not better. The headwise swap objective also omits cross-head error terms;
the residual guard uses an old dense anchor, not current omitted attention.

## Cost and feasibility

The64-question audit measures26.359 swap observations/problem at181.83ms each
(4.793s/problem), versus29.109 residual observations at52.71ms (1.534s/problem).
There are only10.58/10.05 denoising calls per observation, or9.03/8.51 held calls.
These are module elapsed times including launch gaps, not clean kernel times.
All completed clean logs have zero measured-request JIT warnings. Six allocator
OOM warnings across clean engine logs recovered; they are retained and are not
proof of a particular request's slowdown.

The new saved-state profile and graph prototype are documented in
[cost_analysis/attempt001/summary.json](cost_analysis/attempt001/summary.json).
Actual CUDA kernel/copy categories are separated from profiler annotation
spans, which otherwise double count work. At120K, QK/normalization uses about
0.680ms of GPU activity per head/tile, block-PV0.862ms, pair scoring1.296ms,
and combined admission0.379ms. There are404 selection kernels. Instrumented
selection activity totals2.075ms across an8.022ms span; profiler overhead
prevents interpreting that span as ordinary production latency.

Unprofiled warmed selection takes4.074ms at120K and5.467ms at40K. A fixed-shape
CUDA-graph prototype reduces those to2.176/2.160ms, with zero mask, pair-score,
swap-count or budget mismatches on64 saved head/tile units. It excludes moment
construction, graph capture/input copies and native integration. No production
selector was changed. Failed profiler harness attempts are preserved.

At observed amortized S/N, swaps would need39.40% fewer calls than V31 to break
even; residual would need15.18% fewer. These hold cost mix fixed and are not
controlled per-forward prices. Even subtracting all measured value-observation
elapsed time from the audit leaves generation+0.456/+0.876s above audit V31.
This is an optimistic accounting counterfactual, not a formal lower bound.
Granting the old59.016-call saving at current V31 amortized cost provides only
1.441s/problem, or70.25ms per old observation, for all extra method work.
The current182ms swap observation is far over that allowance; graphing selection
alone leaves substantial moment and pair work.

Reusing existing dense outputs/LSE, calculating only retained/candidate PV,
fusing pair/admission work, and bounded workspaces are plausible engineering
directions. They are not all exact drop-in changes: native FA4/BF16 outputs and
tile normalizers differ from the present reconstructed FP32 target, and new
reduction orders can change admission decisions. Graph reuse also needs stable
buffers/shapes and pre-warm capture, with no new captures during timed requests.
Overlap has dependencies: selection must finish before its map is consumed,
and Q/K/V buffers must remain valid. None of these optimizations establishes
a positive native accuracy/latency tradeoff on its own.
