# Native repeatability and refresh attribution: completed bounded follow-up

**The sampled native operator is repeatable, and refresh-phase attribution is now
qualified. The remaining obstacle to a new cheap-refresh selector is insufficient
request-level headroom in these measured batch-one settings.** No selector was
changed or trialed. The GPU is idle; all reservations are closed.

The study used 1334.421052 reservation seconds, **22.24 of the authorized 45
minutes**, including startup, warmups, trace export and the preserved partial
replay. This is diagnostic evidence on selected inputs, not a benchmark accuracy
claim, a measured optimization speedup, or proof of global determinism.

## Protocol, controls and completion

The frozen plan is `docs/REFRESH_REPEATABILITY_PLAN_20261018.md`. Native code is
original source `6aab8632`; capture/clean/NVTX wrappers are source `62ae618b`.
Corrected operator bootstrap is `eaf69a67`, followed by CPU-analysis recovery
source `9d864986`. Every execution source, input schedule, model/runtime binding
and copied-cache context is retained in the corresponding protocol and receipts.
Original sources, inputs and cache inventory verified unchanged after execution.

All native requests preserve the official sampler, output limits and fixed
K8192, Q128/KV64, dense LOCAL/prefill/commit, carried maps, progress threshold 0.15
and sticky 1.386. The fixed16 control changes only the refresh clock. This remains
distinct from the Slack stickiness-off comparison. Yuhan's inherited mass-based
selection and cross-step reuse are the production reference.

| Phase | Allocation | Charged reservation |
|---|---:|---:|
| Native state capture |900s|234.227s|
| Isolated replay, including preserved partial attempt |600s|385.970s|
| Clean control and external attribution trace |900s|567.846s|
| Explicit CPU-analysis recovery, second replacement process |300s|146.378s|
| Total |2700s|1334.421s|

Completed coverage:

- Six native refresh snapshots: LongBench 125/seed1, 329/seed42 and 254/seed42,
  twice each; one implicit and one explicit warmup. These are three unique
  observed tensor states, not six independent task examples.
- Two independent successful operator processes: six snapshots × 16 automatic-
  split replays × 2 processes = **192 measured operator calls**, plus 24 warmups.
  Each process uses eight fixed-buffer and eight fresh-allocation repetitions
  per snapshot. No conditional split=1 run was needed.
- Eight clean measured requests and eight annotated measured requests: the exact
  original V31/fixed16 schedule on indices 339/254, two repetitions, with the
  original implicit/two explicit warmups in each engine.

The frozen plan accidentally described capture index 329 as an old runtime input.
It is a separate 36,314-token input; runtime index 339 has 41,189 tokens. The explicit
frozen capture indices were executed as written, and the runtime schedule was
copied unchanged. Read `attempt001/protocol_description_correction.json`; do not
compare 329 versus 339 as matched trajectories. No extra request was added to fix
this descriptive error.

Two original replay leaves were skipped before launch because the standalone
entrypoint omitted the native FA4 local-window bootstrap. The correction applies
the same patch before observer import and verifies module/class/source identity.
Its first process was then stopped because CPU error reductions dominated
execution. Final artifact review found three complete per-snapshot reports (48
repeats); the interruption note recorded two at the stop decision. The original
note is preserved alongside a separate count correction. It used 238.743s,
retained as a partial attempt and excluded from the qualified 192-call pair.
Replacement code changes only CPU analysis: exact-byte equality skips
unnecessary error reductions, all-active partial extraction avoids boolean
indexing, and diagnostic CPU threads are one. Nonidentical tensors use the
original metrics. Tests establish equivalent results, including NaNs, signed
zero, actual mismatches and inactive slots. Native GPU calls and repetitions are
unchanged. The replacement pair uses the existing replay/recovery allocations;
there is no automatic borrowing or increase in the total cap.

## Numerical result: a qualified sample pass

All 192 successful replay calls match the native captures and their repeated
references **exactly** for:

- BF16 dense attention output;
- FP32 observation statistics z;
- active FP32 SplitKV partial outputs;
- partial LSE and the active-partial mask.

Maximum observed absolute difference is 0. Input hashes and RNG checks pass.
Original shape, dtype, strides, storage offset and 256-byte alignment are preserved
under both allocation policies. Recorded native configurations and compile keys
match the capture in both processes. All six states resolve to 16 splits,
64×64 tiles, pack-GQA off, one Q stage, and the same native forward/combine path.
The loaded patched SM90 source hash matches the native engine. Compile-object
IDs are process-local receipts, not a cross-process machine-code identity claim.

The 70K state is at adapter canvas ID 5/call 9, GLOBAL layer 5, prefix 70,815—the same
recorded event geometry as the earlier control-drift example. However, its Q/K/V
bytes differ from both historical controls, even after conversion to the old
adapter layout. The two new captures match each other exactly. Adapter canvas
IDs also advance through encoder/prefill calls; ID 5 does not mean the fifth
output canvas. All six snapshots are the first genuine progress refresh, at
prefixes equal to their input lengths.

**This passes the bounded same-state gate. It does not explain the old full-engine
control divergence.** No defect was observed on these three unique first-refresh
states. It does not establish repeatability for every later state, arbitrary
request history, concurrent execution or another kernel configuration. Returning
the original output unchanged during capture does not make the instrumented
native trajectory a clone of the old audited engine. No accuracy conclusion is
based on these snapshot-generating requests.

## Runtime attribution: qualified without changing trajectories

All eight measured requests in the new clean run reproduce the original clean
outputs, N/C/T, finish reasons and native counters exactly. All eight NVTX-traced
requests reproduce both clean references exactly. There are zero measured CUDA
graph captures and no monitored JIT warnings after the explicit runtime warmups.

The external trace uniquely qualifies request/prefill/decoding boundaries with
no crossing GPU activity. All 106,440 recorded scopes, including warmups, are
present. Phase counts reconcile with native initial/refresh/held/carried/bootstrap
counters and all 30 attention layers across prefill, denoising and commit. Every
measured attention scope has associated GPU activity. Launch correlation has zero
ambiguities; work is attributed through CUDA APIs and enclosing ranges, not
kernel names. Unlinked model work remains explicit.

An independent CPU implementation uses only nonoverlapping attention-parent
ranges and separately nonoverlapping buffer ranges. It reproduces all 165 grouped
CPU durations, GPU activity sums, call counts and activity counts exactly. It does
not reuse the primary analyzer's bounded-depth child search. Nested observation,
selection and rank costs are never added to their inclusive attention parent.

Clean native metrics, means of two repetitions:

| Input / arm | W seconds | S seconds | N | C | N/C | S/N ms | T | Official score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|41K V31|8.626675|7.535925|325|16|20.313|23.187|3965|1|
|41K fixed16|4.433495|3.343170|146|8|18.250|22.898|2024|0|
|112K V31|10.909985|6.014515|238|11|21.636|25.271|2622|0|
|112K fixed16|11.626005|6.721700|279|13|21.462|24.092|3170|0|

Scores are inherited from the verified original scored records through exact
input/output parity, not a fresh accuracy panel. Fixed16's short-input speed
advantage accompanies an incorrect answer. These two inputs cannot establish a
benchmark ranking. S/N is an amortized generation cost per denoising call, not an
isolated forward price. Audit flags remain off in runtime workers, so this study
adds phase/counter attribution rather than a new instrumented physical-tile
sparsity claim. The selector and full native adapter counters are unchanged.

## Headroom and the go/no-go decision

The following is a deliberately generous **fixed-trajectory deletion screen**:
add inclusive parent CPU elapsed time to associated GPU activity duration, divide
by clean W, and assume a perfect free replacement leaves later work unchanged.
CPU/GPU overlap is deliberately counted twice. These are not measured savings,
additive critical-path fractions or universal bounds on algorithms that change
the generation trajectory.

| Input / arm | Progress refresh only | All dense GLOBAL decode phases | Those phases + held/carried buffer upkeep |
|---|---:|---:|---:|
|41K V31|2.970%|8.136%|10.040%|
|112K V31|2.437%|7.924%|9.130%|
|41K fixed16|2.144%|7.211%|8.891%|
|112K fixed16|1.884%|7.917%|9.251%|

“All dense GLOBAL decode phases” includes bootstrap, initial selection, progress
refresh and GLOBAL commit attention; it excludes prompt prefill and LOCAL
attention. The final column also removes buffer upkeep on held/carried steps.
Those buffer costs are associated with their parent phases and counted only once;
buffer work already inside a dense parent is not added again. This supplementary
scope matters because a redesigned observer might avoid maintaining contiguous
K/V buffers even between observations. It was inspected as a post-hoc scope
extension and does not change the frozen progress/dense-phase gates.

For dynamic V31, progress-refresh CPU+GPU totals are 0.256/0.266s at 41K/112K.
The broader dense-phase totals are 0.702/0.864s. Additional held/carried buffer
upkeep contributes 0.164/0.132s. The 41K extended ceiling only exceeds 10% by about
**3.5ms per request**, before replacement attention, mask construction or any
other replacement cost. That is not credible evidence for a ≥10% net benefit.
The 112K extended ceiling is already below 10%.

The decisions are therefore:

- **Numerical sample gate: pass.** Preserve the limitation that the historical
  engine drift remains unlocalized.
- **Attribution gate: pass.** Clean trajectory transfer, scope counts, CUDA
  correlation and independent arithmetic checks qualify these phase measurements.
- **Progress-refresh-only optimization: no-go for the ≥10% target on these
  requests.** Its gross ceiling is only about 2–3%, even with free replacement.
- **Another cheap-refresh selector in these measured settings: no-go for now.**
  No plausible ≥10% net request improvement is established after replacement
  cost, even under the much broader optimistic screen. This is a scope/regime
  decision, not proof that sparse attention or better selectors are universally
  unhelpful. Changed trajectories, longer outputs or different serving regimes
  require their own measurement.

## Next research direction

Keep inherited V31 and fixed16 as controls. Do not spend the next campaign on a
new value score or another refresh trigger in the same measured configuration.
First qualify an execution or workload opportunity with enough end-to-end margin.

In the current engine, repeated held-sparse calls have much larger aggregate
host-side spans than progress refresh. On 41K V31, held GLOBAL attention accounts
for about 0.852s of inclusive CPU spans and 0.296s of GPU activity, versus about
0.145s CPU and 0.111s GPU for progress refresh. These spans can overlap other
work; they are not all removable host overhead. They identify repeated sparse
execution/dispatch and buffer maintenance as a better place for a small systems
prototype than another selection formula. A prototype still needs exact native
output/trajectory parity and clean end-to-end validation.

For an algorithmic sparse-attention contribution, first profile a feasible
concurrency/long-output setting or a model with a larger measured attention
share. Longer context alone is insufficient: on the 112K dynamic request,
prefill consumes about 44.9% of clean W. Do not assume larger batch or smaller
stride helps; test the actual serving configuration and memory feasibility.
Only after the ≥10% net opportunity is demonstrated should useful future-support
maintenance become a selector research target. The previous factorial's support
signal remains provisional; this study does not turn it into a novel method or
a paper-level accuracy/throughput result.

## Evidence and reproducibility

- `attempt001/{protocol.json,protocol_description_correction.json}` and
  `attempt001/{capture,clean,nvtx}/receipt.json`.
- `attempt001/capture/{probe.json,historical_hash_comparison.json}`; private
  snapshots are hash-bound by the corrected replay protocol.
- `replay_corrected/attempt001/`: preserved partial replay and interruption note.
- `replay_corrected/attempt002/`: replacement protocol, bootstrap receipts,
  two complete operator reports and per-snapshot metrics.
- `attempt001/nvtx/{parity_original.json,parity_new_clean.json,attribution.json}`.
- This directory's `summary.json`, `analyze.py`, `kv_buffer_attribution.json`,
  `kv_buffer_attribution.py`, `independent_attribution_verification.json`, and
  `verify_attribution.py` reproduce the arithmetic and retain all cases.
- Campaign `gpu_budget.json` and this directory's GPU-idle receipt.

Twenty-nine relevant CPU tests passed across the diagnostic, external-trace and
replay-recovery suites. Execution, source, snapshot and original-cache checks
passed. Raw tensors, prompts, completions and profiler traces remain private.
No further GPU work or selector trial is queued. The older full503 campaign
remains paused at 17/128 engine jobs.
