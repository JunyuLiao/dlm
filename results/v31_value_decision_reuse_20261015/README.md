# Fixed-budget value direction and reuse study

Current status,02:07 UTC October16: the new replicated native evaluation is
frozen under `native_replicated_20261016_attempt001/`. It uses two engines per
cell, reversed candidate order and both control bookends. Every structurally
valid outcome counts, including drift. Workload qualifications precede accuracy.
See its README and `docs/V31_REPLICATED_NATIVE_PLAN_20261016.md`.

The completed control-only history and ABBA warmup factor show drift without
value intervention and despite either warmup policy. All six drifting factor
bookends concern long AIME item10 (4/18 control-warmup,2/18 method-warmup).
No numerical cause or repair is established. This motivates replication and
explicit uncertainty, not exclusion of inconvenient outputs.

The older independent-process and exact-parity native panels remain held under
their original protocols. Same-state diagnostics,71 H100 method tests and15
coordinator/analysis checks are completed scoped evidence. No native accuracy
gain or speedup has been established. Clean timing and other campaigns remain
paused. Historical notes below are not current launch instructions.

At09:38 UTC the replicated AIME workload completed: V3151.111%/295.300
calls; full swaps52.222%/301.672; guarded residual51.111%/296.911; raw
residual50.556%/296.861; combined51.111%/298.611. All four fail the joint
AIME screen; the swap gain changes sign between engine replicates and its
problem-cluster interval includes zero. See the new study's reports/aime26
and independent verification. LongBench is running, then MATH and RULER.
This complete workload does not complete the four-workload study.

Authorized scope: three independently implemented directions, same-state
diagnostics before task accuracy, fixed per-query-tile K, and clean timing paused.
The inherited Yuhan V31 machinery supplies the reference maps and native sampler.
The new operators are in `experiments/numerical_qk_reuse/v31_value_decision.py`.

## Plan and evidence

The detailed design is `docs/V31_VALUE_DECISION_REUSE_PLAN_20261015.md`.
The refined causal residual guard and its limits are documented in
`docs/V31_CAUSAL_VALUE_RESIDUAL_GUARD_20261015.md`.

Completed scoped stages:

* `operator_complete.json`: independent head/tile screen, verified immutable inputs.
* `whole_layer/operator_complete.json`: all 16 heads, both Q128 tiles, signed
  errors through the real O projection, 570 usable saved frames.
* `reuse_guard/operator_complete.json`: causal directional/norm trust guards,
  same 570 frames; see `reuse_guard/report.md` and `layer_analysis.json`.

These are operator diagnostics on four previously examined trajectories.
They do not establish task accuracy, native call savings, or speedup. K=27 AIME
replay is counterfactual on states originally generated with K=128. Sparse
AIME evidence comes from one late trajectory; early all-kept states are excluded
from benefit summaries. No full-study completion marker exists yet.

`online_swap8` is a hybrid: full-V online preserve-mass risks rank proposals,
then a full-information joint-error guard accepts bounded fixed-K swaps.
It is not the historical rank32 V2 selector or a purely online implementation.
`full_swap8` uses mass-qualified candidates and true final-output error.
Correction changes the output consumer and is reported separately from selection.

## Completed instrumentation qualification and active follow-up

`suffix_qualification_attempt006/` passed: all six eager trajectories have
identical output hashes,117 native calls,6 canvases and1438 output tokens.
Both active repeats have12 frames/120 offline suffix calls, zero parity
failures, and unchanged RNG/counters. See `report/summary.md`. These eager
states are separate from compiled production accuracy. Failed attempts001-005
remain immutable and excluded from method comparisons.

`decision_batch_protocol.json` freezes paired uninstrumented/active diagnostics
on8 LongBench,6 AIME,8 MATH and13 RULER items, independently selected by index
and task. Each item must preserve its native trajectory; all-kept frames are
excluded from sparse-benefit summaries. Fixed per-query-tile K, native sampler,
and timing pause remain in force. The batch adds a C_gate row-permutation
control and full-swap plus directional-correction arm. Follow
`decision_batch_state.json` and `decision_batch_coordinator_attempt001.log`.

Resume the batch from the repository root with the pinned H100 interpreter:

```
PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python \
  scripts/v31_value_decision_batch.py --stage run
```

Each workload executes its own frozen coordinator and source snapshot. Completed
leaves are verified and skipped; failed/partial leaves receive a new attempt.
Source changes require a new frozen attempt. Check long jobs every30 minutes.

Native refinement and the native development coordinator are implemented but
remain gated. They require a completed multi-workload decision report,
explicit source-bound promotion, H100 correctness checks and native execution
smokes. C_gate and combined swap/correction integration have CPU lifecycle
checks; the expanded GPU suite must run before promotion. Native development
will retain fresh V31, FULL fix51994, PIECEWISE dense and all-kept controls,
use official scoring, and separate extra refinement work from physical consumer
work. No native accuracy run, clean timing run or other paused campaign has
been launched by this study.

At the16:46 UTC monitor, AIME is complete and LongBench is active. All six
AIME native trajectories pass invariance. Its35 sparse frames are summarized
in `decisions_aime26_attempt001/report/summary.md`; these are same-state
decision metrics, not official task scores. The native gate now binds H100
test evidence to every refinement source and test file. Updated GPU pair
ranking and the full-V ordered online-risk kernel still await that test run.

The initial four-workload decision batch is now complete, with all35 native
pairs preserving output/N/C/T and exact suffix parity. Item-level results are
in `decision_analysis/summary.md`. MATH has zero sparse frames, so its completed
job cannot pass the sparse-method gate. All70 expanded H100 tests subsequently
passed; see `receipts/h100_qualification/attempt001/receipt.json`.

`coverage_protocol.json` freezes a separate purposive panel: eight unseen long
MATH controls and14 remaining RULER vt/cwe items, selected without correctness.
The source is a6f339fe. It uses the qualified online traversal/pair-ranking
optimizations and the same fixed budgets. Resume with:

```
PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python \
  scripts/v31_value_coverage_batch.py --stage run
```

Follow `coverage_state.json`; inspect about every30 minutes. Native accuracy
and timing remain gated/paused respectively.

Coverage is complete. See `coverage_complete.json`, `coverage_analysis.json`
and the concrete inclusion decision `native_inclusion.md`. The native panel
contains full_swap8, residual_direction_guard, raw residual as the cache
ablation, and full_swap8_direction_guard, alongside four fresh controls.

`native_development_attempt002/protocol.json` is frozen from79c994ca, with
71 passing exact-source H100 tests. Preparation001 remains unlaunched and
superseded after correcting inherited sampler metadata labels. Actual pinned
defaults are confidence_threshold0.005,entropy_bound0.1 and48 denoising steps;
thinking/output limits remain per-input. Native operators are unchanged by
that metadata correction. Inspect `native_development_attempt002/state.json`.

Run/resume the frozen native coordinator (not a later mutable root copy):

```
study=/home/exouser/ljy/dlm/results/v31_value_decision_reuse_20261015/native_development_attempt002
PYTHONPATH="$study/private/execution/src:$study/private/execution" \
  /home/exouser/dyh/v31_env_20261005/bin/python \
  "$study/private/execution/scripts/v31_value_native_study.py" \
  --stage run --study "$study"
```

The coordinator must complete sparse execution/official-scoring smokes before
the development panels. All completed attempts are verified before reuse;
failed/partial workers receive new attempt directories. Clean timing stays
paused, and full503 remains gated on completed native development results.
