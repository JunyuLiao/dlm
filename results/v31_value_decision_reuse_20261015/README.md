# Fixed-budget value direction and reuse study

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
