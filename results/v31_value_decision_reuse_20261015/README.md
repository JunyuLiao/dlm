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

## Sequential diagnostic qualification

`suffix_qualification_attempt006/` is the current frozen last-GLOBAL-layer
instrumentation qualification. It compares uninstrumented, observer-disabled,
and observer-active native trajectories twice, checks repeated suffix logits,
and checks RNG and adapter counters. All suffix calls are offline diagnostics.
This stage cannot qualify earlier GLOBAL layers or task accuracy by itself.

Earlier failed attempts are immutable: constructor signature (attempt001),
unsupported fullgraph compiler-disable boundary (attempt002), and observer
allocation incompatible with downstream fixed-address CUDA graphs (attempt003).
Attempt004 preserves graph-owned addresses but still changes compilation and
fails native trajectory/suffix equivalence. Attempt005 therefore uses an explicit
eager diagnostic model, with matched eager controls and the exact parity gate
retained. Its states are not production compiled V31 states. Failed attempts
are excluded from method comparisons. Production accuracy requires fresh compiled
controls after the decision gate. Attempt005 failed at the inherited CLI
mode check before model loading. Attempt006 explicitly overrides the runtime
to eager/mode0/NONE behind the required PIECEWISE CLI label, and validates
the resolved runtime mode in every record.

Resume the current frozen coordinator from the repository root using the pinned
H100 interpreter and the absolute study path:

```
PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python \
  results/v31_value_decision_reuse_20261015/suffix_qualification_attempt006/coordinator_source.py \
  --stage run --study /home/exouser/ljy/dlm/results/v31_value_decision_reuse_20261015/suffix_qualification_attempt006
```

The coordinator verifies source/input hashes, skips validated completed leaves,
and gives failed or partial leaves a new attempt directory. A source change
requires a newly frozen study attempt. Inspect long running experiments at
30-minute intervals. Other paused campaigns remain paused. No native-accuracy
worker may bypass the decision-gate receipt.
