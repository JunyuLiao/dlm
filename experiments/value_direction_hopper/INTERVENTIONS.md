# Trajectory-preserving intervention study

Canonical implementation: `intervention_study.py`, `intervention_policies.py`,
`intervention_study_report.py`. Output: `results/value_direction_interventions_v2`.
Existing `value_direction_sparsity_steps_v2` is the frozen baseline; it is not
rerun. Old `intervention.py` and smoke directories are preliminary, superseded
artifacts: their reactive hook never fired and dense-first counted from the
zero-based router iteration. Do not use them for conclusions.

The current controller wraps the existing native denoising step and sampler.
It counts steps from one, leaves native sampling draws/temperature/stopping
unchanged except for each explicitly labelled acceptance intervention, and
selects the existing Gaussian32 threshold or all-retained attention before a
step. No attention kernel mathematics or code is changed. An all-retained
Hopper call implements dense corrections with matched sparse-mask/numeric
semantics, not the original native dense backend.

Independent configurations:

- Dense-first:30/40/50/60/70 overall sparsity.
- Dense-late (dense starting step4):40/50/60/70 overall sparsity.
- Reactive:60/65/70; one completed top1 transition; OR of high churn, poor entropy
  progress or stalled/regressed acceptance; no consecutive dense corrections.
- Protection:60/65/70; previous top1 unchanged, probability>=.999 and margin>=.98;
  reopen on top1 change or probability<.98; use protected top1 as accepted input.
  All queries are still computed. No recomputation-saving claim is made.
- Adaptive entropy-budget acceptance: strength1 or3,60/65/70; multiply the native
  entropy bound by1+strength*current native accepted fraction.
- Near-boundary extra acceptance: random/ranked,60/70; same min(4,pool-size)
  quota over32 lowest-entropy rejected positions with probability>=.5.
  The random generator is independent of native sampling. Rollouts can diverge,
  so candidate availability is reported and post-hoc identical-state selection
  diagnostics compare exact equal quotas. Dense-token disagreement is a proxy,
  not token ground truth.

All25 independent designs are calibrated before final evaluation. Dense pilot
statistics on the26 disjoint calibration prompts determine reactive thresholds.
Generation-level calibration sweeps a common offset to the validated local/global
s70 logtau. Full grid[-3,-2,-1,0,1,2,4] then up to3 refinements avoids assuming
monotonic overall sparsity when iteration counts change. Points are cached by
policy+threshold identity across target levels. Select nearest actual overall
sparsity, with2pp tolerance; preserve every point. Off-target results are labelled
outside the tested range, never relabelled as successful target attainment or
proven global infeasibility. Maximum observed pilot and conditional per-trajectory
upper bounds are reported. Dense steps remain in the eligible-tile denominator.

The13 development prompts, disjoint from calibration and final, choose at most
two complementary schedule+acceptance combinations and two timed finalists.
All25 independent policies and up to2 combinations evaluate on all130 previously
examined final prompts. No final score selects a threshold or a combination.
Healthy reporting criterion is mean<=6 calls and accuracy within2pp of matched
dense; this is not a statistical equivalence margin.

The smoke evaluates9 policy/control paths on2 calibration prompts. Instrumented
and non-instrumented output IDs/call counts must match. Unit tests verify the
one-based schedules, reactive one-step corrections, protection reopening, and
random/ranked shared-state pools and quotas without altering global RNG state.

Timing: two repeats of all130 examples for native dense, matched kernel dense,
and the2 development-selected interventions. Diagnostic draft scoring, trajectory
retention and routing counts are off. Required controller signal computation
remains part of measured runtime. Calls are grouped by policy, with warmup;
clock/thermal drift is a disclosed limitation. Timing generation parity against
the instrumented final outputs is audited. Attention sparsity is not speedup.

Commands (repository root, ljy_dlm environment):

```bash
PYTHONPATH=src:. python -m experiments.value_direction_hopper.intervention_study launch-smoke --root results/value_direction_interventions_v2
PYTHONPATH=src:. python -m experiments.value_direction_hopper.intervention_study launch --root results/value_direction_interventions_v2
PYTHONPATH=src:. python -m experiments.value_direction_hopper.intervention_study report --root results/value_direction_interventions_v2
```

One lock-protected GPU worker; per-sample resumable shards and atomic routing
files. Sources/binaries are archived and hashed. Calibration/dev/final are separate.
Recoverable per-configuration failures are saved; illegal-memory/assert errors
abort rather than reusing a poisoned GPU context. Health watcher reads status
every900seconds. Report regeneration never loads the model or changes baseline.

The referenced LoSA/Streaming-dLLM concepts motivate variants, not claims of
faithful paper reproduction. Their web content could not be retrieved in this
session; this implementation follows the user's explicit algorithm descriptions
and the locally inspected DiffusionGemma decoder semantics.
