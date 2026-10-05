# RULER4K sparsity / iterations and four-step controls

The canonical run is `results/value_direction_sparsity_steps_v2` (results are
symlinked onto the new disk). The failed disk-full legacy40 run and failed
pre-migration-path startup remain preserved. No old result is overwritten.

The study reuses the validated Hopper kernel and the trajectory infrastructure.
Frozen policies at50/60/65/70% are retained. The missing40% policy is calibrated
independently for local/global attention on the existing26 disjoint calibration
prompts, with2pp tolerance and a12-point limit. Calibration never selects using
accuracy. Above-one BLASST thresholds are allowed for this explicitly aggressive
sweep, including40%, to measure each attention type at comparable sparsity.

All130 RULER4K prompts (13 tasks x10) use the original budgets, seed42, BF16,
model revision,128x64 physical tiles and256-token canvas. Gaussian projection
rank32/seed1729 are unchanged. Adaptive generation uses the native48-step
maximum and native entropy/stability stopping. New40/50/60/65 sparse generations
use the same binary as the prior70 diagnostic; we import70 and both dense
controls after exact migration/instrumentation parity on two prompts.

The extra65 target is a resolution point to distinguish60-65 from65-70, not a
new method or a policy tuned on final scores. The four requested targets remain
the main grid. The plots use achieved physical sparsity from summed counts.

`fixed4` performs exactly4 denoising calls for each problem, including continued
sampling if the native stopper would have fired earlier. It sets the native
maximum to4 and suppresses only the stopper return. Temperature progression is
therefore0.8,0.7,0.6,0.5, rather than the first four temperatures of the original
48-step schedule. This is explicit in the report; methods are matched within
each regime, and adaptive/fixed differences include compressed annealing.
Both native dense and the unpruned matched-mask kernel dense control run fixed4;
Gaussian32 and aggressive BLASST each run at50/70 with their frozen thresholds.

Draft observation wraps the native `_denoising_step` without modifying its
sampling or numerical operations. It stores draft token IDs and official draft
scores, direct call counts, and would-stop flags. No vocabulary-sized tensors
or shared-QKV calibration caches are saved. The smoke test requires exact
instrumented/uninstrumented outputs and counts plus archived70/dense parity.

Every prompt has one256-token canvas (the returned output budgets are30-128).
Thus observed calls/problem equal calls/canvas. There is no claim about how the
curve changes across multiple successive canvases, which this cohort does not
exercise. The48-step adaptive cap can censor true convergence requirements.

Run in the `ljy_dlm` environment from the repository root:

```bash
PYTHONPATH=src:. python -m experiments.value_direction_hopper.sparsity_steps launch --root results/value_direction_sparsity_steps_v2
PYTHONPATH=src:. python -m experiments.value_direction_hopper.sparsity_steps report --root results/value_direction_sparsity_steps_v2
```

One detached lock-protected GPU worker saves per-sample checkpoints and continues
independent configurations after a recoverable error. Sources/binaries are
hashed and archived. Repeated launch of a running worker is rejected. Read its
status and log approximately every15 minutes during long execution.

Outputs: report.md, summary.csv/json, per_canvas.csv, per_task.csv,
per_step_scores.csv, fixed4_drafts.csv, fixed4_transitions.csv, knee_intervals.csv,
paired comparisons, raw generations/routing counts, calibration traces,
frozen policies, source snapshots and plots (PNG/PDF).

The migration-only loader change resolves aliases in the bridge provenance
manifest before comparing the unchanged kernel digest. A matching filename
alone never satisfies the check. No GPU arithmetic or kernel schedule changes.
