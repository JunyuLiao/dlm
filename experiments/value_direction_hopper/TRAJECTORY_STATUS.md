# Diagnostic handoff (2026-09-20, 16:37 UTC)

Active user task: explain increased DiffusionGemma denoising iterations, run
matched adaptive trajectories and exact512 controls, finish audit/report.
Do not optimize kernels or alter routing. Do not stop the active worker.

- Canonical root: `results/value_direction_trajectory_v2`.
- GPU worker PID1142449, detached and protected by `worker.lock`.
- Log: `worker_20260920T144202Z.log` in the canonical root.
- Health watcher tool session94761 (restarted after user interruption), emitting updates every900 seconds to
  stdout and `health.jsonl`. Worker continues independently of the watcher.
- All650 adaptive runs completed: 130 each native_dense, kernel_dense,
  blasst_capped, kernel_gaussian32 and kernel_blasst.
- All130 native, Gaussian32 and calibrated BLASST outputs/steps exactly
  reproduce archived results: 520,2123,3619 steps, respectively.
- Unpruned matched-mask kernel:563 steps,90.6923% accuracy. Capped BLASST:
  546 steps,88.8975% accuracy,46.0371% physical sparsity.
- Fixed512 is ongoing:84/390 at16:36UTC. Three methods x130 examples:
  kernel_dense, kernel_gaussian32, kernel_blasst. Every completed run has
  exactly512 calls. All79 most recently audited adaptive/fixed first states match exactly.
- Detached finalizer PID1151140 is queued; see finalizer_launch.json and
  finalizer_20260920T164158Z.log. The earlier waiting helper1150095 was replaced
  without touching GPU worker1142449. It waits for all1040 raw generation shards,
  validates IDs/settings/hash/512 counts, then replaces the collector's obsolete
  cached CPU report stage (only after ALL GPU generations complete). This avoids
  an O(steps^2) analysis bottleneck fixed in the fresh reporter. It does NOT
  interrupt generation. It then runs the fixed48 supplement and fresh report.
- Fixed48 supplement: deterministic first2/task,26 examples x3 methods. Uses
  the original48-step temperature schedule and only suppresses stopping. Code
  trajectory_fixed48.py, root results/value_direction_trajectory_fixed48_v1.
  Queued, not yet started. It distinguishes forced continuation from the
  stretched512 temperature schedule. Main512 experiment remains unchanged.
- Initial v1 worker1142094 was stopped to correct diagnostic topk tie handling;
  its outputs are preserved and marked superseded. v2 uses native argmax and
  verifies it each step. Ten smoke checks passed exact noninterference and
  archived parity. Four unit tests pass.

Key implementation: trajectory.py scopes observation to one model instance;
trajectory_experiment.py reuses the existing adapter, request, routing binding,
kernel binary873c3073e54a37b8, policies and manifests. Fixed512 records the real
stopper but suppresses its result. It stretches the native temperature schedule
from48 to512; this is a documented additional cross-regime change.

Native decoding does not permanently commit masked positions. It initializes
random vocabulary tokens, reselects acceptance each step under an entropy
budget, renoises rejected positions, and stops when all256 argmax tokens are
stable AND mean native-dtype entropy<0.005. Acceptance count is not an exit
condition. The final output is argmax. Confidence recordings use FP32 softmax;
entropy uses native dtype, recorded in each step. Reanalysis must reconstruct
the native-dtype mean AND scalar comparison (BF16 threshold rounding matters).

Report code continues to improve without changing the frozen worker sources.
The worker imports reporting after the adaptive phase and caches it, so after
all390 fixed runs complete, run the reporter in a FRESH process to use the
latest trajectory_report.py / trajectory_interpret.py. Do not modify any
worker source whose hash is frozen in configuration.json.
The finalizer performs this fresh analysis automatically and archives sources.
The reporter appends the same-schedule supplement deterministically via
supplement_reference.json, so fresh regeneration includes it automatically.

Clean archived timing fit (wall=intercept+slope*steps): dense slope71.129ms,
Gaussian32 slope76.946ms, aggressive BLASST70.360ms. Wall/call averages were
misleading due to fixed prefix overhead. With the sparse fitted slope, extra
iterations account for123.35s of125.03s slowdown (~98.6%); descriptive accounting,
not a causal timing intervention. Do not claim sparse decoder steps are cheaper.

Adaptive Gaussian32 vs unpruned kernel first step:11.379% top1 disagreement,
mean confidence lower by5.078pp. At aligned steps,66683 dense-accepted but
sparse-rejected positions still predict the same token,7408 differ. These are
repeated position observations, not independent examples or permanent commits.
Mean repeated unstable positions40.4/256; top8 account for~53.7% of changes;
36.2% of changes occur beyond current output end. Sampled operator error has
negative cross-prompt correlation with confidence loss/steps; don't assert
the desired positive correlation. Final fixed accuracy is still unknown.

Remaining work:
1. Continue approximately15-minute checks (watcher). Investigate failures/stalls.
2. Watch disk (3.6GiB+ free at last check); never delete prior results.
3. Finish all fixed runs, then fresh report regeneration and audit. Check
   all1040 shards, trace hashes, identical settings, exact512, native stop
   reconstruction, final draft scoring, archived reproduction, first-state
   equality. Verify reproducible report generation.
4. Inspect final paired accuracy CIs and draft curves. Add concrete measured
   answers to all eleven questions; do not infer equivalence merely from a CI
   containing zero, or recommend safe early stopping from confidence alone.
5. Inspect plots and give concise final outcome with report link.

The original 390 Hopper evaluation and all kernels remain untouched. Existing
unrelated adaptive changes in the worktree remain untouched. No git commit/push.
