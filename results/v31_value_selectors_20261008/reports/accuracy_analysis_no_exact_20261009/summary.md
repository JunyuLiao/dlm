# Accuracy analysis — value selectors under V31 cross-step reuse

Date: 2026-10-09 UTC. This report analyzes the completed official audit, not the
paused clean-timing stage. The audit covers 503 LongBench-v2 `0shot_think`
items, seed 1, for each of the eight authorized arms. Exact greedy is excluded
by user instruction; its completed diagnostic is retained separately.

## Finding

The value-aware arms do not show an accuracy gain over the V31 attention-mass
control in this protocol. V2 is the closest result, at 266/503 versus
268/503 (`-0.40` percentage points; paired cluster interval `[-3.78,+3.18]`
points), but this is a tie-like result, not evidence of an improvement. V1,
V3a, and batch8 are lower by 4.17, 2.98, and 5.17 points respectively.

| arm | correct/503 | accuracy | delta vs control | GLOBAL sparsity | N |
|---|---:|---:|---:|---:|---:|
| `current_v31_control` | 268 | 53.3% | 0.00 pp | 83.16% | 140,693 |
| `value_v1_online_discard_mass` | 247 | 49.1% | -4.17 pp | 80.04% | 120,301 |
| `value_v2_online_preserve_mass` | 266 | 52.9% | -0.40 pp | 84.11% | 126,072 |
| `value_v3a_singleton_delete` | 253 | 50.3% | -2.98 pp | 83.08% | 137,212 |
| `value_v3b_approx_batch8` | 242 | 48.1% | -5.17 pp | 84.97% | 171,652 |

These are official scores and descriptive paired comparisons. The control's
53.3% should not be read as an intrinsic accuracy improvement: all-kept FA4
scores 50.1% and differs from native FULL on 503/503 output hashes despite
skipping zero tiles. The adaptive sampler and consumer path are part of the
measured behavior.

The length breakdown localizes the larger losses. Control/V1/V2/V3a/batch8
score respectively as follows:

| bin | control | V1 | V2 | V3a | batch8 |
|---|---:|---:|---:|---:|---:|
| Easy | 58.3% | 53.1% | 58.3% | 52.1% | 49.0% |
| Hard | 50.2% | 46.6% | 49.5% | 49.2% | 47.6% |
| Short | 67.2% | 63.3% | 66.7% | 61.7% | 60.6% |
| Medium | 47.0% | 44.7% | 46.0% | 46.5% | 41.9% |
| Long | 42.6% | 34.3% | 43.5% | 38.9% | 39.8% |

V2 is close to the control in every bin, while V1 and batch8 lose most on
long items. That is consistent with reuse/staleness and stopping sensitivity,
although the official run is still a trajectory comparison rather than a
causal length-stratified intervention.

## Why the previous value-aware result does not transfer

The prior positive result was a fresh, current-state value-aware routing
setting. The present experiment changes the optimization problem in four ways:

1. **The value score chooses a map, then the map is held.** V1/V2/V3 compute a
   selector-only score at initial/refresh calls. The production output remains
   the inherited FA4 attention output, and later held/carry calls consume the
   bitmap without reselecting from their current Q/K/V. In the audit there are
   about 500k–755k held GLOBAL calls per value arm, versus only about 32k–35k
   initial selections. A value direction that is useful for the current call
   can be a poor description of later calls; the mass control is a more stable
   support rule for reuse.

2. **The objective is a proxy for a projected attention output, not task
   logits.** The selector uses a rank-32 Gaussian projection of BF16 V and
   tile means. It minimizes a projected PV residual (or an online approximation
   to it), not the post-attention output projection, MLP, denoising confidence,
   or final token decision. A low projected-PV error therefore need not preserve
   the sampler's accepted tokens or stopping decision.

3. **The mask changes the native adaptive trajectory.** Every non-control sparse
   arm differs from the control on all 503 output hashes. Relative to the
   control, V1/V2/V3a/batch8 have token mismatches on 496/498/500/497 items and
   N mismatches on 497/499/497/496 items. Mean denoising-call differences are
   -40.5, -29.1, -6.9, and +61.6 calls respectively. Accuracy is therefore not
   a same-trajectory comparison of two masks; it includes different acceptance,
   re-noising, canvas stopping, and later contexts.

4. **The value masks move too far from the robust mass support.** In the
   sampled offline audit (111 snapshots, 20 sources, layer 5, ages 0/1/7),
   age-0 mask IoU with the mass control is 0.29 for V1, 0.14 for V2, 0.70 for
   V3a, and 0.18 for batch8. The corresponding full-dimensional mean-row
   errors are 0.296, 0.404, 0.253, and 0.218, versus 0.250 for the control.
   V2 also retains only 0.224 mean attention mass at age 0 versus 0.369 for
   the control. These measurements are descriptive sampled snapshots, not a
   causal proof, but they show that the selector is often replacing the
   control's stable support with a substantially different support.

The batch8 result is especially diagnostic: it has the lowest sampled age-0
full-dimensional mean-row error of the value arms, yet it has the worst
official accuracy and the largest positive N shift. The projected/full-attention
error is therefore not a sufficient task-accuracy objective under adaptive
denoising. Its audit selection work is also about 330 billion candidate
comparisons, so it is not a useful accuracy-first direction to continue.

## What the arm pattern says about the implementation

There is no evidence here of invalid rows, an output-path substitution, or a
single obvious kernel failure: the receipts report zero invalid rows, the value
path is selector-only with unchanged FA4 output, and the focused tests pass.
That does not prove every numerical detail is perfect, but the consistent
pattern is more informative than a one-off failure:

- V1's online discard state loses both support overlap and early full-dimensional
  fidelity, and its 80.04% GLOBAL sparsity is not matched to the control.
- V2's denominator-preserving state is the closest official result, but its
  early retained mass and support overlap are the worst. Its near-tie suggests
  that the V signal is not adding a reliable advantage once the map is held.
- V3a uses a full-support deletion objective and nearly matches the control's
  83.16% sparsity (83.08%), yet it is still 15 correct items lower. This rules
  out a simple “V1 only failed because of the online threshold” explanation.
- Batch8 spends much more selection work, changes the support substantially,
  increases denoising calls, and loses accuracy. More exactness in the surrogate
  optimizer does not fix the surrogate/protocol mismatch.

## Accuracy-first next experiment

Do not spend more time on clean timing or broad selector sweeps yet. The next
small panel should isolate map quality from native stopping:

1. Freeze the same Q/K/V snapshots and match the control's per-unit budget.
   Compare mass, V, and hybrid masks at equal kept count, reporting retained
   full-dimensional attention output, post-output-projection/logit change when
   available, argmax changes, and mask IoU.
2. Add a **mass-anchor hybrid**: keep the V31 mass map as a floor and allow V
   information to replace only a small, explicitly bounded fraction of tiles.
   Enforce a minimum retained mass and a high overlap with the anchor. This
   tests whether V provides useful *orthogonal* information without discarding
   the support that survives reuse.
3. Separately test **fresh versus held value selection** on a small accuracy
   panel. If fresh V selection helps while held V selection does not, the
   failure is map staleness; if neither helps on matched snapshots, the rank-32
   projected objective is the problem.
4. For V2, audit the online state against a full-support reference before any
   generation rerun. Its “advance denominator while leaving normalized output
   unchanged” rule is mathematically defined, but the observed low retained
   mass makes it a likely source of unstable map choices under reuse.

Only after one of these tests shows a clear same-trajectory or matched-snapshot
benefit should it return to official generation. The paused timing stage is
resumable from `longbench/target_no_exact/attempt002/timing_stop_20261009.json`.

## Sources

- Official scores and paired intervals:
  `reports/audit_scoring_no_exact_attempt002/summary.md` and
  `diagnostics/complete_audit_summary_no_exact_attempt002.json`.
- Selector-only/output-path receipt and lifecycle:
  `experiments/numerical_qk_reuse/vllm_adapter.py` and the per-arm audit
  receipts under `longbench/target_no_exact/attempt002/`.
- Offline sampled attention diagnostics:
  `diagnostics/offline_accuracy_analysis_20261009/aggregate.json` with
  completion marker `aggregate_complete.json`.
- Control definition and provenance:
  `SOURCE_NOTE.md` and `protocol_execution_attempt005_no_exact_20261008.json`.
