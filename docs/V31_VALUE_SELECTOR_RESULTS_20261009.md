# V31 value-selector panel: methods, results, integration, and reproduction

This document is the reproducibility record for the completed eight-arm
LongBench-v2 panel in `results/v31_value_selectors_20261008/`. The panel uses
DiffusionGemma-26B-A4B, vLLM 0.30.0, Torch 2.13.0+cu130, one H100, native
adaptive denoising, `0shot_think`, seed 1, and the official 503-item manifest.
The generation source is commit `44674c3a`; the continuation protocol is
`protocol_execution_attempt005_no_exact_20261008.json`. The current repository
HEAD at the time of this record is `7cdfadc3`.

The panel measures accuracy and denoising trajectories. Clean timing was
stopped before completion and has no publishable latency result. The exact
Greedy arm was removed from future generation by user instruction; its partial
128-cell diagnostic remains under
`results/v31_value_selectors_20261008/diagnostics/exact_completed_shard01/`.

## Frozen execution contract

All sparse arms use only the five GLOBAL layers (5, 11, 17, 23, 29), with
native dense LOCAL layers, prefill, and canvas commits. Physical GLOBAL tiles
are Q128 x KV64. Canvas boundary tiles are always kept. The native sampler is
unchanged: 256-token canvases, maximum 48 calls, entropy bound 0.1, stability 1,
confidence threshold 0.005, temperature 0.8 to 0.4, and native EOS/stopping.

The inherited V31 reuse lifecycle is shared by the control and value arms:
selection at call 1, first-call carry enabled, settledness trigger 0.15, sticky
factor 1.386, GLOBAL budget 8192 tokens, and no sink/recent tile additions.
The value projection is a fresh rank-32 Gaussian projection of current V at
each initial/refresh selection, seed 1729. Value selection is selector-only;
the executed attention output remains the FA4 path.

### Arm definitions

| Arm | Definition | Selection/reuse behavior |
|---|---|---|
| `dense_full_fix51994` | Native vLLM FULL dense path with the exact DiffusionGemma mask fix corresponding to upstream PR #51994. | No sparse map; native dense trajectory. |
| `dense_piecewise` | Matched PIECEWISE dense reference. | No sparse map; native dense trajectory. |
| `allkept_fa4` | FA4 sparse consumer with every eligible tile kept. | No physical GLOBAL skipping; exposes consumer/path effects. |
| `current_v31_control` | Attention-mass-only `qblock_max`: worst valid query-row prefix mass share, fixed budget, sticky refresh retention. | Inherited cross-step map reuse and carry. No projected V in the score. |
| `value_v1_online_discard_mass` | Online value-output risk using projected tile means; skipped tiles do not advance the running mass denominator. | Uniform frozen threshold 0.027; no fixed quota; inherited reuse. |
| `value_v2_online_preserve_mass` | Same online value-output risk, but skipped tiles advance the running denominator while the normalized output state is unchanged. | Uniform frozen threshold 0.027; no fixed quota; inherited reuse. |
| `value_v3a_singleton_delete` | Full-support projected-PV residual; remove individually safe tiles until the fixed prefix budget is reached. | Support-constrained singleton deletion; inherited reuse. |
| `value_v3b_approx_batch8` | Same full-support deletion objective, removing candidates in batches of eight with cleanup. | Explicit approximation; not exact Greedy; inherited reuse. |

The V1/V2 threshold was accepted from development without exact sparsity
matching. Their achieved sparsities therefore differ from the control by
protocol design.

## Official accuracy and denoising trajectories

`N` is total denoising forwards, `C` is the number of canvases, `T` is total
output tokens, and `N/C` is forwards per canvas. GLOBAL sparsity is physical
eligible-tile skipping; LOCAL sparsity is zero in every arm.

| Arm | Correct / 503 | Accuracy | Delta vs control | GLOBAL sparsity | N | C | T | N/C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_full_fix51994` | 247 | 49.1% | -4.17 pp | 0.00% | 163,345 | 7,627 | 1,888,404 | 21.417 |
| `dense_piecewise` | 247 | 49.1% | -4.17 pp | 0.00% | 163,345 | 7,627 | 1,888,404 | 21.417 |
| `allkept_fa4` | 252 | 50.1% | -3.18 pp | 0.00% | 147,260 | 7,134 | 1,761,430 | 20.642 |
| `current_v31_control` | 268 | 53.3% | 0.00 pp | 83.16% | 140,693 | 6,840 | 1,686,311 | 20.569 |
| `value_v1_online_discard_mass` | 247 | 49.1% | -4.17 pp | 80.04% | 120,301 | 6,388 | 1,568,793 | 18.832 |
| `value_v2_online_preserve_mass` | 266 | 52.9% | -0.40 pp | 84.11% | 126,072 | 6,406 | 1,575,774 | 19.680 |
| `value_v3a_singleton_delete` | 253 | 50.3% | -2.98 pp | 83.08% | 137,212 | 6,758 | 1,664,826 | 20.304 |
| `value_v3b_approx_batch8` | 242 | 48.1% | -5.17 pp | 84.97% | 171,652 | 6,926 | 1,712,374 | 24.784 |

The paired intervals are descriptive question-cluster bootstrap intervals, not
non-inferiority claims. Against the control, V1 is `[-7.55,-0.80]` pp, V2 is
`[-3.78,+3.18]` pp, V3a is `[-6.36,+0.40]` pp, and batch8 is
`[-9.15,-1.39]` pp.

The length breakdown is:

| Bin | Control | V1 | V2 | V3a | Batch8 |
|---|---:|---:|---:|---:|---:|
| Easy | 58.3% | 53.1% | 58.3% | 52.1% | 49.0% |
| Hard | 50.2% | 46.6% | 49.5% | 49.2% | 47.6% |
| Short | 67.2% | 63.3% | 66.7% | 61.7% | 60.6% |
| Medium | 47.0% | 44.7% | 46.0% | 46.5% | 41.9% |
| Long | 42.6% | 34.3% | 43.5% | 38.9% | 39.8% |

The full official table and scorer hashes are in
`results/v31_value_selectors_20261008/reports/audit_scoring_no_exact_attempt002/summary.md`.
The machine-readable source is
`diagnostics/complete_audit_summary_no_exact_attempt002.json`.

## Trajectory interpretation

Every sparse arm differs from the control on all 503 output hashes, so the
accuracy comparison is not a same-trajectory mask comparison. Token/N
mismatch counts versus the control are:

| Arm | Output-hash mismatches | Token mismatches | N mismatches | Mean Delta N |
|---|---:|---:|---:|---:|
| V1 | 503 | 496 | 497 | -40.54 |
| V2 | 503 | 498 | 499 | -29.07 |
| V3a | 503 | 500 | 497 | -6.92 |
| Batch8 | 503 | 497 | 496 | +61.55 |

The control is also not interchangeable with native dense: all-kept FA4 skips
zero tiles but differs from native FULL on all 503 output hashes and scores
50.1% versus 49.1%. This is why native FULL, PIECEWISE, all-kept FA4, and V31
control must remain separate references.

## Integration instructions

The implementation is split into three layers:

1. `experiments/numerical_qk_reuse/v31_value_selectors.py` contains the trusted
   CPU formula oracles and selector dispatch.
2. `experiments/numerical_qk_reuse/v31_value_kernels.py` contains Triton
   statistics and selection kernels. V1/V2 use `triton_log_mass_online`; V3a
   and batch8 use `triton_parallel_cached_deletion`.
3. `experiments/numerical_qk_reuse/vllm_adapter.py` calls the selector only at
   initial/refresh map creation. It computes current rank-32 projected V,
   retains native FA4 observation masses, and returns the map to the unchanged
   FA4 consumer. LOCAL routing is not installed.

To integrate a new selector:

1. Add its stable name to `SELECTORS` in `v31_value_selectors.py`.
2. Implement and test its CPU oracle first, preserving `Stats` geometry
   (`U,J,Q` masses, `U,J,Q,R` means, `U,Q` reference norms) and mandatory
   support handling.
3. Add the matching Triton dispatch in `v31_value_kernels.py`; return only a
   GLOBAL keep map and candidate-evaluation counters.
4. Register validation in the adapter constructor. The selector must remain
   selector-only: do not replace the native FA4 output or alter native sampler
   stopping, acceptance, re-noising, or self-conditioning.
5. Emit a receipt containing selector name, threshold/budget, projection seed
   and rank, kernel, invalid rows, source hashes, phase tile counts, and CUDA
   capture count.
6. Run the focused selector tests and one-cell smoke qualification before
   freezing a new protocol. Use a new dated attempt directory; never overwrite
   a frozen run.
7. Run audit/scoring separately from clean timing. Audit instrumentation must
   not be used as a latency result.

The current control environment variables are `MAGE_K=8192`, `MAGE_STEP=1`,
`MAGE_CARRY=1`, `MAGE_RESELECT_TRIGGER=0.15`,
`MAGE_TRIGGER_SIGNAL=settle`, `MAGE_GRAN=qblock_max`, and `MAGE_STICKY=1.386`.
For V1/V2, set `VALUE_SELECTOR` to the arm name and
`VALUE_THRESHOLD=0.027`; keep `VALUE_AUDIT=1` only for audit and disable it for
clean timing. Keep `FIX_51994=1`, `FA4_LOCAL_FIX=1`, GLOBAL-only scope, and the
same model/runtime pins.

## Reproduction and resume

From the repository root, use the pinned interpreter and private manifest paths
recorded in `results/v31_value_selectors_20261008/private/continue_longbench_no_exact_attempt002.py`.
The frozen target protocol is:

```text
results/v31_value_selectors_20261008/protocol_execution_attempt005_no_exact_20261008.json
```

The official audit is already complete. To reproduce scoring from sanitized
records, use the official scorer identified by the protocol and compare its
outputs with:

```text
results/v31_value_selectors_20261008/reports/audit_scoring_no_exact_attempt002/
```

Do not rerun clean timing automatically. It was intentionally paused after
419/4024 records. A future timing continuation must reuse completed shards,
retain the partial current-control attempt, and create a new leaf attempt as
specified by:

```text
results/v31_value_selectors_20261008/longbench/target_no_exact/attempt002/timing_stop_20261009.json
```

Before any generation rerun, verify: branch/source commit, model revision,
manifest hash, host/GPU, vLLM/Torch versions, sampler settings, threshold,
arm list, and output root. Then run focused tests, one-cell smoke, receipt
validation, audit generation, official scoring, and only afterward clean timing.
Never infer accuracy from sparsity or timing from audit instrumentation.

The offline diagnostic aggregate is complete at
`results/v31_value_selectors_20261008/diagnostics/offline_accuracy_analysis_20261009/aggregate.json`.
It is a sampled masked-operator diagnostic, not a replacement for official
model generation or scoring.
