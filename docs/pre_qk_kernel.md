# Conservative pre-QK kernel

The production control remains the existing 128x64 sparse BLASST kernel. It
computes QK for every physical tile, then may skip softmax/PV. The pre-QK
experiment is a separate Triton specialization, so disabling it removes all
proxy metadata loads, stores, and predicates from the control kernel.

## Runtime modes

`PreQKKernelConfig` exposes two explicit switches:

- `use_pre_qk_kernel=False, enable_pre_skipping=False`: original sparse kernel.
- `use_pre_qk_kernel=True, enable_pre_skipping=False`: metadata/deferred-V
  kernel, but the early gate is forced off. This isolates kernel-layout cost.
- `use_pre_qk_kernel=True, enable_pre_skipping=True`: certified previous-step
  score thresholds may bypass K, V, QK, softmax, and PV for selected tiles.

Enabling pre-skipping without selecting the specialized kernel is rejected.
All defaults are disabled.

## Decision logic

Each layer owns FP16 double buffers shaped
`[batch, heads, ceil(S/128), ceil(S/64)]`. Values are base-2 logarithms of the
ordinary physical BLASST score. The first diffusion step has no source and
therefore computes every QK tile while producing metadata.

At a later step, before loading K or V, a tile is pre-skipped only if:

1. a verified previous-step score is available;
2. its score is strictly below a certified, transition/layer/head threshold;
3. it is not in the initial traversal warmup;
4. it is not a periodic refresh tile;
5. it is not in the protected local region or sink tile.

Otherwise QK is computed and ordinary BLASST makes the exact downstream-skip
decision. There is never a predicted-keep shortcut.

The query/KV geometry is asymmetric: one 128-row query tile covers two 64-row
KV tiles. Local protection accounts for this 2:1 mapping.

## Non-cascading invariant

A pre-skipped tile is written as `+inf`, never as a predicted score. Every
later score in that query-tile traversal is also written as `+inf`, because a
false pre-skip could have changed its running maximum. Consequently,
approximate state is never admitted as a source on the next diffusion step.

## Benchmark

First produce certified thresholds from repaired traces:

```bash
python -m proxy.calibrate_proxy_thresholds artifacts/proxy_traces_v3 \
  --output-dir artifacts/proxy_calibration_v3 \
  --false-skip-budgets 1e-2,1e-3,1e-4 \
  --confidence 0.99 --min-predictions 128 --previous-step-only
```

Then run the target-step H100 benchmark:

```bash
python scripts/llada_pre_qk_kernel_benchmark.py \
  --source-mask-ratio 0.9 --target-mask-ratio 0.5 \
  --false-skip-budget 1e-3 --warmup 2 --repeats 10
```

The report contains compiled dense FlashAttention, sparse BLASST without
pre-QK, the specialized kernel with its gate disabled, and pre-QK enabled.
Counter atomics run in separate forwards. It reports speedups among all modes,
QK avoidance, and masked/all-token top-1 agreement with both dense and sparse
baselines.

`--manual-proxy-threshold` exists only for an explicitly labeled uncertified
kernel ablation. It must not be reported as a safe deployment result.
If no eligible threshold exists, the benchmark still runs all baselines and
the specialized-kernel ablation with a `-inf` gate, yielding zero pre-skips.

The current 2,405,376-record legacy-v2 trace corpus has only four calibration
and four held-out samples. Under the family-wise Wilson/New-Max criteria above,
none of 6,342 previous-step threshold candidates passed, including at the 1%
false-skip budget. Legacy-v2 traces are additionally ineligible for deployment
because they predate the corrected 2:1 local geometry. The safe runtime result
is therefore zero pre-skips; more independent v3 prompts/seeds are needed
before latency with the gate enabled is a defensible optimization result.
