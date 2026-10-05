# v9 request trace accounting (section 6)

Diagnostic only. S (production M1 preqk, LOCAL prefix_block_summary), aime26/2, one process,
request truncated after canvas 8 step 1. Deployed `3e88d0b`. Machine-readable:
`request_trace_accounting.json` (raw Chrome traces on the large disk, referenced by sha256).

## Why "430 ms per forward" was never a per-forward cost
v7's figure was request wall / decoder calls for a first-in-process request. In v9's clean E2E:

| request | attempt 0 (cold in-process) | warm repeat (median) | new Triton specializations in attempt 0 |
|---|---:|---:|---:|
| L /2 (ran first) | 63.85 s | 19.53 s | 51 (39 new disk entries) |
| S /2 (after L) | 34.66 s | 19.59 s | 10 |
| S /8 (first on /8) | 82.27 s | 33.98 s | 61 (61 new disk entries) |
| L /8 (after S) | 38.32 s | 33.72 s | 4 |

Warm: 187-192 ms per decoder call (request level, including prefill/commit), vs corrected replay
steps 153-167 ms. Most of the cold excess is Triton compilation, which the compile timeline
below explains.

## Compile timeline (pass A, fresh process, warm disk cache)
`K`, `KT` and `PREFIX_TILES` are `tl.constexpr` in `_route`/`_pv`/`_preqk_pv`. The key length took
10 distinct values by canvas 8: 365, 621, 877, 1133, then local saturates at 1279/1389
(crop-aligned) while global keeps growing 1645, 1901, 2157, 2413, i.e. new specializations
every canvas for the whole request. 47 in-memory misses cost only 0.35 s here because the E2E
had already populated the disk cache (~1-3 ms per load). A new prompt length hits cold
disk entries, and each one is a real compile (~0.9 s). **This is the largest verified cost of the
L/S path on a new request (+44-48 s on a 20-34 s request). D compiles nothing.**

## Wall partition, pass A (unprofiled host timeline, 16.38 s truncated request)
| part | seconds |
|---|---:|
| initial prefill (first encoder, includes first-call warmup) | 0.60 |
| 75 denoising steps (includes 0.35 s of Triton miss handling) | 14.37 |
| 8 commit/encoder calls | 1.13 |
| unattributed (prompt prep, canvas finalize, Python) | 0.28 |
Steps: canvas 1 = 171.5 ms/step, canvas 8 = 169.5 ms/step (no sync inside; step boundaries only).

## Two profiler windows (pass B, same trajectory, in-memory warm; 2 steps each = anchor + ordinary)
Device intervals are unions; label = innermost record_function enclosing the launch.

| per step (avg of 2) | early (canvas 1) | late (canvas 8) |
|---|---:|---:|
| kernels launched | ~10.7k | ~10.6k |
| GPU-active union | 67.8 ms | 80.5 ms |
| MoE experts: kernels / GPU ms | ~5.2k / 29.8 | ~5.0k / 29.0 |
| selector total GPU ms (route, fused route+PV, anchor scores, preqk PV, sketch, call glue) | 14.0 | 27.1 |
| of which `_route` kernel | 8.5 | 18.1 |
| selector-attributed launches | ~1.4k | ~1.4k |
| attention module (projections, norms, rope) GPU ms | 5.6 | 6.1 |
| T observe_logits GPU ms | 1.6 | 1.6 |
| sync-like runtime calls | 289 | 289 |
| cudaMalloc/cudaFree calls | 0 | 0 |
Profiling overhead: windows spanned 636/597 ms against ~2x170 ms unprofiled (~1.8x). So
host gaps are inflated, while kernel durations are device-measured.

Reading: the decoder step is **host/launch-bound**. Unprofiled steps take ~170 ms while the GPU is
active ~68-80 ms. About half of all launches come from the MoE expert loop, which native
shares. The selector adds ~14-27 ms of kernel time per step (mostly `_route`, growing with
key length; not split per layer) and ~1.4k launches (+~13%). There is no allocator churn.

## Explicitly unassigned
- Native was not traced (window budget): the split of the ~35-45 ms/step L/S-minus-native gap
  between added selector kernels, added launches, and overlap is NOT established.
- Replay vs request at canvas 1: 153-162 ms replay vs 171.5 ms in-request. Roughly 10-18 ms/step
  is unassigned (candidates: stream not drained at step start in replay vs the native per-step
  `torch.all` sync, CanvasCalls clone, cross-step overlap). At canvas 8, request ~170 ms vs replay
  canvas-6 ~167 ms (different canvases).
- L2 residency: not tested; no claim.
