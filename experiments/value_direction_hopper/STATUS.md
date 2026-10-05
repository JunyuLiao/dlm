# Engineering status (in progress, 2026-09-20)

## Current handoff — September 20, 2026, 08:32 UTC

All jobs launched in this resumed iteration have completed. No GPU worker needs
monitoring now. Final390 outputs and all postrun measurements are finished;
**production-ready/speedup objective remains NOT achieved**. Primary report:
`results/value_direction_hopper_v1/systems_report_v3/report.md` with independent
audit, raw-only CSV/JSON, figures and exact limitations. Read that before older
notes below.

Final exact frozen binary873... yields85.28% Gaussian32 accuracy at70.13% physical
sparsity,196.30s total vs native dense90.26%,71.27s:2.75× slower. Global SDPA is
verified math fallback, local Flash~32µs vs our165µs. No whole-model/FA3 claim.
Cached historical Gaussian32 comparison: −1.923pp,95%CI[−4.846,+0.808],71.22%
token agreement. Native sparse/reference behavior is not bitwise interchangeable.

Additional completed variants:

* float-WGMMA131c...:42tests pass, native outputs/masks/state bitwise equal to
  frozen binary, local~8% faster/global~1% slower; NOT substituted into final.
* fair-TMA df806...:42pass/2ATen-skips (no bridge provided), exact CP/TMA dense/
  BLASST parity. `microbench_tma_controls_v2.json`: BLASST local148µs/global
  532–606µs versus value local153µs/global671–676µs; same loader, physical rules.
* explicit PTX release/acquire cluster834...:42pass/2ATen-skips under racecheck,
  zero hazards, all6native outputs bitwise equal to df806... . Sparse timing
  difference<0.2%; no meaningful gain, kept as an unselected control.
* Frozen873... synccheck8selectedtests passed,0errors. Existing memcheck/racecheck
  also clean. NCU hardware counters remain permission-blocked; Nsight Systems
  trace succeeded and CSV summaries are saved. Occupancy queries are resource
  bounds, NOT achieved occupancy/tensor utilization.

Current working source has optional VD_F32_OPERANDS/VD_PTX_CLUSTER and TMA dense/
BLASST controls; frozen binary's complete source archive remains under
`build/873c3073e54a37b8`. `build_variant.py` isolates variants. A new run must use
new output/configuration, never resume the finished old contract with changed
working-tree sources. No calibration/model/decoding/adaptive sources changed.

Remaining for a genuine production release: eliminate router critical-path and
register/occupancy costs enough to beat matched FA, characterize more native
states/near-threshold decisions, resolve reference/native local-mask mismatch
without silently changing science, validate full diffusion-model TRT integration
if required (attention-plugin support is real but not full model conversion),
and repeat end-to-end performance/accuracy on any new selected architecture.

## Resume checkpoint — September 20, 2026, 07:48 UTC

### Completion/update — 08:14 UTC

Final v1 finished **390/390, zero execution errors**. Detached worker and postrun
queue completed. Raw report and independent official rescoring/settings/per-step
layer-head audit both pass. Results:

|Method|Accuracy|Whole/global/local sparsity|Wall seconds|Denoising steps|
|---|---:|---|---:|---:|
|Native dense|90.2565%|0/0/0|71.2732|520|
|Gaussian32 kernel|85.2821%|70.1321/71.1173/69.4837%|196.3006|2123|
|Calibrated BLASST kernel|82.5128%|69.6796/69.8601/69.5622%|287.3750|3619|

Gaussian/native wall ratio0.363: **2.75× slower**, not a speedup. Native dense
differs from cached historical dense (mask/numerical convention); reporter now
correctly reads the latter from `dense/dense/shards`. Gaussian versus previous
Gaussian: −1.923pp, paired CI[−4.846,+0.808],71.22% token agreement,48.46% exact
sequences. Local rounding/denoising effects matter; do not assert preserved final
behavior merely from shared-state parity. Final sources/binary unchanged.

`systems_report_v2/report.md` and `audit.json` consolidate evidence. Postrun native
microbench completed: local TMA164.65µs vs matched-window Flash32.04µs; global
666.76µs vs SDPA math1258.22µs. **D512 is explicitly math fallback**, not FA3.
Naive Triton local522.87/global2180.49µs. Serial→overlap CP helps only~2%; TMA
further reduces latency. Nsight Systems trace and CSV stats exist. NCU still
permission blocked. CUDA resource queries show255regs/D256 and168regs plus432B
local spill/D512 TMA; at most1CTA/SM. Split4CTA native grid requires≥2waves.

After frozen final, compiler/SASS inspection found C7519 serialization between
individual WGMMA instructions. An isolated float-operand+CUTLASS fence variant
`131c9c8372d08e9f` removes all20warnings,42tests pass; all six native shared-state
cases are **bitwise identical** to frozen873... . Local improves~8% (164→152µs),
global regresses~1%. Preserved in `float_wgmma_comparison_v1.json` and
`assembly_audit_v1/`. Not substituted into final results.

New fair-control variant `df8066c5ad7dd4b4` extends TMA to dense and BLASST (same
mask/max/threshold semantics), so both algorithms can use the same loader.
Its tests currently exec session66314, output `kernel_tests_tma_controls.xml`.
No final rerun launched. `cuda.py` now permits TMA for those optional new controls;
this means resuming the old final executable directly would reject changed source
hashes (correctly). Old completed shards and archived sources remain authoritative.

Next: finish TMA-control tests and microbenchmark with `--tma-controls`; sanitize
and profile if useful; update systems report/README and preserve all qualifications.
The speedup objective is NOT achieved. No production-ready claim is justified.

This section supersedes the historical checkpoints below. Still **not production
ready** and no demonstrated matched-FA/end-to-end speedup.

* Frozen final evaluation exists at `results/value_direction_hopper_v1/final_ruler4k130_v1`.
  Two interactive interruptions stopped earlier workers after97 and152 completed
  shards. No experiment errors were recorded; all executable source hashes match.
  It was resumed detached at07:41 UTC, PID1120751, persistent log
  `worker_20260920T074121752680Z.log`. Do not launch a second worker. Use its
  `worker.lock`, status file and host process check. Check health roughly15min.
* A detached postrun queue (PID1121346, started07:48 UTC) waits on that same lock
  and requires final status complete. It runs native microbenchmarks, CUDA
  resource-limit queries, synthetic length/sparsity sweep and Nsight Systems
  sequentially. Logs/status under the engineering bundle. No concurrent GPU tests.
* Selected core: `value_direction_873c3073e54a37b8.so` (ABI3, FAST_SFU,
  INLINE_ROLES, packed masks, register-fed TF32x3, two-CTA TMA value path).
  ATen bridge: `torch_99ec6c443d8742eb/value_direction_torch_99ec6c443d8742eb.so`.
  Core source is archived under its hash; later source experiments must not
  silently replace the frozen binary. Final contract fingerprint
  `e379846cff792f34d2f2f5cf43f9f5c16048b751e682dda491af025396ea4e34`.
* Inlining improved captured local latency245→165µs, global1010→~667µs.
  Local native SDPA is~35µs, global SDPA fallback~1263µs; no FA3 D512 claim.
  Shared-P and online-mass-ratio experiments did not improve the selected path.
  The new BLASST control skips softmax on dropped tiles and uses PRIOR seen max.
* 41 CPU/CUDA tests passed in `kernel_tests_frozen.xml`. Selected core memcheck:
  zero errors (`memcheck_inline_v1.log`). TMA/ATen/partial-batch racecheck:
  zero hazards (`racecheck_inline_aten_v3.log`). No synccheck evidence yet.
* Native shared-logit qualification passed: zero physical-mask disagreements;
  state/LSE max<1e-4, BF16 online output relativeL2<1%. Native independent-QK
  absolute gate still failed and is preserved. `qualification_inline_v1.json`
  includes retained mass and full-dimensional operator error on layers0/5/29.
* Two official128-budget development questions scored1 for all four paths.
  Kernel versus historical sparse: one exact sequence, one24/25 matching tokens;
  denoising counts differ. See `generation_smoke_official_v2/summary.json`.
* All13 development tasks completed for native/value/BLASST (39 outputs each)
  in `development_profile_v1` and `development_profile_fused_v2`. These contain
  CUDA-event instrumentation and first-shape costs, NOT final latency claims.
* Fused projection/cache refresh passed native numerical/routing checks, ~3–4×
  lower preparation GPU latency (`projection_fused_v1.json`). It uses unchanged
  Gaussian32 matrices with FP32/TF32x3 arithmetic. The final uses fused projection,
  ATen launch, compact masks and native prompt-length-specific warmups.
* Real TRT-LLM functional static/serialized/stream/graph tests passed
  (`trt_engine_smoke_v3.json`, earlier core1a0...). Selected core873... linked
  plugin `trt_value_direction_inline_v4.so` passed dynamic D256/D512 batch1/2
  partial-shape parity (`trt_dynamic_v5.json`). Early dynamic harness failures
  were output-dtype/return-value handling and were corrected. Full DiffusionGemma
  TRT model conversion remains out of scope of the implemented attention plugin.
* CMake build succeeded at `build/cmake_inline_v1` with CMake4.3.2/CUDA13.1.
  Archived-source resource probe built as `resources_7ecb74680a0b5f7b.so`.
* `experiment report` regenerates only from shards. New `systems_report.py`
  adds independent official rescoring/settings/coverage audit and consolidates
  kernel/retained-mass/resource evidence. Run it after final and postrun finish.
  No earlier/adaptive results or sources were changed by this work.

Next: finish resumed390 outputs, inspect postrun failures, complete raw-only
reports and review measured bottlenecks before deciding another engineering
iteration. Keep all speedup/readiness and baseline-mask qualifications explicit.

## Historical checkpoint (superseded)

Still NOT production-ready and no final130 run launched. Main shortcoming is
performance, not inferred sparsity. Source variants and failures are preserved.

* Real TensorRT10.14.1.48 / TRT-LLM1.3.0rc6 V3 plugin engine build, serialize,
  reload and execution passed exact CUDA-harness parity at D256 and D512:
  `results/value_direction_hopper_v1/trt_engine_smoke_v2.json`. That was plugin
  version1 linked to ABI2 library19bed33d1b801af3. Current plugin source is version2
  (ABI3, internal workspace mask packing and TMA), not yet runtime-tested.
* Native QKV captured from the first development prompt at layers0(local),5 and29
  (global), first step; scale1.0. States/index under `native_qkv_v2/`.
* Important audit: original installed SDPA ignores its `sliding_window` keyword
  with None mask (encoder cache already clipped); frozen experimental reference
  additionally enforces per-query window. Keep frozen sparse semantics, report
  original native dense separately; do NOT claim exact unpruned native parity.
* Native QK roundoff audit: 148/4.72M local and1251–1314/17.24M global BF16 logits
  differ from Torch cuBLAS, max0.0625. FP64 audit finds fewer rounding mismatches
  for our QK than Torch's. Given IDENTICAL captured logits, reference router has
  zero mask differences, state max error1.72e-5–2.67e-5. Details in
  `native_qk_rounding.json`. Initial native absolute2e-4 gate failed and remains
  marked failed in profile files; do not hide it. Native output relativeL2~0.14–0.23%.
* Two capped16-token development generations match historical sparse output
  exactly (both16/16). `generation_smoke_v1/summary.json`. Capped scores all0
  including dense: NOT accuracy evidence. Kernel timing~0.68–1.75s vs native
  0.45–0.64s, includes first-use costs, NOT a performance claim.
* DiffusionGemma integration now exists: request-local producer lease cache,
  encoder-entry invalidation, actual prefix object/version checks, refresh
  boundary/canvas, fixed original Gaussian32 matrices. Compact structural masks
  avoid a dense QK validity tensor. Formal cache/geometry tests added.
* BLASST convention audited/fixed: current block max is compared to PRIOR seen
  max, then seen updates even on skip. Original/aggressive parity tests pass.
* 24 tests passed (`kernel_tests_online.xml`); added two async schedule tests
  subsequently. TMA/split racecheck reports0 hazards,0 errors,0 warnings in
  `racecheck_async_v2.log` (library87a1ad0c7f81fb3e).
* ABI3 adds optional diagnostic-only score output, six GPU timestamp fields;
  checks exported version and Params size. CPU refs/older .so require explicit
  `allow_legacy=True`; no final generation should use unversioned libraries.

Schedule work:

* CP.async16B + WMMA TF32x3 -> register-fed TF32x3 (precision2), padded bool masks,
  packed BF162 P writes, optional SFU math, __grid_constant__ params, TMA Q/K.
  IEEE and staged TF32x3 remain controls. No full PV on skipped tiles.
* Four-CTA alternative gives QK/router and PV separate register files/DSM P
  buffers, but native full-shape measurements REGRESS (~2ms global vs1.2ms
  two-CTA). Keep as unsuccessful controlled alternative, do not select it.
* Max/scaled-sum normalizer experiment regressed too and was reverted to log-space
  normalization; source/binaries preserved (b57f...,88911...). Squared-risk compare
  remains mathematically equivalent in the fixed path, log-risk diagnostic emitted.
* Fast-SFU + TMA native result51fdab2cdf574487: local~245us vs nativeSDPA34.7us;
  global~1006–1012us vs nativeSDPA1263us. Global native SDPA is NOT demonstrated
  FA3; installed Hopper FlashAttention rejects D512. No overall speedup yet.
* New in-progress hypothesis: padded (stride72) shared FP32 P + rolled TF32x3
  MMA loop (precision3) to cut live registers/instruction footprint. Build running
  exec session21944 at this checkpoint. Not tested yet. Current plugin only allows
  precision0/1/2; extend only after testing precision3. Latest completed core
  build1a0dcfd98d0dea78 has internal mask pack export and precision0/1/2.

Remaining: evaluate precision3; profile correct native baselines/BLASST early
skip-softmax baseline (current mode2 still computes softmax before deciding, so
not a fair optimized BLASST performance baseline); sanitize selected variants;
runtime-test V3 plugin2 with nondefault streams/CUDA graphs; strict C ABI guards;
build reproducibility; native generation and frozen130 final once stable. NCU
counters still permission-blocked; user async question unanswered. No old or
adaptive sources/results modified. Disk~2.8GB free; do not delete them.

## Earlier iteration notes (historical)

No production or speedup claim; do not launch the 130-question final sweep yet.
No historical/adaptive sources were edited. New files are isolated in this
directory and results/value_direction_hopper_v1.

Completed:

* Reviewed official TRT-LLM FMHA/XQA skip paths and installed Hopper FlashAttention.
* Fixed-rank32 fused Triton control, then C++ CUDA FMHA specialization using
  official TRT-LLM Apache-2.0 WGMMA primitives and CUTLASS shared-memory layouts.
* CUDA path uses a 2-CTA Hopper cluster: 64 queries per CTA, one shared conservative
  decision across all 128 queries. Router WG and separate PV WG(s). Two P slots;
  QK(next) overlaps PV(previous). Strictly increasing KV order, unchanged skipped
  retained state. Original full-dimensional V only loaded/used on retained tiles.
* BF16 QK/scaling rounding matches old reference. GQA, partial tiles and explicit
  bool/BF16 structural masks. CUDA graph capture works through the C ABI harness.
* Small/full-shape routing checks report zero mask differences, projected-state
  max error around 1e-6; output differs ~0.3% relative L2 from historical eager
  final-normalized-BF16-PV due to online rounding. Not bitwise generation parity.
* D256 probability-scratch CUDA racecheck: zero hazards (tool output in session).

Variants preserved as hash-named libraries/build logs in results/.../build:

* e3df5eb846e5f19f: first clustered kernel, D256 correctness, D512 initial register
  allocation too high. Initial scale redundant writers were fixed later.
* d59c5a83384b71f0: launch-bound/unroll adjustment, not benchmarked.
* aeb78e7be544c441: separate router/consumer functions; correctness including
  D512, but register spilling and scalar load latency make it very slow.
* b5b7abe45f093517: reuse K shared staging for FP32 probabilities, avoid dynamic
  indexed accumulator loads. Racecheck clean. Benchmark vs real dense backend.
* ab51b87eacf8b23d: optional GPU stage timestamps (extra ABI pointer at end).
* b3028ac39d38a368: aligned 16-byte cp.async Q/K/Z/V loads. Full synthetic
  correctness completed in cuda_async_copy.json, zero routing disagreements.
  Local ~360us and global ~1.8ms still much slower than dense.

Evidence / bottlenecks:

* `cuda_scratch_bench256.json`: D256,Q256,K1024, synthetic all-valid, SDPA dispatch
  is cuDNN (~68us); installed Hopper FlashAttention ~23.6us. Our initial dense
  schedule675us and value~795us at ~69% physical sparsity. Not a win.
* Stage timing before cp.async: QK including staging~30us/tile, PV including
  staging~28us, softmax/routing~9us dense/~15us value, cluster voting~1.1us.
  Scalar BF16 loads had a latency chain; cp.async substantially improved it.
* NCU performance counters are blocked by ERR_NVGPUCTRPERM. An asynchronous
  question requested host profiling access; no response yet. Do not bypass
  permissions with privileged containers or driver configuration changes.
* Existing Docker image `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc6` is available
  (the exact BLASST artifact environment). A read-only, network-disabled CPU
  import probe was launched, exec session35227. Check completion.
* No TRT runtime adapter or native DiffusionGemma generation integration yet.
  Current C ABI is NOT validated TensorRT-LLM runtime support.

Next useful work:

1. Profile post-cp.async stages; reduce FP32 PZ cost using validated TF32x3 tensor
   products (as Triton control already tested), or WMMA with hi/residual inputs.
   Keep FP32 state and compare masks; never silently BF16-quantize sketches.
2. Eliminate scalar mask-load chains (packed structural mask specialization) and
   standard exp/log overhead after accuracy validation. Current PZ SIMT has a
   clear cost. Separate mode specializations / register lifetime improvements.
3. Keep serial/overlap controls and validate after every change; sanitize async
   copy path. Test ties/first support/nonfinite rejection/cache invalidation.
4. Reuse existing DiffusionGemma binding and matrix cache, instrument true native
   QKV capture from calibration samples. Use immutable encoder KV identity/version
   for a safe fast cache, not tensor pointer alone; canvas always refresh.
5. Build/test actual TRT integration using existing container, verify baseline
   dispatch. D512 is rejected by installed Hopper FlashAttention; disclose.
6. Only after correctness/performance stability: frozen v19 Gaussian32 policy,
   matched130 RULER4K IDs, native decoding/budgets, measured end-to-end results.

Run commands: `PYTHONPATH=src:. OMP_NUM_THREADS=4` with
`/home/exouser/miniconda3/envs/ljy_dlm/bin/python`; GPU commands require host sandbox
escalation. `python -m experiments.value_direction_hopper.cuda` builds explicitly.
`validate --cuda-library PATH --output UNIQUE.json` validates/caches graphs.
`benchmark --library PATH --width 256 --keys 1024 --output UNIQUE.json` measures
controls/dense. --profile emits just one custom CUDA kernel. Hash-named build
archives now preserve source headers as well as compiler logs. Never overwrite
old result JSON. GPU worker count one. No active final-evaluation job exists.
