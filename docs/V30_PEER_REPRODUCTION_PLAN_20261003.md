# V30 peer reproduction proposal — root review required

This is a proposal and dependency audit, not an execution decision or a GPU
result. Junyu pin: `b890ff49488474c5d476df44019597dc045a5969`. Haowei pin:
`6f7279c1625f7fa53bacb233b96fb69a385df8c1`. The user has authorized isolated
reproduction and necessary documented fixes in our own directories. Peer branch
files and collaborator directories remain unchanged. Root must review the design,
freeze a new source/spec, then authorize deployment. Existing formal/component
queues and their pins are untouched.

The machine-readable proposal is
[V30_PEER_REPRODUCTION_SPEC_DRAFT_20261003.json](V30_PEER_REPRODUCTION_SPEC_DRAFT_20261003.json).
It is deliberately not launchable. Its CPU guard is
`scripts/v30_peer_reproduction_plan.py`, with six CPU tests. The script reads Git
objects and inventories source; it does not import, deploy, compile or run peer
code. Private inventories contain paths and source-byte manifests and stay out
of Git.

## Proposed first E2E question

Does Junyu's **unweighted Gaussian32 current-QK retained-state value router**
remain useful when only the five GLOBAL layers are eligible, LOCAL stays on the
unchanged native dense path, and all arms share full8192 generation and native
stopping? No C_gate, temporal coefficients, stale summary, R6, DP, carry-first,
or regroup is added. Peer mathematics is attributed to Junyu; this scope and
generation protocol is a named reproduction variant, not his original whole
attention experiment.

Proposed arms on one host/source/runtime per cell:

1. Original HF/SDPA native dense.
2. Same GLOBAL-only peer kernel at threshold negative infinity: all tiles kept,
   retaining projection/router/dispatch costs as a matched implementation control.
3. Same GLOBAL-only peer kernel with one independently bound Gaussian32 policy.

The first protocol is explicitly **HF/SDPA qualification and E2E evidence**, not
an acceleration claim against official vLLM/FA4. Peer `experiment.py:72–77` and
README:357–360 disclose the SDPA baseline; changing the title does not repair
this limitation. A subsequent strong-reference protocol must put the same peer
selection and matched all-kept consumer on the official FA4 substrate, with
native dense and hook controls. No such port is implemented by this proposal.
Root may prefer to fund that port first instead of this HF stage.

Multi-question, eight-seed design is mandatory. Proposed first bounded panel is
three predeclared AIME and three predeclared LongBench32K items ×8 distinct
seeds ×3 arms =144 timed requests. This is a development pilot, with no fresh
held-out claim. Full expansion is AIME30 + LongBench59 ×8 ×3 =2136 requests,
disabled until root review and complete qualification. Longest-prompt/full8192
qualification per arm only proves pipeline execution; it is not a speed or
quality conclusion. All questions remain in scoring, including caps/nonanswers.

The proposal uses thinking ON and generation8192 for every arm, so it differs
explicitly from peer `_request`, which uses thinkingFalse and the row's historical
budget (`experiments/diffusion_gemma_solattn_blasst_multibench/runner.py:_request`).
Canvas256, step48, native0.8→0.4 schedule, confidence.005, stability1 and entropy.1
must be checked against actual loaded generation configuration. Seed equality
does not establish identical RNG trajectories. No final score is used to retune
thresholds. Existing policies are usable only after exact bytes, calibration
scope and source provenance are verified; no suitable policy is presently bound.

## Exact integration boundaries to review

Peer `integration.install(adapter,library,thresholds,**kwargs)` creates a
binding/router and installs `binding.runtime.attention_override=router`
(`integration.py:168–181`). `Attention` accepts precision, collect, tma, mode,
projections, profile, projection, torch_library and blasst_tma (`:78–87`).
For Gaussian32 use mode=value, precision=tf32x3_register, projection=fused,
sensitivity=None, and no `query_adaptive.observe` state wrapper.

The current install defaults cover both LOCAL and GLOBAL. A new own scope wrapper
must route only layer5/11/17/23/29 and delegate LOCAL and ineligible encoder/commit
calls to the original native callable, preserving mask/scaling/kwargs. This is
not achieved merely by setting the LOCAL threshold to negative infinity: that
would still replace LOCAL arithmetic and pay the peer kernel's work. The wrapper
does not yet exist and must get independent unit, layer/call counter and GPU
output qualification before launch. Peer `experiment.generate` is a useful call
reference but is not directly reusable unchanged because it binds historical
private SOURCE/policies and calls its historical `_request`.

Do not use the historical query-adaptive one-canvas assertion for full8192:
`query_adaptive_study.generate:138–140` assumes one native canvas. The unweighted
router itself has request-local prefix-sketch invalidation and per-call boundary
refresh (`integration.py:17–68`). A new runner must count every actual canvas,
denoising forward, commit and prefill, preserve native stopping, and prove prefix
lease invalidation across encoder entry and requests.

## Build and source plan

1. Export committed peer objects into a new private own directory; preserve its
   SHA and source-byte manifest. Do not import similarly named files from our
   working tree, merge his branch, or execute his old binary by assumption.
   Candidate dependency roots are declared by the CPU inventory script. Add
   transitive dependencies only after CUDA-hidden import checks; every missing
   import, syntax repair or path patch gets a precise diff and new source pin.
2. Peer CUDA builder requires the TensorRT-LLM FMHA Hopper headers, CUTLASS
   `cute/tensor.hpp`, CCCL, driver/runtime headers, a real nvcc, C++17 and sm_90a
   (`cuda.py:44–78`). Pin the entire header closure, not only two direct headers.
   No TensorRT runtime/plugin is required for the plain C ABI/ATen route;
   CMake's plugin build is optional (`CMakeLists.txt:10,29–37`). It is not a
   TensorRT-LLM runtime integration.
3. `cuda.build(fast_sfu=...,inline_roles=...,online_ratio=...)` hardcodes system
   nvcc/runtime and writes a content-hashed build under results; `torch_build`
   hardcodes system CUDA include/link paths. In the isolated copy, change only
   toolchain/output/header paths to own roots, record the exact diff and compiler
   command, and preserve algorithm flags. Use the documented fast_sfu+inline_roles
   build as a named pinned variant; qualification must cover its numerics. Do not
   select a faster variant by looking at final E2E results.
4. Build an ATen bridge with exactly the chosen Torch runtime and kernel. Pin
   kernel hash, bridge provenance, compiler/CCCL/runtime/header versions and ABI.
   `cuda.Kernel:84–117` validates ABI3/4 and Torch/kernel provenance, but existence
   of an old `.so` does not qualify a rebuilt source/runtime combination.
5. CUDA-hidden import/tests and standalone compilation precede any GPU process.
   GPU qualification follows on an idle registered H100: finite complete outputs,
   existing independent FP32/kernel oracles, original row identity, masks/tails,
   GQA16/2 and D512, Gaussian projection, no unexpected LOCAL routes, all-kept zero
   skips, graph/stream semantics and complete native counts. Existing test tolerance
   is frozen, never widened after a failure. Failure gets a new retained attempt.

[NVIDIA nvcc12.8 documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html)
supports sm_90a compilation.
[NVIDIA's Hopper compatibility guide](https://docs.nvidia.com/cuda/archive/12.8.2/hopper-compatibility-guide/index.html)
warns that architecture-conditional sm_90a binaries are not general portable
fallback binaries; qualify on the actual registered Hopper device.

Read-only own-host metadata on2026-10-03 found HF Torch2.12.1/Transformers5.11.0
and vLLM Torch2.13.0/Transformers5.18.0 environments on dllm/dlm2; both are H100
80GB. dlm2 has the earlier own CUDA12.8.93 redist compiler. Limited known-path
checks did not find the peer TensorRT-LLM checkout on either host, nor a compiler
in the checked dllm locations. This is not an exhaustive absence assertion.
dllm's original own configuration references two existing libraries; their
equivalence to b890 source/Torch has not been proven. A matching CUDA13 compiler
may already be available in other own paths and must be inventoried before
choosing/rebuilding a compatible runtime. No package/environment changed.

The candidate source inventory also finds one historical unrelated report file
that Python3.11 cannot parse (`diffusion_gemma_solattn_blasst_multibench/report.py:104`,
same-quote nested f-string). It is not a demonstrated runtime import dependency
of the proposed worker. Do not alter peer source broadly; minimize the executable
closure or document a compatibility-only fix if it is actually imported by the
chosen registered interpreter. This does not contradict syntax qualification of
the newly added Hopper files.

## Haowei regroup cooperation candidates, maximum three

Haowei's pinned code has a concrete zero-global-Q-copy mechanism:
`blasst_wgmma_kernel.cuh:359–364` loads original Q rows via q_ids into shared memory;
the split merge writes directly to qrow (`:534–547`). Its FP32/TF32 custom kernel
and GQ32 are not our BF16 FA4 D512 consumer. The scheduler explicitly calls its
CPU grouping a scaffold (`regroup/regroup_scheduler.py:4–14`); the pinned-buffer
H2D/ready event is not proof of free CPU construction or overlap on our model.

| Candidate | Why earlier negatives do not fully cover it | Fair component test and complete cost |
|---|---|---|
| Tile-only dispatch order, no row regroup | Reorders only program tile order. Original contiguous Q and O remain untouched; support and total selected tiles remain identical. Prior full gather/scatter negatives measure a different data path. | Compare natural versus fixed support-similarity/load-balanced program order on the **same consumer**, plus official FA4. Include order construction/upload and any persistent scheduler; full outputs/row identity and zero extra skips must match. Benefit can only be locality/load balance, not less attention arithmetic. |
| Consumer-internal q_ids and original-row output | Haowei demonstrates the insertion points; avoids separate Q gather and O scatter. Earlier sparse-cycle component still pays forward+inverse restoration, so it does not establish this consumer's cost. | Independent opt-in consumer design requires root review. Identity-map version is the matched reference; official FA4 remains the strong baseline. Full GQA/BF16/D512/mask oracle, same frozen held support/order, metadata load and noncontiguous Q shared-load cost all counted. Losing TMA/coalescing may erase the saving; no promised speed. |
| Bounded GPU support-key grouping on existing R6 support | Earlier held-order CPU search costs about1s/state. A bounded small-key build avoids that search; it cannot be treated as equivalent to the old order. Haowei distinguishes count versus set-key grouping on nonnested support (`_skip_regroup.py:198–231`, `_skip_regroup_jax.py:116–141`). | At most a predeclared fixed bucket/key and bounded adjacent exchanges, using already available per-row support. No current-step future information. Preserve per-row required coverage, protected boundaries and original positions. Include packing/sort/upload, actual moved rows/bytes, gather or indexed consumer, and reconstruction at invalidation. A new policy needs fresh same-state support oracle and end-task quality qualification. |

For every candidate, measure construction time B and per-use total saving d
against the matched fused natural reference. If d<=0 there is no break-even.
Otherwise required valid uses are ceil(B/d), compared with an **observed** support
epoch lifetime; R6 is a redecision interval, not evidence of six valid reuses.
With1s build and20microseconds saving the optimistic requirement is50000 uses.
Q gather for H16×Q256×D512 BF16 reads+writes is8MiB per layer/call; a same-size
O scatter adds another8MiB. Removing that movement is only a bound, not a timing
prediction. New key building
must independently beat its own cost rather than inherit a free offline order.

## Remaining blockers and planning envelope

Root decisions: HF qualification-first versus direct FA4 port; exact registered
runtime/compiler; GLOBAL-only wrapper boundary; existing policy provenance versus
new calibration; predeclared item list and seeds. Technical blockers: header
closure/pin, matching own bridge build, multi-canvas accounting, actual capture/JIT
qualification and scorer proof. A new FA4 port is research implementation work,
not an environment-only deployment fix.

Once those decisions are frozen, CPU source/import/build preparation is plausibly
tens of minutes to a few hours, with no guaranteed deadline before resolving
headers/compiler compatibility. The144-request pilot could reserve roughly6–12h
if full8192 requests take150–300s each, plus fresh-engine startup/warm/teardown
(up to48 processes if each dataset/arm/seed is isolated);
this is a scheduling scenario, not a measured forecast. Qualification receipts
must replace it before launch. Full2136-request expansion needs a separate measured
schedule and is not proposed for automatic execution now.

All output text, RNG states, tokens, gold, private paths and source-side private
artifacts remain in own private storage. Publish per-request numeric evidence,
worker/resource spans and question-cluster uncertainty only after strict scoring.
