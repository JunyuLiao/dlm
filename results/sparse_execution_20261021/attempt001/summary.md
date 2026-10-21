# Sparse execution diagnostic progress, 2026-10-21

Benchmark attempt004 remains paused/partial. These results are synthetic qualification, not benchmark accuracy or request speedup. Source files are copied/hashed before each screen; native001 is separately frozen.

## Held execution, complete 12-cell synthetic matrix

Q256/H16/Hkv2/D512 BF16, keys1024/4096/8192/16384, realized physical GLOBAL sparsity near40/50/60%. All cells pass finite/output checks and actual timed CUDA-capture counter0, with no other GPU clients. Compilations and graph captures occur before timing.

Host/module medians at exactly50% physical sparsity (microseconds):

| Keys | Plain dense FA4 | Native paged dense FA4 | Previous paged sparse | Warm held bucket, charged | Compiled Flex, resident |
|---:|---:|---:|---:|---:|---:|
| 1024 | 62.74 | 141.90 | 266.46 | 33.62 | 114.78 |
| 4096 | 175.12 | 206.70 | 278.89 | 101.35 | 410.73 |
| 8192 | 365.39 | 294.96 | 277.84 | 128.41 | 788.20 |
| 16384 | 726.83 | 480.37 | 277.67 | 208.20 | 1561.54 |

Warm bucket charges Q staging, graph replay, output delivery, merge and synchronization. Current K/V are read directly from the native interleaved cache; no prefix/canvas copy or selector/list build is executed on held calls. Resident compact metadata is prepared at refresh/carry and timed separately. Host timing includes Python and one final synchronization per30-call batch. Native model KV writes are common work; model/module/request times still require paired native controls.

Graph/event kernel timings and all40/50/60% cells are in screen005/receipt.json. FlexAttention is correct but slower, retained as a negative prototype. GraphPagedBucket changing-length/mask/page/Q/current-KV tests pass2 cases; persistent direct-output tests plus fused projection parity pass10.

## Refresh, complete 12-cell synthetic matrix

V2 remains true online conditional full-PV in one traversal. Projection/RMS, sketch packing, mask/index creation, K/V copies and module/host timing are included as named components in refresh001/receipt.json. QK/softmax/projected-PV/risk and full PV share one kernel and are not separately timed; the all-kept-core comparison is an ablation rather than a PV-only timer. No fabricated hardware stall fractions or request times.

At50% actual physical sparsity, graph/event complete refresh medians (microseconds):

| Keys | Dense plain | Native paged dense | V31 K128 | Online V2 | Fused-pack V2 | Two-pass diagnostic |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 47.18 | 136.93 | 55.79 | 84.06 | 82.94 | 186.71 |
| 4096 | 172.02 | 199.39 | 189.03 | 238.69 | 236.98 | 615.25 |
| 8192 | 362.18 | 286.86 | 365.50 | 457.56 | 459.29 | 1167.24 |
| 16384 | 708.80 | 468.52 | 719.94 | 897.47 | 906.91 | 2259.05 |

Fused current-V projection/packing is bitwise equal to the original sketch/norm/RMS/packed operands across449/1024/4096/8192 keys, but its performance is mixed and it is not promoted into native execution.

Unique-QK packed three-WG router variant removes four-way QK/routing duplication, uses167 registers and zero compiler spills, and passes4 changing-input graph cases. It remains slower:8K complete fused refresh517.50us versus459.29us selected four-group variant. Preserve refresh003 and its source/build receipts as a negative; no promotion.

## Native qualification and remaining gates

native001 resident direct-paged fixed-split variant: one bounded HumanEval K128 request, audit/clean output hash and all204 sampler calls/16canvas intervals agree;0captures, no validation errors. Audit695held calls have0KV copy/selector/list builds;775held+carry calls, phase tile receipts complete. This is not a scored benchmark or an end-to-end speed comparison.

native002 warms static execution buckets and tests the charged graph-bucket variant. Qualification is in progress. No full panel can be relaunched yet: native V2 target40-50% held sparsity per workload, native matched controls, current graph-bucket audit/clean agreement, and faster V2 refresh remain unresolved. The saved old threshold is not accepted as new calibration.

Failed projection compilation attempts tests002/003, the empty test selection unique_tests001, missing build-capability metadata refresh002, and unsupported nested-capture harness screen004 are retained. Their corrected successors are separate attempts.

## Direct paged V31 and additional V2 scheduling screen

refresh004 adds current-cache paged V31 observation, including a paged rare
underflow repair. Two operator cases with changing mappings/values/extreme
logits match the inherited exact-budget selector. At8K direct paged302.55us
vs native paged dense288.34us;16K492.32us vs470.09us. At1K/4K the paged path
loses to contiguous observation, so the named runtime dispatch stays contiguous
through4096 and uses direct paged observation above4096. No installed FA4 edits.
The two-channel router/dual-PV variant passes4 graph cases and compiles168regs/
no spills but remains slower than selected V2 at8K484.74us; retain as negative.

Native002 graph-bucket audit/clean trajectories match exactly (49sampler calls,
5canvases, output hash and intervals), with0captures. Timing is withheld because
selector JIT warnings occurred near the measured request. Native003 was prepared
but never launched after synthetic dispatch selection changed. Native004 adds
selector specialization warmup, explicit per-request Triton compilation counters,
and the static V31 dispatch above. This is bounded qualification only.

Native004 explicit selector prewarm passes the audit/clean trajectory pair with
0CUDA captures and0Triton compilations. Native005 long-output prior-examined
HumanEval145 audit executes497denoising forwards/32canvases,170direct paged
V31 refreshes and2000held calls;0timed captures/Triton compilations/extra forwards/
budget errors. Its clean pair is pending. No comparison of these request times
with different split arithmetic or sampler trajectories is a speed estimate.
Unique retained-only queue refresh005 is a negative:8K480.41us, lower than
unique all-event queue517.50us but above selected four-group459.29us.

Additional qualified numerical/execution variants: full/partial Gaussian32 fused
packing plus online output pass7 parity cases. Within-canvas unchanged-prefix
projection reuse yields82.82/236.67/449.56/875.65us at1/4/8/16K compared with
full projection83.64/238.61/457.89/913.31us in refresh006. It is an experimental
within-canvas cache only; new canvases recompute current V. It has not been
promoted into the native worker. Register-source PV for the retained queue
passes4 current-Q/K/V graph cases but regresses8K to599.81us (refresh007).

Native005 audit/clean output hash,529sampler calls/32canvas intervals match;
170direct paged refreshes,2000held calls,0timed captures/Triton compilations.
Native006 V2 tau0.14 long examined diagnostic passes0timed JIT events/captures/
LOCAL skips/extra forwards, but held sparsity72.52% exceeds the40-50% target.
A lower scalar threshold is being measured. No workload-wide calibration claim.

Nsight Compute2025.4.1 profiling attempt profile001 failed ERR_NVGPUCTRPERM;
GPU hardware counters are inaccessible. No driver/module permissions changed.
A separate clock-instrumented diagnostic is being prepared; it will not be used
as clean latency or hardware stall attribution.

Native006 tau0.05 on the examined long HumanEval145 diagnostic achieves45.338%
held GLOBAL physical sparsity. Audit/clean output hash,563sampler calls and
canvas intervals match exactly;0timed JIT events,0Triton compilations,0captures,
0LOCAL skips and0extra forwards. Whole GLOBAL eligible5,978,400/kept3,283,050
(45.087% sparsity). The per-phase and first-canvas physical work are recorded
in native006/paired_v2_tau005_qualification.json. This scalar is a one-item
diagnostic threshold, not per-workload calibration or a benchmark conclusion.

Stage-clock diagnostic stage_clocks001 covers all12 synthetic context/sparsity
cells with bitwise output/mask parity. At50%, the instrumented per-CTA cycle
intervals for QK+softmax+sketch staging are63.5-65.7%, projected PV12.0-12.5%,
risk vote9.0-9.4%, conditional PV+value waiting9.6-10.7%, next-key waiting3.7-
3.8%. These fractions are perturbed by instrumentation and are not clean
latency fractions, FLOP shares or hardware stall counters. Four-way QK/routing
duplication remains present in the selected production core.

3D TMA bulk descriptor variant passes13 kernel/graph/current-V cases, complete
8K449.02us (refresh008) versus selected457.89us in refresh006. Modest gain,
still slower than dense. Hoisted QK descriptors also pass13 cases but give
8K457.14us, no material improvement (refresh009). They are retained separate
variants, not promoted to benchmark evaluation.

Mutable campaign controller now refuses explicit PAUSED_USER_REQUEST attempts
and disables its legacy full-run/preparation paths. Two guard tests pass. Frozen
old coordinator/source/records remain untouched. A future relaunch needs a new
qualified held-sparsity protocol; legacy first-canvas work matching is disabled.

The full-output single-QK Triton prototype is also retained: four-CTA automatic
planning fails a compiler assertion (distributed001); one-CTA two-stage requires
335872bytes shared memory (distributed002);16warps cannot meet WGMMA register
minimum (distributed003). One-stage8warps compiles and matches masks/output,
but8K core1826.84us loses badly to dense360.73us (distributed004). No promotion.

Final combined regression001:40passed,0failures/0skips. Current experimental
source hashes saved under ../qualified_source/attempt001/source_manifest.json.
Native attempts pin their exact earlier snapshots; late short-list-reuse and
extra JIT receipt fields are not silently assigned to older native receipts.
Full evaluation gate remains closed. Final device/process check saved separately.

Local Git staging did not execute: automatic approval review timed out on the
initial request and permitted retry. No commit created; source snapshot hashes
remain the implementation identity. See commit_status.json.
