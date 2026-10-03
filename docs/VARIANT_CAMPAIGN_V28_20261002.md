# V28: bounded variant and model qualification campaign

User authorization: try every variant judged worth testing (2026-10-02, US Central UTC-5).
Branch: `research/vllm-variants-20261002`, starting at `9b027df8f`.
This independent worktree does not change the frozen V18b deployment or its parent-branch finalizer.
All hypotheses below are prospective, not measured improvements.

## Candidate inventory and order

| Candidate | What is tested | Necessary reference | Initial gate |
|---|---|---|---|
| Q64-native | Existing 64-row selector refinement on the native vLLM alias-split consumer | unchanged main on exactly the same deployment | list/coverage/numerical tests, identical-state kernel bench including selector costs |
| KV-view | Avoid adapter KV copies when the allocated pages form a proven contiguous view, copy fallback otherwise | copy-backed all-kept and main, plus native dense | stride/layout/fragmented-page/identity tests; core consumers must honor strides |
| Lifecycle | Release stale request/canvas tensor references if audit demonstrates retention | existing lifecycle, same mathematical method | CPU ownership tests then peak-memory and allocator-retry counters; never insert empty_cache to hide a leak |
| Regroup-balance | Group rows using the two-way alias-split load objective, with a conservative acceptance gate | natural q64; old q64r as historical/control diagnostic | true-need CPU screening: total, max and p95 work, no dropped required bits, 64-row groups, permutation bijection |
| Regroup-amortized | Reuse a selected permutation across held calls or fuse permutation with an existing layout transformation | q64 and the same unfused regroup | include Q gather/output scatter on every actual use; break-even test before request panel |

The last two are cooperation candidates attributed to Haowei/`chw/value_aware` for the regroup idea.
No peer code is merged or copied. Per-head full-canvas count/set/greedy and cross-head pack/count/set/greedy
are already covered by historical negative diagnostics. They are controls, not new discoveries.
Do not repeat unchanged q64c/c01 or a same-threshold C-gate panel without a new, separately specified reason.

## Stage gates

1. Commit this specification before generation. Inventory idle GPUs, environments and private real-state dumps.
2. CPU implementation/ownership tests and private need-matrix screening. Publish aggregates only.
   Work proxies are not GPU time; reject a candidate if its optimistic saving cannot pay its unavoidable costs.
3. Freeze source and configuration, deploy to an isolated dyh directory, bind immutable inputs.
   Qualify numerical behavior, path receipts and graph/compilation counters before any timing claim.
4. Benchmark identical real Q/K/V states with warmup and interleaved order. Include selector, permutation,
   split and merge costs. For standard optimizations, apply the optimization to eligible matched controls too.
5. Advance plausible candidates to same-host official dense / optimized-reference / candidate request panels.
   Use identical items, stopping/budgets, deployment and engine-seed blocks; never change a running panel.
   Report W, S, S/N, actual N, commit count, correctness and question-cluster 95% intervals.
   Preview samples cannot establish noninferiority. Failed runs are retained and relaunches use new directories.

Frozen V18b currently runs on dllm. Its logs show recoverable allocation failures while generation continues;
their cost belongs to that implementation's measured W. Do not discard slow rows or silently reinterpret them
as allocator-free timings. Lifecycle and memory optimization are follow-up comparisons, not retroactive edits.

## New-model track

LLaDA2.1-mini and I-DLM-8B are not yet sparse-method ports. Environments and weights are prepared only on dllm.
Use an idle second GPU after an isolated environment/weight transfer and toolchain qualification if feasible.

1. LLaDA upstream SGLang/JointThreshold public-toy smoke; compare official dense candidates before selecting
   a headline baseline. CUDA 13.0 shim is only qualified for that upstream environment.
2. I-DLM bundled SGLang/IDLMBlockN public-toy smoke, with a separately qualified CUDA 12.8 toolchain.
3. Implement model-specific clock/cache/sampler adapters. LLaDA block=32 and I-DLM causal strided verification
   are not Gemma canvas=256; carry, temporal sensitivity and query-tile assumptions require explicit redefinition.
4. Before new sparse algorithms: all-kept equality/control and actual-forward receipts. Then matched main/hold/
   relevant efficient variants. Do not port every historically rejected knob or label unimplemented variants tested.

One GPU worker per card, launch only after idle checks, all writes/caches inside own dyh directories.
Track reserved GPU seconds including failed qualification. Never publish prompts, token arrays/hashes, gold,
private paths or raw generation records. No shared package/cache modification.

## Initial state

Specification and implementation preparation only. No V28 performance or quality result yet.
Parent V18b is a separate running campaign; consult its original branch for scored results.

## Implementation checkpoint — 2026-10-02 19:08 (UTC-5)

31 CPU tests pass across lifecycle, adapter guards, support refinement and regroup.
`request_clear` is opt-in, defaults remain legacy, and releases bounded retained
request objects even on close errors. It adds no empty_cache or synchronization.
The actual native cache has HK=2 interleaving between pages; a flat zero-copy KV
view is invalid. Reject KV-view rather than silently copying or misaddressing.

`v28_alias2_q64_bench` qualifies random physical-page order and actual cache
strides against each support's masked FP32 reference. It uses synthetic QKV
with real prefix needs and keeps all canvas tiles. This is a component test,
not an identical-real-QKV benchmark or accuracy panel; selector, KV-copy and
model costs are excluded. Natural Q64 and Q128 use the same alias2 consumer.
Warm kernels before interleaved timing; held lists must not rebuild in timing.

Isolated idle-host public package installations completed. LLaDA weights downloaded;
GPU smokes and model-specific adapters remain pending. Parent V18b stays frozen.

## V28 regroup screen and native qualification freeze — 2026-10-02 19:18 (UTC-5)

Real historical prefix-need snapshots: 144; natural Q64 work 794361 tiles.
Default bounded swap search accepted 7/144; gated load proxy ratio 0.9971768.
Expanded search (12 swaps/head, 16 candidate signatures/group) accepted 19/144;
ratio 0.9893759, total work +0.0402%. Aggregate max/p95 loads unchanged.
These are CPU prefix-only proxies, not GPU acceleration; no standalone regroup
request panel warranted. Source: `results/v28_20261002/regroup_screen/`.

Native q128/q64 request_clear qualification specs are frozen under
`results/v28_20261002/specs/`. Both use identical main parameters and source,
common 0.85 reservation, five GLOBAL layers, alias2 and lifecycle cleanup.
Each future preview uses first two E14 items per length bin x two repeats;
qualification first uses the longest selected item, one warm plus one timed.
Native dense/native-hook/all-kept controls retained. Wrapper records actual N,
untimed actual KV copy checks, effective q64 counters and allocator retry deltas.
30 CPU wrapper/lifecycle/adapter tests passed; 13 regroup and 3 component tests passed.

Component attempt 001 completed in 30.7184 reserved GPU seconds, but remains
diagnostic only: exact-length rather than nominal-bin sampling and non-native
Q strides were found during review. New attempt corrects these and strengthens
finite checks, disables TF32 for FP32 reference, and explicitly excludes first
list/split builds. Old run is preserved; no headline speed is taken from it.
LLaDA first upstream smoke loaded weights but failed RoPE JIT missing CCCL header;
isolated toolchain repair ongoing. Model sparse ports remain unimplemented.

## V28 native Q64 component result and request qualification — 2026-10-02 19:26 (UTC-5)

Corrected component002 passed finite IEEE-FP32 masked references on six historical
need snapshots with synthetic QKV, actual native tensor strides and random pages.
Q64/Q128 alias2 GPU-time geometric mean = 0.94575 (about 5.4% component reduction).
Selector, KV copies, first lists/splits and other model costs excluded; no request
or accuracy claim. Source: `results/v28_20261002/q64_component/`. Worker time
18.6512 GPU s; prior diagnostic001 30.7184 s. Dense component key corrected to
fixed-length diagnostic: it is not the official varlen serving baseline.

Native request qualification002 runs on mpk, frozen deploy e7b5ff061, one worker
at a time. Q64 method passed: actual N=66 (65 retired+1 unused), 330 GLOBAL calls,
60 q64-refined routes/list builds, five GLOBAL warm copy checks passed; no timed
CUDA captures/backend/inductor compiles or allocator retries/OOMs. Warm long
requests did show recoverable allocation retries, so request_clear does not
eliminate canvas allocation peaks. These one-request checks are not performance
or accuracy evidence. Main128, allkept, native-hook and dense follow serially.
Environment used OMP_NUM_THREADS=4; future performance preview should freeze 1
for all arms, following official serving warning. Do not compare qualifications
as a strongest-baseline speed panel. Further Triton/CuTe compilation coverage is
being audited independently from existing CUDA-graph/backend/inductor counters.

Held-regroup optimistic ablation now implemented with 8 CPU tests: natural Q64
versus token-major per-head gather, alias2 and scatter; at most one CPU-accepted
state per nominal bin. Search/first builds are excluded deliberately to ask whether
amortization could pay even in this favorable setting. No GPU result yet.
LLaDA CUDA13 closure mismatch identified (runtime13.4 headers with nvcc13.0);
isolated CPU small-kernel compile now passes with matching headers. No sparse
model port or qualified new-model accuracy result yet.

## V28 seed expansion, held-regroup negative and canvas release — 2026-10-02 19:44 (UTC-5)

User explicitly reiterated large seed-dependent denoising-step variance. New
seed4 protocols freeze four independent engine seeds (28001–28004), two sequential
repeat labels per engine, six question clusters: 48 requests/arm. Repeats are not
extra independent seeds; same engine seed is not a matched random trajectory.
Keep per-seed N distributions, paired geometric W/S/SN/N and correctness, with
question-cluster intervals. This remains preview; expand question coverage for
request-level claims. Existing qualification002/V18b protocols are untouched.
Common OMP_NUM_THREADS=1; new requests record pre-deduplication monitor Triton/CuTe
JIT event deltas as well as CUDA capture/backend/Inductor counters. Nonzero/missing
timed monitor receipts fail the new run. Events can include cache loads/failed
compiles, exclude autotuning/other workers/earlier aliases; zero is not universal
absence of compilation. Existing log warnings use warning_once, so the original
panel's zero post-warm warnings are only a lower-bound audit. Its block-0 main
had 30 recoverable post-warm allocation failures, retained in W.

Held-regroup component003 completed: natural-Q64-relative total times
1.04498/1.00462/1.03343 at nominal32/64/96K, geomean1.02753, even excluding all
search/initial-build cost. Three accepted states only; reject this unfused held
implementation for request expansion, not all grouping/fusion hypotheses.
Source: `results/v28_20261002/regroup_held/`; 45.8499 reserved GPU seconds.

All five qualification002 request arms passed (q64/main128/allkept/native/dense),
with existing graph/backend/Inductor checks and intended-path receipts. One item
per arm is implementation qualification, not performance or quality evidence.
A q64 alias1/2/4 component sweep is now specified and CPU-tested; independent
adapter per split avoids the split-cache identity-key pitfall. GPU run pending.

New standard optimization `canvas_buffers=release_after_invalidate` releases all
old contiguous adapter KV and prefix views only after successful encoder hooks
and same-CUDA-stream guards. Core observation/projection uses that stream;
async selector reads independent arrays with existing record_stream protection.
Carry and split maps are retained; no synchronization or empty_cache added.
Defaults remain legacy. Must qualify real release counters, numerics and allocator
behavior; no measured memory/performance claim yet. q128/q64 plus matched allkept
use the same option in separate frozen specifications. 92 relevant CPU tests pass.

LLaDA official SGLang JointThreshold dense smoke now passed, three toy checks,
forward count unavailable. This is environment qualification only; none of our
sparse variants are ported yet. Source: `results/v28_20261002/llada_dense_smoke/`.
Six attempts reserved345.5495GPU seconds including failures/stops. I-DLM own-stack
CPU setup/toolchain qualified and first dense smoke is compiling/running; no
new-model sparse or benchmark result claimed.

## V28 qualification and four-seed family — 2026-10-02 20:04 (UTC-5)

Qualification004 closed five request workers on the same dcfdb8730 source and OMP1.
Q128 legacy/release each N72; Q64 legacy/release each N66; allkept release N81.
Timed monitored JIT events and allocator retry/OOM deltas were zero. Release
counters show5 canvas invalidations for each method (9,731,891,200 cumulative
bytes released, not peak memory saved). Process peaks were nearly unchanged;
no peak-memory or speed advantage is established. Reserved cost867.9727GPU seconds
includes the14.3026-second alias sweep. Qualifying one item does not establish accuracy.

The six-arm family spec freezes dense, native-hook, allkept release, main legacy
canvas, main release and Q64 release. All use request_clear and shared generation
source. Four engine seeds28001–28004, two sequential repeats each, six questions:
48timed/variant; 288timed+144warm in24fresh engines. Variant order reverses in
alternating blocks. Paired W/S/P/SN/N and correctness use question-cluster intervals
and per-engine-seed step distributions. These are preview data, not a powered
noninferiority test. Family spec: `results/v28_20261002/specs/v28_seed4_family.json`.

Alias1/2/4 initial sweep uses three historical need states and synthetic QKV.
Alias1/2 ratio1.29397; alias4/2 ratio0.99264 with 64K slower1.01820 and96K faster0.96739.
Keep alias2; extend to all144 correlated snapshots with descriptive per-bin
aggregates, not question-level CI or request speed claims.

I-DLM chat follow-up executed successfully but all3calls hit512tokens despite
answer-presence checks3/3. Source and CPU EOS checks found no obvious stop-set
mismatch; actual output was not retained so repetition/think-closure diagnostics
are unavailable. Baseline stopping/quality remains unqualified. All new-model
attempts reserved723.0345GPU seconds. No sparse ports implemented; model-specific
clock, causal mask, rollback and all-kept controls are prerequisites. See
`docs/NEW_MODEL_PORT_AUDIT_V28_20261002.md` and
`results/v28_20261002/idlm_chat_smoke_followup/`.

## V28 preview launched and strict summary ready — 2026-10-02 20:14 (UTC-5)

seed4_preview001 launched on mpk from unchanged dcfdb8730 generation deployment,
using the six-variant family spec committed at8c9cd31ae. Four independent engines
per variant, shared seeds28001–28004, six questions, two sequential repeats per
engine:288timed+144warm total. One GPU worker at a time; immutable new run directories.
The full144 alias component run ended and released the GPU before preview launch.
Alias4/2 mean0.98836, with32K slower1.01731,64K0.98315,96K0.96533; keep alias2.
Reserved component cost27.0913GPU seconds. No request/accuracy claim from this sweep.

The new CPU summary validates all24workers and288records, exact bound source/input
bytes, protocol/config/host/software/seed identities, intended execution counters,
actual N, zero observed timed monitor JIT events and CUDA captures, and private
completion joins before unchanged NeMo scoring. Outputs contain aggregates only,
including W/S/P/SN/N, correctness, by-seed N distributions and length-stratified
question-cluster95%intervals. The main_legacy/dense extra comparison is labelled
descriptive; other seven comparisons are frozen family pairs.17CPU tests passed.
Qualification004 release/legacy private output equality holds for each Q128/Q64
pair (one item each); this is a limited numerical control, not model quality evidence.

CPU-only new-model event prototypes and12tests cover absolute-position identity,
prefix/layout epochs, edits including A-to-B-to-A, request reset and rollback.
They install no native hook, measure no forwards, and implement no sparse attention.
Unchanged token IDs do not prove unchanged hidden states/QKV or safe support reuse.
New-model adaptation audit states all unimplemented GPU/mask/KV/stream boundaries.

I-DLM diagnostic003 preparation: unchanged official chat/thinking/sampler,4096
output cap supported by its README, three toy calls in one engine seed. Retain
outputs privately so closure/repetition/EOS diagnostics can be measured; no
early stopping heuristic or sparse method, no benchmark-quality inference.

## I-DLM diagnostic003 launched — 2026-10-02 20:23 (UTC-5)

Frozen private spec verified byte-for-byte before launch on idle dlm2. One
engine seed0, three sequential toy requests, official chat/thinking/sampler
unchanged, max4096. Raw Engine text/output IDs remain in private own directories;
only scalar stop/closure/repetition diagnostics will be published. An initial
prelaunch newline-transfer mismatch was caught before GPU launch (0GPU seconds);
the failed prelaunch directory is preserved. This is baseline diagnosis only.

D2H metadata audit: existing CPU scheduler/sample-count fields cannot replace
the full exact phase/step/sequence-length tuple. CPU length is an upper bound;
next commit state and actual/retired execution identity differ. Async copy plus
a wait at prepare would merely relocate synchronization. No routing optimization
or shadow trace implemented; frozen preview unchanged. See
`docs/V28_CPU_METADATA_AUDIT_20261002.md`.

## I-DLM natural stopping diagnostic003 complete — 2026-10-02 20:33 (UTC-5)

All3official dense toy calls naturally stopped at EOS with closed thinking sections,
at868/549/1200 output tokens. Actual API IDs verify one EOS at the final position;
8gram excess repetition fractions0/0.00738/0.01006. This supports the earlier512
budget being too short for this toy; it is not a matched causal experiment because
later RNG trajectories change with earlier output length. Oneengine seed0, not3seeds.
Answer-substring presence does not establish exact final-answer or benchmark accuracy.
No sparse method or verified actual-forward count is implemented for this model.
Source: `results/v28_20261002/idlm_stopping_diagnostic_003/`. Reserved193.1752013GPU
seconds; I-DLM all attempts570.6601686; all new models916.2096868. Failed prelaunch
transfer check used0GPU seconds. Worker ended, GPU released, private output/statistic
recomputation and frozen-byte checks passed. Earlier reports remain unchanged.
