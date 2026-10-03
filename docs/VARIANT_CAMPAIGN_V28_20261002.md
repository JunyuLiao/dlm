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
