# V29 regroup review — 2026-10-02

Read-only review and prospective gates. No peer code copied or merged, no GPU run,
and no implemented fused consumer claimed. V28 source/results remain unchanged.

## Implementable scope after coordinator review

The single V29 prototype is now restricted to **fusing O scatter into alias2
LSE merge**. Q keeps the same Torch gather; FA4 Q loading/TMA and the adapter are
unchanged. Natural Q64 uses exactly the same fused merge with an identity row map.
Incremental regroup is `held_fused / natural_fused`, while natural fusion versus
standard merge is a separately reported generic implementation gain. The broader
ideas below are review context and deferred proposals, not GPU launch plans.

Prototype files: `experiments/numerical_qk_reuse/v29_lse_merge.py`,
`scripts/v29_regroup_merge_bench.py`, and `tests/test_v29_lse_merge.py`.
Eleven CPU tests pass using the bundled Python runtime; the system Python lacks
NumPy. CPU contract tests cover stable two-split LSE weighting, per-head original
row writeback, an empty split, malformed/nonfinite inputs, and immutable output
guarding. Torch/Triton imports are lazy. GPU compilation and numerical qualification
remain pending; this is not an implemented model port or measured acceleration.

The diagnostic script uses the unchanged historical held gate/order, original
paged alias2 calls and four rotated routes: natural standard, natural fused,
held standard plus scatter, held fused. All routes require finite full-consumer
IEEE FP32 masked oracles; mapped merges also require a Torch softmax/LSE oracle
before timing. CPU order reconstruction, map/list/split construction and first-use
compilation/oracles are separately recorded. Timed calls include Q gather, table
and used allocations, FA4, merge and writeback. Online signal generation/transfer
and a real validated map reuse lifetime are unmeasured, so no free online build
or amortized request result is claimed. No GPU launch or adapter change occurred.

## Broader review recommendation

Prioritize a bounded **consumer row-map feasibility** check, then a distinct
**CTA scheduling-only** candidate. Do not repeat the standalone pointwise
permutation negative with another launch configuration. Holding a permutation
amortizes its construction, but does not amortize copying fresh Q/O every forward.
The consumer row-map direction overlaps Haowei's regroup/BLASST interfaces and
must remain a collaboration candidate, not a claim of originating permutation.

## Audited local evidence

The local remote-tracking ref `origin/chw/value_aware` was inspected at
`e44c3375b4acdecbbbfc6f8544a84c94e727a8d2` (2026-09-30 09:21:20 UTC,
git author `coconight01`, subject `add grouping`). This is the locally available
ref, not a claim that it equals today's upstream head; no fetch was performed.
The repository context attributes regroup to Haowei.

| File at that pin | Evidence and consequence |
|---|---|
| `regroup/regroup_scheduler.py` | CPU regroup, pinned staging, side-stream H2D and event wait. Overlap requires a usable earlier signal and a real dependency/event test; merely putting work on a stream does not remove it. |
| `gemma/gemma/diffusion/_skip_regroup_jax.py` | Count/set sorting and prior-frame `q_ids` inversion; per-layer/head weight helpers. Frame identity and bitmap identity must travel together. |
| `gemma/gemma/diffusion/_sampler.py:1010` | Previous-step inputs and host/device pipeline controls are already explicit. Delayed history is theirs; current-step information cannot be called an earlier signal. |
| `jax_ffi/fa3_attn.py:207` | FA3 wrapper explicitly gathers Q by `q_ids` then reverses output order. A per-token absolute KV window accompanies reordered rows; tile exclusion alone cannot enforce a partial window. |
| `jax_ffi/skip_attn_ffi.cu:11` | BLASST FFI receives shared-across-heads `q_ids`; its interface declares original-order output from combine. The included `blasst_wgmma_kernel.cuh` implementation was not located in this tracked tree, so this audit verifies interface intent, not its device implementation or performance. |

`origin/ljy/value_aware` resolves locally to
`47c47d9d7b29662d359525c7d59053364c673845` (2026-09-18, JunyuLiao).
Only ref metadata was checked here; no new claim about its regroup implementation
or newer methods follows. The established value-aware selector and grouping work
remain attributed to their owners. No private traces, prompts or output text were
used for this review.

## Primary external evidence checked online

[PBS-Attn paper, v1](https://arxiv.org/html/2510.21270v1) proves query permutation
equivariance with inverse output mapping, and paired KV permutation invariance.
Its segmented causal strategy and main query-aware **key** permutation target
long-context AR prefill. Its query-permutation ablation notes GQA overhead. Those
results do not establish a win for 256-query diffusion decoding, head dimension
512, paged KV or native stopping. This is substantial prior art for permutation
as a sparsity improvement; V29 needs an implementation-specific contribution.

The [official PBS-Attn code](https://github.com/xinghaow99/pbs-attn) is available.
Its [permutation implementation](https://github.com/xinghaow99/pbs-attn/blob/main/pbs_attn/src/permute_states.py)
computes last-query-block key scores, sorts within segments, and uses Torch gather
to materialize permuted keys. It is a reference for semantics/cost inventory,
not evidence that arbitrary row indirection is already supported by our FA4.
The online `main` links are mutable; pin before any dependency or reproduction.

The [PyTorch FA4/FlexAttention engineering note](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/)
describes block-sparse scheduler metadata and longest-processing-time scheduling.
It also documents block-size limits and pointer-load costs. This motivates a
metadata-only scheduling experiment; score/mask-mod arbitrary loads do not imply
arbitrary Q tensor row-load support. Its Blackwell CLC mechanism cannot be assumed
available on H100. This is backend evidence, not a benchmark for our workload.
The [official FA4 forward source](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd.py)
is an upstream starting point; the actual deployed/patched pin must govern edits.

## Cost bound from our measured states

These are existing V28 held-state synthetic-QKV component measurements, not an
online model result. Natural Q64 and held consumer-only were on the same host.

| Nominal bin | Natural, ms | Held consumer only, ms | Saving, microseconds | Maximum saving / natural |
|---|---:|---:|---:|---:|
| 32K | 0.367840 | 0.347968 | 19.872 | 5.402% |
| 64K | 0.540400 | 0.516832 | 23.568 | 4.361% |
| 96K | 0.447552 | 0.430544 | 17.008 | 3.800% |

Source: `results/v28_20261002/regroup_held/summary.json`. The full held path was
1.027533 times natural in geometric mean. The later same-host standalone Triton
permutation diagnostic was 1.626493 times Torch; its component-call CUDA-event
cost includes possible dispatch waiting and is not isolated device bandwidth.
Do not add times measured on different hosts or assume isolated costs are additive.

For these fixed selections, an optimistic fused path needs
`extra_row_map_cost + online_build_cost / valid_reuses < consumer_saving`.
This is a measured consumer-only ceiling for these states, not a universal bound
on a different kernel's performance. Request gain is smaller: if this component
occupies fraction f of request time and fractional saving s, the ideal fixed-step
request speedup is `1 / (1 - f*s)`. Native extra steps reduce or reverse it.

With Q=256, H=16, D=512 and BF16, each Q or O tensor is 4 MiB. Materialized
gather plus scatter adds roughly 16 MiB of temporary read/write traffic beyond
the necessary Q read/O write. A per-head int32 row map is 16 KiB. Removing copies
does not remove noncontiguous accesses or prove enough speed; bandwidth-derived
limits are not measured timings and cache effects can change traffic.

## Candidate A: row-map inside consumer — collaboration candidate

For grouped slot s and head h, load post-RoPE Q from original row `P[h,s]` into
the consumer's shared-memory tile and write O to that original row. Keep KV
physical pages and pairing unchanged. There is no global reordered Q/O tensor.
Map the query row's absolute position for causal/window masks, full-canvas
protection, and partial-tile predicates. Alias2 partial output and LSE must use a
consistent frame: either combine in grouped slots and scatter in the existing
combine epilogue, or write partials by original row and combine naturally.

The difficult part is the actual FA4 loader: an arbitrary per-row map is not a
constant-stride rectangular TMA transfer. A new vectorized per-row load/cp.async
path may lose overlap or coalescing; a faster standalone gather is not a solution
to that loader problem. D=512 resource use, tile64, native stride, paged KV and
the deployed alias split/merge need qualification. Changing to a weaker Triton
attention kernel solely to support a map does not establish an incremental win
against official FA4. Start with an identity-map control on the same consumer.

Required gate: CPU bijection/frame/mask/support tests, then identity and mapped
all-kept GPU oracles, identical-state masked FP32 oracle, then matched fused and
unfused totals including online build, dispatch, allocation and merge costs. Only
a positive same-state total advances to request quality/native-step validation.
This note supplies no speculative patch or hook claimed to work.

## Candidate B: reorder CTA work only — no tensor permutation

Keep natural 64-row groups, exact support and original Q/O locations. Build a
work descriptor permutation over `(layer, head, q_tile, alias_split)` by retained
KV count or a calibrated tile-cost estimate; dispatch longer work first in a
compatible scheduler. No row grouping changes, gather, scatter or altered masks.
This can reduce a scheduling tail, not block count. Compare descriptors' mapping
multiset exactly; retain original output index in every descriptor. Whole-tile
reordering cannot reduce union support and must not be advertised as regroup
sparsity. Amortize only metadata on states with a proven validity lifetime.

First measure whether the pinned consumer actually exposes schedule order and
whether load imbalance is material. A regular CUDA grid is not guaranteed to
execute a supplied order as LPT; a persistent scheduler or explicit rank map may
be required. Public FlexAttention schedule metadata is an existence example,
not proof that our alias2 kernel exposes it. Freeze only after this is resolved.

## Deferred candidate: fuse into existing projection/layout operation

If an existing Q projection/RoPE or output layout kernel already writes the
needed representation, add a mapped destination/source there rather than a new
permutation launch. Per-head orders make output projection especially awkward:
different heads at a logical token can come from different grouped slots, so a
standard contiguous GEMM cannot consume the grouped buffer directly. Moving a
scatter into a separate staging operation is not fusion. This is lower priority
until a concrete deployed operation can absorb it without losing its fast path.

## Fairness and stopping gates

Reuse the same private state, physical page permutation, dtype, original-position
mask, support coverage, warm-ups, host and source pin for every arm. Required
controls are natural Q64, identity-map fused Q64, mapped fused Q64, existing
unfused held, mapped/all-kept, and strong official dense for eventual requests.
Keep selector/model algorithm unchanged while measuring a consumer intervention.
Historical dump selection is an upper-limit diagnostic, not fresh online quality.
Do not reuse stale need/order across changed tokens, canvas epochs, boundary
movement or rollback without the corresponding validity/coverage check.

Record kernel/body/reservation timing separately and real compile/capture counts.
Any request panel includes first online construction, per-layer map lifetime,
native forward/step changes and accuracy with question-cluster inference. A small
component gain is acceptable; it still needs a positive total and a correctly
qualified strong baseline. Stop a candidate at its failed gate rather than tuning
the frozen negative or removing costs from the reporting boundary.
