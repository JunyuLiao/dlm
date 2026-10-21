# Sparse execution and refresh optimization, 2026-10-21

Explicit user instruction pauses results/optimized_refresh_benchmarks_20261021/
attempt004. Keep its immutable partial records and PAUSED_USER_REQUEST marker.
New benchmark short budgets: AIME26/MATH500/HumanEval all K128; LongBench K8192.
No native evaluation relaunch until synthetic and native qualification pass.
No old K1728 threshold inherits equivalent computation work.

Optimize resident held-mask execution first: refresh-time compact indices and
split metadata; persistent device page-table/used lengths; direct paged K/V;
no selector, mask decisions, list packing/splitting or prefix copy on held
steps. Compare compiled FlexAttention BlockMask against native FA4 sparse.
Support current canvas changes without stale values. Keep addresses stable
within static context buckets and warm compilation before clean timing.
All consumers execute current Q/K/V; a held map never caches attention output.

Synthetic scope: B1/Q256/H16/Hkv2/D512 BF16, keys1024/4096/8192/16384, realized
GLOBAL physical tile sparsity40/50/60 percent, mandatory canvas tiles retained.
Compare native dense FA4, all-kept FA4, previous paged split consumer, resident
paged FA4, FlexAttention, optimized V31 refresh, true-online V2 and diagnostic
two-pass. Charge actual metadata/copy/launch/merge operations. Separate clean
graph/event and host/module timing from selector/phase audits. Count actual
CUDA captures and test changed Q/K/V/page tables/masks on graph replay.

V2 refresh remains one ordered online QK/softmax/projectedPV/PV traversal;
reject full PV before execution. Investigate unique router handoff, repeated
QK, registers/WGMMA scheduling/barriers, currentV projection/packing and masks.
V31 uses FA4 online softmax observation, fused topk/list. Record negative
variants and compiler resource evidence without inventing hardware stall shares.

Before relaunch require finite/parity/budget/extra-forward/source identity,
changing-input graph and native audit/clean trajectory checks; dense LOCAL,
prefill/commit and native sampler fixed. Target held steps faster than native
dense after all execution overhead at40-50 percent realized sparsity. Target
V2 held sparsity40-50 percent per workload from actual eligible/kept receipts;
report initial/refresh/held/carried/first-canvas/whole request separately.

SparseD (INV-WZQ, ICLR2026) is an attributed execution design reference;
its early pattern + compiled FlexAttention reuse does not define our selector.
Inherited Yuhan cross-step reuse and Junyu Gaussian32 stay separately named.
