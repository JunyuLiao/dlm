# V2 online refresh: key-partitioned two-warpgroup pipeline

Continuation authorized by the user on 2026-10-21, base755b528c. New campaign
`results/v2_online_pipeline_20261021/`; earlier studies stay immutable.
Preserve Yuhan reuse and Junyu Gaussian32/Hopper provenance. Native sampler,
held-mask consumer, GLOBAL scope, Q128/KV64 gate and ordered threshold semantics
remain fixed. This phase starts with synthetic kernel qualification and timing.

Design: two compute warpgroups per Q64 CTA partition QK's KV columns, then each
owns half of full output channels. They share full-block normalization and
projected-PV input probabilities. Each owns 16 sketch dimensions for PZ/state;
row-risk partial sums are combined before the two Q64 CTAs vote on Q128.
Remove the dedicated router warpgroup. Reuse dead QK shared storage for PZ
staging. Use a retained-only maximum/scaled-sum output accumulator and normalize
once in the epilogue. Separate optional early V prefetch and next-K prefetch
from mandatory conditional full-PV execution. A fixed scalar is calibrated to
synthetic work before timing; no change to selection order or mandatory support.

BLASST reference: NVIDIA TensorRT-LLM read-only fmha/warpspec/compute.h and
kernel_traits.h; see the official skip-softmax tech blog. Borrow register/role
organization and double-buffered votes, not its earlier max-only gate. Our gate
still needs full QK softmax and Gaussian32 projected PV. Original BF16 V and
FP32 sketch are retained; a BF16x3 PZ numerical variant is separately qualified.

Required checks: independent ordered-selector mask/state/log-mass comparison,
FA4 same-mask output, finite/extreme logits, partial Q/K, sticky and all-kept;
physical keep counts; warmed interleaved CUDA-event graph timing with actual
capture counter0; fresh projection/RMS and list construction charged separately
from core. Strong plain/dynamic FA4 and all-kept custom controls are retained.
No native accuracy, trajectory or end-to-end claim follows from synthetic work.
Freeze source by hashes/snapshot before promotion; preserve failed attempts.
Resource checks before GPU runs; authorized borrower reclamation only under
AGENTS.md identity and signal rules. No previous paused campaign is resumed.


Implemented candidates now also include full-QK WG0 with WG1 Z preparation,
shared-storage-separated K prefetch, fused local max/sum rescaling, leader-only
acquire/release epoch exchange, TMA Q/K loads, and retained-only TMA V loading
with K-first outer tiling of the MN-major V layout. The four-KV wavefront keeps
ordered routing through distributed shared state and reduces full-PV partials
in a separate kernel; both costs are charged and its slower screen is retained.
The legacy builder option tma3d currently emits eight 2D TMA transactions; the
initial nonmonotone-stride 3D descriptor was not promoted. Report actual code.

A GR00T profiling campaign appeared intermittently. Runs with a foreign client
before/after timing are excluded. The user explicitly extended preemption scope
to this GR00T campaign on 2026-10-21 (gr00t_n15_venv, sprof_1020). Re-identify
live workers and dedicated supervisors by executable, arguments, cwd, PID and
start time; use SIGTERM first and retain a resource receipt. Historical PIDs are
not targets. Correctness runs sharing the GPU are not clean timing evidence.
No foreign runtime or environment is modified.

Additional isolated controls include double-buffered QK/PV overlap, routing-state
stashing, base-2 softmax, full-sketch PZ duplication, duplicated QK per WG, and
approximate division (--prec-div=false without global fast math). The channel-
split candidate uses one WG per CTA and two independent output-channel groups,
doubling CTA count at Q256/H16 while preserving the Q128 gate and full ordered
Gaussian32 state in each group. Its duplicated QK and projected PV are charged.
Each numerical variant must pass the same selector and same-mask output checks;
a synthetic comparison does not qualify native accuracy.

Channel-pair variants use two WGs per Q64 CTA, each owning 128 output channels,
with two independent 256-channel groups. QK and routing are duplicated across
these groups, but full PV covers each output channel once. This yields 128 CTAs
at Q256/H16 and reduces accumulator registers. The serial version avoids the
spills of QK/PV overlap. The channel_router version assigns full Gaussian32 PZ
and risk to WG0 and removes the inter-WG partial-risk barrier. It retains full
QK once per channel group, ascending state updates, Q128 decisions and
retained-only V TMA/full PV. Approximate division is explicit in its build.
The duplicated-QK-per-WG control did not improve performance.

The experimental OnlineRefresh entry point charges fresh projection/RMS, online
output and reusable-list compaction. It returns sparse output on refresh;
published selector-only V2 returned a dense observation output on refresh.
Consequently this is a named experimental output variant and requires native
qualification rather than inheriting old V2 accuracy. No native default or
held-mask consumer is replaced. Synthetic thresholds here are calibrated per
geometry to about 50 percent GLOBAL tile sparsity, frozen before timing. They
are not first-canvas HumanEval/AIME/LongBench calibration results.

A final reduced-unrolling overlap control removed compiler spills in both the
channel-pair/full-sketch-router pipeline and full-output pipeline. Both remained
slower than the selected serial channel-router design. The compiler reported
WGMMA accumulator serialization for the reduced-unrolling router variant.
This is compiler evidence, not hardware-counter attribution. Keep these screens
as negatives; zero spills alone does not establish an effective pipeline.
