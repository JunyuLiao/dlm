# V31 refresh overhead and true-online V2 optimization

User-authorized continuation on one H100, rooted at verified repository commit
755b528c. Prior prototypes and frozen studies remain intact. This independent
study writes only under results/refresh_kernel_optimization_20261021 and adds
named implementations. It does not resume any paused campaign. Inherited Yuhan
V31 reuse and Junyu Gaussian32/value-direction Hopper code retain provenance.

V31 optimization preserves qblock_max prefix-mass normalization, fixed budget,
sticky multiplier, mandatory support, and current dense output. Remove unused
canvas-tail QK/softmax for this selector. Fuse normalization, row maximum,
top-k threshold, tie handling and sparse-list construction in Triton. A private
FA4 source copy observes the existing softmax probabilities after exponentiation
instead of recomputing per-tile softmax. It has a distinct subclass and auxiliary
tensor metadata to prevent collisions with inherited compile caches. The safe
candidate disables intra-workgroup overlap. Overlap-enabled observation fails
exact output checks and is not promoted. Compare explicit bidirectional and
dynamic-causal dispatch settings, plus split-KV controls; disclose any dispatch
or output-rounding differences instead of claiming equal execution by shape.

V2 uses native CUDA SM90 WGMMA and an ascending KV64 traversal. Each physical
Q128 decision combines its two query halves. Routing mass advances over every
block; normalized projected routing output advances only when kept. Separate
retained-only output normalizers produce current full-dimensional attention.
Rejected blocks execute no full PV and cause no output V loads; computing the
current Gaussian32 projection still loads full V and is charged separately.
Uniform fixed thresholds are frozen before timing after work-only synthetic
calibration. First support and all boundary/current-canvas blocks stay dense;
sticky refresh support is tested separately from initial selection.

Named CUDA alternatives: dedicated router/PV warpgroups with exact TF32x3 PZ;
BF16x3 PZ (three products with an omitted low-low term, separately qualified);
lean max/scaled-sum recurrence; TMA Q/K loading; compact consumers that duplicate
QK/PZ across output channel groups; and staged output that restores single QK
discovery. Finite QKV is a precondition for lean/compact/staged prototypes. No
custom routing state is used as model output. Compiler receipts preserve register
spills, failed dynamic-register settings, and original attempted sources.
Double-buffered cluster votes replace two per-tile rendezvous with one; the
next rendezvous protects the previous vote slot before reuse. A separate
conditional-P-staging candidate adds a rendezvous only on retained tiles and
does not support split-CTA roles. Qualification covers its supported schedules.

The local NVIDIA TensorRT-LLM reference uses double-buffered skip votes in
`fmha/warpspec/compute.h` and `kernel_traits.h`. BLASST decides before softmax
and BMM2; V2 must first complete normalized QK probabilities and projected PV.
Consequently the same fraction of rejected blocks saves less work here.
Official reference: https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/sparse-attention.md

Qualification: independent existing ordered selector, exact mask agreement,
projected-state/log-mass errors when exported, current FA4 on the identical
mask, partial Q/K, all-kept, multiple thresholds and sticky retention. Then
interleaved CUDA-event graph replay with capture outside timing and an actual
capture counter; charge projection, RMS, list construction and all execution.
Report core-only timing separately. No native task accuracy or request claim
follows from synthetic timing. Native HumanEval and the earlier conditional
AIME/LongBench expansion remain unlaunched until a qualified online kernel
meets the performance gate. Source snapshots and hashes freeze evidence without
changing Git history, consistent with the previous automatic-review decision.

Resource handling follows the user's explicit preemption authorization: identify
current executable/argv/campaign/start-time, terminate only FastWAM workers and
verified dedicated supervisors, SIGTERM then rechecked SIGKILL as needed, record
the reclamation receipt. One borrower campaign started during a late screen;
that screen is contaminated and excluded. The MPS server was not signaled.
Check compute clients before every final shape. Nsight Compute hardware counters
are unavailable to this account; do not infer a measured stall breakdown.
