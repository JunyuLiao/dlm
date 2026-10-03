# FA4 baseline and fused-observation audit — 2026-10-02

Scope: V29 source at `26d6b10dd`, inherited frozen main and the opt-in copy/merge
implementations. This is a source/test audit, not a new speed or accuracy result.

## Which paths actually run

| Arm/path | Attention implementation | Interpretation |
|---|---|---|
| Default dense | vLLM native FA4; GLOBAL uses dynamic-causal dispatch and effective split-KV | Headline system reference, including its default graph mode |
| Matched native | The same native attention forward, with adapter hooks and PIECEWISE execution | Measures the effect of those execution conditions; no sparse selection |
| Adapter all-kept | FA4 block-sparse interface with every legal block retained, paged alias splitting and LSE merge | Controls the adapter consumer/copy machinery |
| Main, sparse output calls | Same paged FA4 consumer, with selected support | Sparse QK and full-dimensional V output over retained blocks |
| Main, fused observation call | Custom Triton dense-output-plus-summary kernel | Not a native FA4 call with a free statistics hook |

All arms must share the pinned installed FA4 implementation, including the
disclosed SM90 paged sparse-read correction. This does not mean their dispatch,
split mechanism, numerical summation order or graph modes are identical. Keep
all references; do not weaken native dense to improve a reported ratio.

`v27_fa4.py` retains an old HF-context docstring describing all-kept as its
fastest configuration. That wording is not a current serving-baseline claim.
The superseding measured native dynamic-causal evidence is in
`VLLM_PORT_NOTES_20261002.md`, section on split-KV. Historical HF speedups do
not establish the same speedup against current native vLLM.

## QK reuse is present

`integration.py::_fused_bootstrap_observation` calls
`v27_consumer64.py::fused_observe` with `mu_precision='bf16'`. In main's normal
observation mode, each visited score tile is computed once as QK, and its
exponentials are shared by:

1. Full-dimensional V multiplication and online-softmax output accumulation.
2. The rank-32 projected-V weighted summary used for later routing.

There is no second QK pass for these observation statistics. Projected V is
used for selection statistics; it does not replace the model's full V output.
The separate named `observe_carried` path must not be conflated with main.

Sharing QK does not prove negligible incremental time. The custom kernel also
writes block summaries and a score tail; later DP selection reads the summaries.
`cached_executor.py::allocate_summary` stores mu in FP32. For 16 query heads,
256 queries, 32,768 prefix tokens, KV tiles of 64 and rank 32, mu alone occupies
`16 * 256 * 512 * 32 * 4 = 268435456` bytes (256 MiB) per GLOBAL layer.
This is allocated summary size, not a measured bandwidth bottleneck or a
critical-path cost estimate. Other summaries, projections, copies and merges
also exist. Observation occurs once per canvas, not on every denoising call.

## Correctness evidence and remaining checks

The current source preserves full-V output, the model's scale, GQA head mapping,
finite-output checks and asynchronous event dependencies for route consumption.
Installed official Gemma GLOBAL scale is 1.0 (with its QK normalization); the
adapter passes `impl.scale`, rather than imposing head_dim**-0.5. The installed
FA4 returns natural-log LSE, consistent with the adapter's softmax-weighted
FP32 merge. Alias list boundaries floor(count*i/splits) partition the retained
list without overlap. Current configuration has zero attention softcap and no
extra ALiBi/sink mask. These facts do not qualify reuse for arbitrary future
configurations: the adapter does not yet reject every unsupported extra feature.
The native hook forwards to the original native attention function. Main and
all-kept obtain K/V from the native paged cache; prefix and canvas copy paths are
separate. Only the configured GLOBAL decoder layers are intercepted; prefill,
encoder commits and LOCAL attention retain the native route.

The local constructor/adapter, paged-copy, mapped-merge and profiler suites pass
42 CPU checks in this audit. They do not execute CUDA attention. Historical
GPU paged/alias-output checks and the V29 copy/merge component oracles are
recorded separately; the latter do not qualify a new full-model deployment.
No blanket proof of numerical correctness or task noninferiority follows from
these checks. Explicit GPU checks of the actual BF16 observation-summary mode,
including scale1.0, are prepared because the original fused-observation unit
test uses its default TF32x3 summary mode.

An independent profiling-tool defect was identified: its decoder-layer name
regex fails to classify the installed native layer names. This causes missing
GLOBAL/LOCAL attribution, not a demonstrated model-routing error; the adapter
uses its own layer matching. Preserve the frozen diagnostic and rerun under a
new source/binding after fixing and testing the classifier. Missing attribution
must never be reported as zero attention cost. CUDA memcpy API CPU duration can
include an implicit GPU wait; it is not pure DMA or enqueue overhead.

## Fair interpretation of improvements

Copy fusion and LSE-merge fusion are ordinary implementation improvements and
must apply to both eligible all-kept and main arms. A regroup claim requires an
equally fused natural-order reference, including Q gathering and output scatter.
The existing V29 matched regroup component is approximately tied; do not label
the shared merge improvement as a regroup gain. Any request-level improvement
still needs native dense, matched controls, intended-path receipts, full warm-up,
zero timed captures/compilations and question-cluster uncertainty.
