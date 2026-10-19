# Fixed BLAS workspace as a diagnostic factor

Native operator definedness attempt002 passes all four checks, including the
NaN-prefilled FA4 observer and native H16/HK2/D512 selector scratch poisoning,
canaries and growth/shrink reuse. Attempt001 was a controller binding failure
before a worker started; it is preserved with a conservative36.299s wall charge.
Passing this bounded screen does not rule out other memory or ordering faults.

Remaining observed differences arise in raw model logits before the affected
sampler update. Current Inductor kernels report deterministic settings, but
current native workers have no CUBLAS_WORKSPACE_CONFIG setting. Model and
selector paths include actual CUDA BLAS matmuls outside that Inductor reduction
policy. Fixed BLAS workspace is therefore a justified diagnostic factor, not
an established cause. Earlier pre-correction BLAS-only negatives are retained;
they do not isolate the residual variation after the Inductor correction.

Freeze a new attempt005 with exactly one inference configuration change:
CUBLAS_WORKSPACE_CONFIG=:4096:8 at process startup. Keep model/helper compiler
options, Torch deterministic-algorithm setting, memory-fill setting, native
FULL/PIECEWISE modes, methods, budgets, thresholds, seeds and native formulas
unchanged. Extend only the exit receipt with the preferred BLAS backend and
configured cuBLAS/cuBLASLt workspace bytes. Require the configured cuBLAS size
to be33554432 bytes; an environment-variable claim alone is insufficient.

Run the same two independent clean full-cohort diagnostic engines as attempt002:
80 observed warmup/qualification requests followed by72 unobserved repeats,
152 per engine. No attention-level probe, raw capture, scoring or accuracy-based
selection. Compare every seed/method trace and within-engine observation check,
including failures. Require real logit fingerprints on every observed call.
All original source/protocol and failed attempts stay intact.

Matching bounded outputs does not prove BLAS caused the original differences.
Do not promote based only on a favorable final answer. Check effective settings,
complete tensor-state trajectories and the supported library guarantee; a
native causal-operator claim still requires actual tensor comparison/replay.
This is not a new selector or an eager fallback. No full accuracy matrix is
released by this diagnostic controller.

Two1000-second reservations charge existing phasek256 without expanding its
16200s cap or total43200s. Use gpu_budget.json for actual remaining charges,
including the conservative controller-failure charge. Both matrices and the
K128 followup remain held. Any candidate clean-performance claim needs matched
controls under its final frozen numerical policy.
