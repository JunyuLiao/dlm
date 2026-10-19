# HumanEval reproducibility investigation — active, unqualified

Original K256 and both older paused LongBench campaigns are unchanged. The
full corrected K256 and new K128 matrices remain gated; user authorization
persists, but reproducibility qualification has not passed.

Frozen starting checkpoint63ccf1e6; plan c467882a; native snapshot20e6c0be.
Every probe has its own source hash, environment, exact launch and immutable
attempt marker. Six questions0/23/129/106/145/116 were chosen using lengths and
original trajectory tails, never correctness. Native decoding is unchanged.

Verified findings:
- Independent unobserved FULL and PIECEWISE clean repeats each change all12
  measured output pairs (six questions, twice), with matched prompt/seed.
- Same-engine clean/sync/geometry/finite/audit/clean groups agree on all six.
  Cross-process comparisons alone cannot establish instrumentation causality.
- Two post-load fingerprints match all900 actual parameters and187 model
  buffers (51.676GB of parameters), yet their12 output pairs differ.
- Native FA4 probes find zero dynamic-causal specialization-context collisions.
- Deferred observation has identical clean/observed/clean groups in independent
  engines. Across both observed processes, all six valid prefill/initial-denoise
  token IDs, positions, embeddings and default CPU/CUDA RNG states agree, but
  first denoising logits differ. Q0 full-logit mean/max absolute differences
  are0.740/9.923, with20/256 row argmax changes. Physical page IDs differ, while
  logical block-table/slot-map consistency checks pass.
- Input padding is not a sufficient explanation: question23 has no unused
  rows; question129 has identical unused rows; both still diverge. The frozen
  padding probe is currently unstarted.
- Four cuBLAS4096:8 workspace tests and the two small-cuBLASLt-workspace tests
  remain nonrepeatable. V31 clean independent repeats also change all12 output
  pairs. No numerical policy is promoted as a fix.
- Native attention observation is inert in both engines' six question groups.
  Layer0 prefill Q/K/V/output and logical cached K/V match exactly across
  processes. Layer1 Q/K/V first differ (Q397/700416 elements, max0.03125);
  the interval after layer0 attention and through layer1 Q/K/V remains under
  investigation. This does not yet identify a particular GEMM or MoE operator.
- Disabling BF16 split-K using Torch's tuple flag with the existing cuBLAS
  backend fails during initialization, before generation. The failed attempt
  is retained. A separate supported boolean accumulation-precision contrast
  retains the backend and split-K policy. Its results are pending.

Next: observe completed kernels in the unchanged first compiled prefill
segment, then qualify any correction on V31/dense and both thinking settings.
The observer includes selected Triton configurations and clean bookends;
its focused CPU regression passes. Non-thinking diagnostics remain pending.
No selector improvement or accuracy ranking is established. Raw activations,
RNG bytes and completions are private. Sanitized comparisons and independently
rescored diagnostic records are under analysis/.

The allocation addendum preserves the43200s overall ceiling: diagnosis10800s,
K25616200s and K12816200s. Read allocation_addendum.json, gpu_budget.json and
state.json. HOLD_AFTER_CURRENT.json is an internal diagnostic-queue hold, not
a user pause or a change to either older paused campaign.
