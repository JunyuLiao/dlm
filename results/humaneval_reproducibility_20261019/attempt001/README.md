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
- Deferred observation has identical clean/observed/clean groups in the first
  engine. Across both observed processes, all six valid prefill/initial-denoise
  token IDs, positions, embeddings and default CPU/CUDA RNG states agree, but
  first denoising logits differ. Q0 full-logit mean/max absolute differences
  are0.740/9.923, with20/256 row argmax changes. Physical page IDs differ, while
  logical block-table/slot-map consistency checks pass.
- Input padding is not a sufficient explanation: question23 has no unused
  rows; question129 has identical unused rows; both still diverge. The frozen
  padding probe is currently unstarted.
- Four cuBLAS4096:8 workspace tests remain nonrepeatable. A separate cuBLASLt
  workspace contrast is running. No numerical policy is promoted as a fix.

Next: compare native per-layer prefill Q/K/V/output and logical cache states,
qualify V31 and non-thinking paths, identify and test the smallest correction.
No selector improvement or accuracy ranking is established. Raw activations,
RNG bytes and completions are private. Sanitized comparisons and independently
rescored diagnostic records are under analysis/.

The allocation addendum preserves the43200s overall ceiling: diagnosis10800s,
K25616200s and K12816200s. Read allocation_addendum.json, gpu_budget.json and
state.json. HOLD_AFTER_CURRENT.json is an internal diagnostic-queue hold, not
a user pause or a change to either older paused campaign.
