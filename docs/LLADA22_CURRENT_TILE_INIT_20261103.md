# LLaDA2.2 HumanEval current-tile initialization — 2026-11-03

User-authorized implementation, six-prompt recalibration and full164 HumanEval
evaluation of `value_v2_current_tile_init_gaussian32_reuse_r4`. Parent evidence:
`results/llada22_humaneval_sparsity_audit_20261103/attempt001/` and the direct
EOS/support diagnostic. This is a new exploratory variant of Junyu's Gaussian32
selector using Yuhan's inherited map-reuse machinery, not a modification of
the original V2 result or the discarded EOS-confidence intervention.

Initialize online log mass, normalized projected output and support from the
last KV64 tile, which is the only mandatory tile for aligned LLaDA32 canvases.
Then process tile0, tile1, ..., tile J-2 in chronological order. Each tile is
consumed once. Keep the V2 mass-on-skip rule and normalized-output-on-skip rule.
There is no first-chronological-tile protection. Threshold decisions remain
online and unconstrained by a top-k repair. Fail closed on invalid statistics.

Use the original224f8f11 numerical snapshot, model revision
e4f77d25ab5ca86f2223c6cdc6a72b887e20c8a5 and original sampler without edits.
Only replace its online selector function. Reuse the existing optimized
128-tile chunk kernel after initializing the state; use unchanged Gaussian32
projection, Q128/KV64 discovery and full-D128 FA4 sparse attention consumer.
All20 layers,16 query heads, refresh every fourth call, reset on canvas and
structural changes, no cross-canvas carry. Prefill/commit attention stays dense.
Native block32, steps32, M2T threshold0.5, T2T threshold0, poststeps16,
temperature0, EOS stopping, cap1024. No EOS status or confidence modification.

Before calibration, require four meaningful CPU tests,30 CUDA-versus-independent
CPU recurrence cases, sparse consumer agreement with FP32 masked attention,
and12 saved-control replays: first four items each for original V2,V31 and
all-kept FA4, with exact output hashes and forward/work counts. Any parity
failure stops execution and preserves the attempt.

Recalibrate only on the same six handwritten context-matched development code
prompts, seed142, cap1024. Logarithmic bisection in[0.01,4], at most8 candidates,
stop at pooled48–52% tile sparsity; otherwise choose closest and disclose.
Repeat just those six prompts to confirm the frozen selected threshold.
Calibration uses no functional score or benchmark examples. Run the complete
164 HumanEval problems once, seed42, four41-problem resumable evidence shards.
No post-benchmark recalibration even if sparsity drifts. A killed worker gets a
new attempt directory. Record nvidia-smi every900seconds, one H100 worker.

Compare against saved original native dense, all-kept FA4,V31,V2,V3a, with
source/host/prompt/seed/cap/scorer identities audited. Official functional
scoring and10,000 problem-bootstrap paired intervals. Report accuracy,N,C,N/C,
output length,EOS/cap counts,syntax/completeness,geometric tile sparsity
`1-sum(kept)/sum(eligible)` including mandatory tiles and excluding discovery,
prefill and commit; no clean latency/speedup claim. Repeated HumanEval is
exploratory and previously examined, not held-out confirmation. Freeze source
hashes and protocol before GPU generation. Preserve raw tasks/outputs privately.
