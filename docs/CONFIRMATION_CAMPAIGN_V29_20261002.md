# V29: expanded fairness checks and small implementation gains

User authorization: 2026-10-02, US Central UTC-5. Expand questions and seeds;
protect AIME/HumanEval quality; explore worthwhile variants including regroup.
Branch `research/vllm-confirmation-20261002`, forked from V28 `e5e1ffcc8`.
V28 generation and its completion/publication coordinator remain untouched.

## Questions and interpretation

W is request-boundary wall time including prefill and adapter setup/cleanup,
excluding engine loading and HTTP transport. S is decode span. N counts actual
denoising forwards including unused speculative execution; commits are separate.
S/N is amortized cost, not isolated attention kernel time. All speed ratios are
candidate/reference; below one means less time.

Main is the frozen M3 R6 A64 dense-prefix-risk configuration, minus-ln2 threshold,
asynchronous route selection, fused once-per-canvas observation, carry_first,
rank32 V projection, current query temporal sensitivity, Q128 and alias2 FA4.
Only five GLOBAL layers are sparsified. LOCAL, native sampling, thinking and
stopping remain unchanged. New variants are named separately.

32K V18b S/N is1.0302 relative to default dense but .9862 on the smaller native
control subset. No measured cost decomposition yet assigns causal shares to
graph mode, host metadata synchronization, copies, observation or routing.
Use an independent diagnostic profile, not profile-overhead timings as speed evidence.

## Expanded panel preparation

Prepare all59 existing LongBench questions, all30 AIME questions and all164
HumanEval questions with eight distinct engine-seed blocks and one timed repeat.
Keep all four arms on every question: default dense, native-hook PIECEWISE,
adapter all-kept, unchanged main. Qualification and scoring must pass before launch.
Each cell runs on the same host/deployment/settings for every arm. More seeds
do not make repeated questions independent, or make RNG trajectories identical.
Report W/S/SN/N, commits, correctness, caps and by-seed step distributions;
use question-cluster CIs and separate dataset/length strata. Preserve selection
provenance for previously used versus held-out LongBench questions.
There is no user-approved accuracy noninferiority margin. Publish uncertainty;
do not infer noninferiority from nonsignificance or claim a permitted quality drop.
AIME and HumanEval have short prompts but can have long thinking/output.

## Candidate 1: fused paged KV copy (standard optimization)

The old adapter allocates gathered page tensors, transposes and copies K and V
separately. The opt-in `kv_copy_backend=triton` directly copies both to the same
preallocated head-major buffers, honoring actual interleaved source strides and
logical-to-physical pages. Prefix copied once per canvas, tail refreshed per call.
Default stays `torch`; routing, support, dtype and output math do not change.
Apply the candidate to all-kept and main equally and retain native/default dense.
Any speed gain from this copy is a standard implementation gain, not novelty.

Component qualification fixed before GPU execution:

- Synthetic BF16 native layout `[physical_page, KV_head, slot, K+V]`, Hkv2,D512,
  page64; seed2901; randomized physical page order including unused pages.
- `(prefix, canvas)` pairs `(0,1),(65,256),(32768,256),(65536,256),(98304,256)`;
  full-buffer and tail-only copy for every pair. Untouched destination prefix
  sentinel must stay bit-exact; all GPU outputs equal original Torch copy.
- Both arms preallocate destination; original intermediate allocation remains
  in the original cost. Warm8 each;100 alternating-order CUDA event samples.
  Report per-case medians/ratios, host dispatch gaps included in event span.
- CPU address/bounds/lifecycle tests, then real-model warm copy oracle, counters
  and zero timed JIT/capture qualification before any request-speed claim.
- No tuning from these timing results. Preserve failed attempts and GPU seconds.

## Candidate 2: regroup with output scatter fused into LSE merge

Cooperation candidate with Haowei/CHW; peer code stays read-only. Reuse the
existing held order; no search retuning. Keep the same Torch Q gather. Fuse O
scatter into alias2 LSE merge, and apply the same standard merge fusion to the
natural Q64 reference. Compare total gather+consumer+merge within one host;
report first-order construction and selector costs separately. Consumer savings
must exceed all remaining costs before requesting a larger quality panel.
An unchanged negative implementation is not rerun to search for a lucky result.

## Operating rules

Spec -> commit/freeze -> isolated own-directory deploy -> bind -> qualification
-> launch -> unchanged scoring -> aggregate summary. One worker/GPU, idle check,
own dyh caches/environment only. No private prompts/gold/tokens/hashes/paths in Git.
Follow repository unit tests and effective-path receipts for every new variant.
Update HANDOFF, docs and STATE.current at each result/status commit and push.

## Fused paged-copy component001 passed (2026-10-02 21:53, UTC-5)

Source3a3a3c52d;10 synthetic native-stride GPU cases bit-exact,100 alternating
samples after8 warm. Long tail copy ratios .55754/.55732/.55517 at32/64/96K;
full-copy ratios .18225/.16055/.15203. Event spans include host dispatch and
original intermediate allocation. Real-model qualification and W/accuracy pending;
standard optimization must also apply to all-kept. No request-speed claim.
Reserved3.867908GPU seconds on dlm2; run closed. Sources:
`results/v29_20261002/paged_copy001/{summary.json,receipts.json,README.md}`.


Independent32K profiler (10 CPU tests) and mapped alias2 merge prototype (11 CPU tests) ready for separate GPU qualification. Fixed component protocol: historical3 accepted states, seed2903, warm8,32 rotated samples, natural_fused matched reference, frozen order/search unchanged; GPU Torch LSE and IEEE FP32 masked oracles before timing; online construction remains unmeasured. Merge is tolerance-qualified, not bit-exact. Source docs: docs/V29_32K_COST_AUDIT_20261002.md and docs/V29_REGROUP_REVIEW_20261002.md. No GPU result for these diagnostics yet.
