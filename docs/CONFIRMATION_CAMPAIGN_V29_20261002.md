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

## Regroup fused writeback component001 closed (2026-10-02 22:05 (UTC-5))

Sourcee07aec657; three fixed accepted states passed GPU numeric oracles.
Held_fused/natural_fused1.00566/.97856/1.01061, geometric.998177: approximate
tie and no general regroup speed evidence. No request expansion for this held
search. Standard natural merge fusion alone gives.90340/.92598/.90358 ratios;
keep as shared implementation candidate, not regroup novelty. Not bit-exact.
Reserved51.200860GPU seconds, including36.475206CPU selection; source
`results/v29_20261002/regroup_merge001/`. No W/quality evidence.

Adapter `merge_backend=triton` opt-in now CPU-qualified alongside copy backend;
default remains torch. Identity constructed from known CPU indices, no D2H
validation; count builds/calls, reuse only immutable index geometry, clear at
request boundaries. Apply to main/all-kept equally; all request initialization
cost stays in W. Real-model/GPU adapter qualification remains pending.
Independent32K profiler updated with denoise/encoder scope where exact existing
CPU phase is available; default FULL missing scopes stay unknown.11 CPU tests.
Combined copy/merge/profile/adapter suites67 tests pass.


## 32K diagnostic launched and third vLLM host ready (2026-10-02 22:14 (UTC-5))

Four-arm independent32K profile running on dllm, frozen39e08c521, same one
question/engine28001, source/input/CPU-test guards passed. First preparation
attempt used wrong system Python lacking Torch; failed before GPU work (0s),
preserved. Attempt002 uses qualified own vLLM interpreter. Each arm new engine
and run/cache, GPU-idle gate,11 profiler CPU tests pass. No speed inference from
profiled records. Fifth PIECEWISE-dense-without-hooks diagnostic is specified in
`docs/V29_DENSE_CONTROLS_20261002.md`, not yet launched and not part of formal
panel. Main/Q64 already run in vLLM; component candidates have narrower scope.

dlm2 own vLLM0.30/Torch2.13cu130/Triton3.7.1/FlashInfer0.6.18.post1 installed,
202 runtime distributions match mpk, pip check/CPU imports pass, FA4 paged-patch
bytes match qualified mpk. Model606shards validated read-only. Setup GPU work0;
GPU numerical qualification remains required. No shared environment modified.

## Latest audit and diagnostic status (2026-10-02 22:35, UTC-5)

First independent32K diagnostic closed four arms successfully, reserved1081.689752
GPU seconds. Source39e08c521; result `results/v29_20261002/cost32k_002/`.
GLOBAL/LOCAL name classification and partial GPU associations prevent a valid
GPU cost breakdown. Preserve the original diagnostic; fix/test the profiler and
add separately qualified event scopes before repeating. No profile speed claim.

Source audit confirms main fused observation shares one QK pass between complete
V output and rank32 summaries; observation itself is custom Triton, while the
sparse output consumer and native dense use FA4 through different dispatches.
See `docs/V29_BASELINE_CODE_AUDIT_20261002.md`. Main has no2K length gate; short
prompt panels cannot inherit the HF AIME gated-method conclusion. New BF16-mode
oracle protocol adds20cases to4existing CUDA tests, source core unchanged;
GPU correctness execution pending, see `docs/V29_FUSED_OBSERVATION_AUDIT_20261002.md`.

Expanded panel infrastructure remains under CPU review and is not launched.
Separately, V28 generation closed24workers; its CPU scoring stopped on a
snapshot-symlink comparison bug, fixed on that branch7c8c608ed with18CPU tests.
Original generation/bindings are unchanged; repaired CPU rescoring is pending.

## New diagnostic and expanded-panel freeze (2026-10-02 22:45, UTC-5)

Expanded runner/scorer CPU validation passes33tests, including canonical model
snapshot paths, complete scorer-only artifact mirrors, original byte pins,
implicit Q128 defaults, native task contracts and same-host qualification proof.
Committed specs select engine seeds29001-29008, all59LongBench/all30AIME/all164
HumanEval questions and all4arms (8096timed plus8096warm total); suites assigned
dllm/dlm2/mpk respectively. No formal or qualification GPU worker is launched
yet; full qualification and real scorer toys remain required. Main stays Q128,
legacy canvas, torch copy/merge, no2K gate. New code never silently changes it.

Corrected independent profiler has16CPU checks: actual model.layers naming,
copy API wait caveat, optional same-stream event spans and fused-observe/DP-build/
DP-route leaf scopes. New four-engine PIECEWISE diagnostic (dense-nohook/native/
allkept/main) is prepared, not yet launched. Default FULL result remains unknown
for layer breakdown. Timings are diagnostic only, never formal panel evidence.

BF16 observation correctness protocol now covers actual scale1.0 and the prior
scale512**-.5:40BF16 cases plus4original=44. First CPU preflight found no pytest
and stopped before GPU work (0s). A separate own runner dependency directory
now supplies pytest; qualified vLLM environment unchanged. New attempt002 GPU
correctness awaits this source freeze. No tolerance was changed.

V28 complete preview was repaired/scored/pushed on its own branch91db65396.
See `docs/V29_V28_PREVIEW_INTERPRETATION_20261002.md`: Q64 request ratio.94245
against matchedQ128, but S/N1.00516 and seed/length signs vary; six questions
only. Canvas release W1.01228[.99035,1.03470], no supported gain. Main/allkept
W.99576[.91015,1.08942], no demonstrated increment. Keep matched native whose
preview accuracy is43/48 versus main40/48 and Q6441/48. No noninferiority claim.

## GPU correctness passed; event diagnostic active (2026-10-02 22:51, UTC-5)

Fused observation44GPU tests pass on H100, source54163f4e7:40BF16 FP32-oracle
cases plus4original STORE/LOAD tests. Includes actual scale1.0, full V output,
GQA, splits1/2 and boundary tails. Fixed tolerances unchanged; synthetic numeric
qualification, not task accuracy, bit-exactness or speed. Reserved32.737210GPU
seconds. First missing-pytest attempt stopped on CPU (0GPU); second used isolated
runner pytest directory, leaving the qualified vLLM environment unchanged.
Source `results/v29_20261002/fused_observe_bf16_001/`.

Corrected four-PIECEWISE-arm event diagnostic003 is active on dllm, frozen
54163f4e7,16CPU checks passed, idle gate passed. It includes native dense with
no hooks and GLOBAL/LOCAL plus fused observation/DP leaf event spans. Profiler
overhead remains excluded from formal evidence; no breakdown result yet.

LongBench59x8x4 source/binding frozen on dllm, no GPU qualification launched.
First CPU preparation missed an archived provenance spec and stopped (0GPU);
new002directory includes the committed E14 source, all parent input/source
hashes pass, strict rebind passes and full1888timed/1888warm inventory verifies.
Generation-host CPU checks and scorer qualification remain pending. AIME and
HumanEval CPU deployment preparation proceeds on their assigned hosts; no formal
panel launched. No existing run or frozen binding was edited.
