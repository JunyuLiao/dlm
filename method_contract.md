# M1 numerical score reuse with current V — frozen initial contract

This is our operational interpretation of the v5 specification, **not a verbatim
meeting equation**. Sole execution specification: user-supplied v5 on2026-09-24.
Private transcript and superseded prompts are not inputs to the repository.

Base Junyu `053441c6c6741ada6728dfc97d8faa6ea2be72aa`; fetched latest equals pin.
Read-only Haowei reference `b23f969a3c52417ba84a999ff7a31e3dd00bb697`, also unchanged.
Our donor `3df48e0c307f03e4743ad1ffda4a6fdc493d08e0` is historical provenance,
not a numerical qualification of this native Torch path.

## Meaning of one cached score

One entry contains genuinely observed **transformed per-key scores**, FP32 storage
of Junyu's BF16 QK / BF16 scale / BF16 additive-mask result. Structural illegality
is -inf. It is not a block mass, Boolean bitmap, raw unscaled QK, or old PV.
The production output is BF16; routing and online normalizers are FP32.
No lower-precision cache is substituted for timing.

For each row and increasing KV64 block, compute block logsumexp z, within-block
weights w, and mu = w @ (current V @ R). R is Junyu's unchanged Gaussian32 bank,
seed1729, independently keyed by layer and native KV head. The reference divisor
is the RMS full-dimensional current valid-V norm used by Junyu (not per-element
RMS). With retained state (l,u), alpha=exp(z-logaddexp(l,z)); risk is
alpha*norm(mu-u)/ref. Apply causal completed-step T sensitivity to each row
BEFORE taking the maximum over that head's Q128 tile. Strict '< threshold'
drops a tile; ties and first legal support are retained. Only retained blocks
update state. Empty rows output zero. NaN/+inf is an explicit invalid-input
qualification/fallback event, not an excuse to delete support.

M1 output is `softmax(cached_scores restricted to retained legal keys) @ current_V`.
The retained denominator is recomputed. Both current projected-V routing and
current original-V output remain; Q/K/V model projections are unchanged.
Ordinary valid score reuse has **no current QK** even for retained output.
Fresh scores are not silently reconstructed to make the reference look better.

## Geometry, lifetime and storage

Semantic/physical selection uses Junyu **per query head × Q128 × KV64**, with GQA
sharing native K/V. It is not the old vLLM GQA8×Q2×KV32 method. Output execution
may use smaller query tiles while retaining each Q128 bitmap without regrouping.

Mask boundary identified before candidate generation: when there is no explicit
mask, pinned Junyu applies a query-relative LOCAL lower window bound. Installed
Transformers 5.11 native SDPA instead attends the supplied already-truncated
LOCAL prefix plus canvas (its SDPA interface ignores the extra window keyword).
This first reproduction preserves Junyu's convention identically for fresh T,
M1 and M3. Native dense remains untouched. Thus T-versus-dense includes that
inherited support difference; it is NOT an all-kept native-mask equivalence claim.
No historical result is rewritten. A later native-mask correction must be named
and qualified separately, not silently mixed into these receipts.

Identity includes request, canvas, encoder-write epoch, actual native prefix
ownership/length, absolute position range, layer/heads/shape, dtype/device,
scale and structural mask signature. New/changed legality forces observation.
Encoder/prefill/commit and new requests clear the cache. Unsupported dynamic
mask contents force fresh scores rather than guessing equality from a pointer.
No tensor-content CPU comparison or `.item()` occurs in steady score reuse.
LOCAL storage crops only a source-qualified legal range, aligned to KV64 so scan
order and block boundaries remain unchanged. GLOBAL retains its legal range.

FP32 score storage costs B*H*Q*Kcompact*4 bytes per layer. Cache controller checks
the aggregate 2GiB bound before allocation; exceeding it is an explicit failure
or separately reported native fallback, never stale reuse. Scores, bitmaps and
metadata are request-local. Same-stream enqueue order is the initial publication
dependency; no side-stream overlap or CUDA-graph claim is made in checkpoint1.

## Two independent clocks

Canvas-local decoder calls start at0. Genuine score observations occur at0,8,16,…,
and on invalidation. Each observation also rebuilds the value/T decision. This
period8 is an engineering starting point, not a meeting requirement or optimum.
M1 rebuilds its decision every call; M3 R2 rebuilds after2 actual calls from the
last decision. Thus0,2,4,6,8,… with fixed geometry. R3 gives0,3,6,8,11,… because
score observation at8 also rebuilds the decision. R1 equals M1. Holding a bitmap
does not project V or run the value router, and never advances score timestamps.

## Controls and limits

Native dense uses the unbound original SDPA adapter. Fresh Junyu T uses the
unchanged current-QK kernel and separately frozen local/global T50 thresholds:
local -1.0099318265914916; global -3.1366905212402343. beta3, EMA.5, rank32.
They came from peer work calibration, not new AIME answers. Actual work must be
reported; do not inherit50% achieved sparsity or peer accuracy.

All-kept cached-score output is compared to the same cached-score reference,
not asserted equal to current-score dense. A same-state retained-support
current-output comparator isolates stale-score error, clearly named
`routing_only_current_output`. Its current QK cost is charged.

Native adaptive sampling, full8192 answer budget, thinkingON and EOS remain.
No forced FIXED16, confidence threshold change, residual or self-conditioning
modification. First outputs remain immutable; quality is offline. Timing is
whole request plus true initial-prefill-excluded generation only where observed;
per-canvas counts are actual calls. No inferred token-stream TBT.

Haowei's fetched executor is a separate integration interface: its cited JAX E2E
driver documents mask/window limitations. We do not migrate to it or inherit
its performance. This is a partial joint prototype until a qualified compatible
executor is connected by its owner.
