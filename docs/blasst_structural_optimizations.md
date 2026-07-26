# Structural BLASST optimizations

The calibrated baseline remains a 128-by-64 parent tile with reverse KV
traversal and a strict `score < lambda` decision. Its default high/mid/low
thresholds are `0.04858582466840744`, `0.4334796965122223`, and `1.0` at the
unchanged 0.75 and 0.25 remaining-mask boundaries.

`blasst.runtime` provides the predictor and lifecycle layer. Metadata is keyed
by request and transformer layer and double-buffered across denoising steps.
Each row stores `ceil(num_kv_tiles / 64)` signed `int64` signature words, FP32
row sparsity, and int64 top-1/top-2 tile IDs. Each 128-row supertile stores FP32
keep fraction and top-score coverage vectors. Decisions are aggregated across
heads by majority by default; `all` and `any` are supported.

The rejected local-grouping experiment used a stable key ordered by current mask state,
previous top-1 tile, previous top-2 tile bucket (width 4), previous row-sparsity
octile, a 16-bit compressed signature bucket, 32-row position bucket, and
original row as the final tie breaker. Local mode never crosses the original
128-row supertile. With no previous metadata it deterministically falls back to
mask/position grouping. The retained experimental kernel code can consume the resulting
query permutation, loads Q using original positions (therefore preserving
RoPE), and scatters O to original positions before the output projection. K and
V are never permuted.

KV order is a top-K prefix followed by every unselected tile in the original
reverse order. With previous metadata, importance is

`1.0 * keep_fraction + 0.5 * top_score_coverage + 0.05 * locality + 0.05 * visible_fraction`.

Locality uses original supertile/tile coordinates. Ties use ascending original
KV tile ID. Prefix K=0 is exactly reverse traversal. The fused kernel accepts
one order per request, layer call, and query supertile, shared across heads.

The unfused correctness path implements 16/32/64/128-row groups. A group skips
only when every valid row votes to skip. Partial tiles update online-softmax and
PV only for active groups; full skips leave all row state unchanged. Dense
fallback is explicit and counted. It reports row, microgroup, parent-tile,
BMM2-group, V-load, and fallback rates separately.

## Hardware limitation

The repository's accepted path is one Triton program with four warps
per 128-row tile, not the included TensorRT-LLM SM90 CuTe/TMA reference tree.
The disabled experimental specializations retain one 128-by-64 QK dot while splitting probabilities into
compile-time 64/32/16-row fragments. Each fragment has its own uniform branch
around `tl.dot(P,V)`; the dense-fallback branch uses the original 128-row dot.
V loading is deferred until at least one valid group is active. Fully padded
groups are excluded from decisions and counters, including at lambda zero.

All four variants offline-compile to SM90 PTX. The 32-row PTX, for example,
contains a predicate branch around its `mma.sync` region, demonstrating that
the fragment MMA instructions are control-flow-skippable. The installed Nsight
Compute 2021.3.1 predates Hopper and cannot provide trustworthy H100 retired
instruction counters, so the runtime comparison below uses CUDA Events.

## H100 acceptance result

The model benchmark used LLaDA at sequence length 4096, batch size 3 (one
context at each 15/50/90% mask ratio), two warmups, and five timed forwards on
an H100 80 GB. Times are end-to-end attention-kernel totals over the model.

| Path | Median (ms) | vs. dense | vs. sparse baseline |
| --- | ---: | ---: | ---: |
| compiled dense FlashAttention | 496.071 | 1.000x | 0.954x |
| sparse baseline, new features off | 473.333 | 1.048x | 1.000x |
| query grouping only | 476.553 | 1.041x | 0.993x |
| KV priority only | 476.471 | 1.041x | 0.993x |
| query grouping + KV priority | 479.237 | 1.035x | 0.988x |
| 64-row microgroups | 540.139 | 0.918x | 0.876x |
| 32-row microgroups, fallback 3 | 527.664 | 0.940x | 0.897x |
| all features, 32-row/fallback 3 | 600.176 | 0.827x | 0.789x |

The sparse baseline physically skips 29.06% of parent BMM2 work. KV priority
increases parent skips to 31.60%, but is still 0.66% slower than the baseline.
Microgroups increase logical skip opportunities (up to 47.45% at 16 rows), but
fragment MMAs, extra branches, and selection overhead dominate the saved work.
The grouping/order plan itself costs 86.67 ms for this shape.

Consequently, `skip_group_rows=128`, reverse KV traversal, and no query
permutation are the only accepted production configuration. The public kernel
entry point rejects query grouping, KV reordering, alternate query-tile sizes,
and microgroups so a slower path cannot be enabled accidentally. This is an
acceptance decision based on measured latency, not merely static PTX. The
implementation and historical artifacts remain for auditability.

### Why 32-row microgroups lose

On SM90 the 128-row PV path maps to Hopper WGMMA, while each 32-row fragment
maps to legacy `mma.sync` instructions (128 such instructions are present
statically in the generated PTX). The hybrid kernel must also retain both the
dense and fragmented PV paths, four uniform group branches, split/join
transposes, and the full 128-row FP32 accumulator. A fallback-threshold sweep
isolates this overhead: threshold 1 performs dense PV for every non-skipped
parent and produces baseline-equivalent output, yet takes 3.265 ms versus
1.881 ms for the sparse baseline on synthetic 4096-token Q/K/V. Thus the loss
is not primarily a poor fallback threshold.

Prefetching V for the fallback variant was also rejected: it regressed the
32-row/fallback-3 synthetic result from 3.788 ms to 4.771 ms. Avoiding V loads
for fully skipped parents is more valuable than restoring that overlap.

The tested `query_tile_rows=64, skip_group_rows=32` specialization cuts
the group fan-out from four to two and reduces the model result from 527.664
ms to 503.113 ms. It is still slower than dense FlashAttention (496.823 ms)
and 6.13% slower than the calibrated sparse baseline (474.055 ms). Its high
noise agreement is only 93.67%, so recalibration to the 95% quality constraint
would necessarily remove sparsity and cannot rescue its latency. Four warps
and two pipeline stages remain optimal; tested 2/8-warp and three-stage
launches regress. The specialization is therefore disabled together with the
other rejected structural paths.

Use `scripts/blasst_calibration_regression.py` to validate the checked-in
artifact. Its 38.6-second elapsed field is intentionally reported only as
unfused calibration/reference-forward time.

Reproduce the reported run with `scripts/llada_blasst_kernel_benchmark.py`.
The complete measurements and counters are stored in
`outputs/blasst_structural_benchmark_4096_final.json`; the synthetic Q/K/V
cross-check is in `outputs/blasst_structural_kernel_h100_4096_segmented.json`.
