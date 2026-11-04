# LLaDA2.2-mini / TraDo LongBench v2, seed 42

Authorized 2026-11-04: five arms per model, native dense, all-kept FA4,
V31 K512, V3a K512, and current-tile-initialized Gaussian32 V2. This is a new
campaign; historical cancelled LongBench/Gemma jobs remain stopped.

Use the pinned official 503-item LongBench v2 dataset, revision
`2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`, and THUDM/LongBench checkout
`2e00731f8d0bff23dc4325161044d0ed8af94c1e`. Official length labels select
395 short+medium items for LLaDA2.2 and 180 short items for TraDo. Length labels
are words, not model tokens. Use official zero-shot prompt and answer extraction;
unparsed outputs count wrong, with no chance compensation. No extra CoT pass.

Use each model's existing chat template and native sampler, BF16, batch 1,
one H100. Output cap 128 for both. Model limits are 131072 and 32768 tokens,
including the output reserve. Follow official head/tail middle truncation when
necessary; rebuild the chat wrapper, verify the final token limit, and record
original/retained token counts and truncation. Keep every selected item. Report
length groups, difficulty and truncated/untruncated slices separately.

Retain source-pinned adapters from the completed multi-model campaign:
LLaDA numerical snapshot `224f8f11173bd16e5cf10054a963c74233758cf2`, model
`e4f77d25ab5ca86f2223c6cdc6a72b887e20c8a5`; TraDo numerical snapshot
`8c0d14b031636ddcc708df75fa1d57aefa8b0ba4`, model
`2d37bd3ebb743a62a3bfecc4cd27b92cf396c5a7`. No EOS masking or sampler change.
LLaDA: 32-token blocks, steps 32, threshold .5, editing threshold 0,
max post steps 16, temperature 0, native early EOS. TraDo: 4-token blocks,
4 steps, temperature 1, top-p 1, top-k 0, confidence .9, native EOS.

All denoising layers are eligible; prefill and commit forwards stay dense.
Native dense uses SDPA; all-kept FA4 is the matched consumer control.
The full-D128 FA4 sparse consumer uses Q128/KV64 tiles, per query head/block,
four-call map reuse, no cross-canvas carry. Credit Yuhan's inherited map reuse
separately from Junyu's selectors and the current-tile initialization variant.

K512 means 512 KV **tokens**, i.e. eight KV64 tiles, including current mandatory
tiles. Each selection keeps min(J, max(8, mandatory_count)) eligible tiles.
V31 ranks attention mass; V3a ranks Gaussian32 full-information singleton risk,
projection seed 1729+layer. Audit exact budgets, mandatory support, eligibility,
and actual physical tile counts on selection and held calls.

Finish all dense and fixed-budget panels before calibrating V2. Compute each
model's reference sparsity by pooling V31/V3a actual kept/eligible tile counts.
V2 uses the existing current-tile initialization and preserve-mass recurrence,
same map lifecycle and consumer. Calibrate on six prompt-length strata per
selected length group, count-only, with gold/quality withheld. These prompts are
transductive and disclosed. At most eight candidate thresholds, starting .01
(LLaDA) or .03 (TraDo), logarithmic bracket [1e-8, 4]. Seek the measured
reference; accept calibration sparsity at least 95%, within three percentage
points of reference. Do not run full benchmarks for calibration. Freeze the
selected threshold before each full V2 panel. Report full-panel drift; only if
below 95%, permit one new count-only calibration/full-panel attempt, preserving
the first attempt. No accuracy-based selection or repeated target chasing.

Before launch: committed spec/source, frozen identities, focused CPU tests,
independent CPU/CUDA masks and FP32 masked-attention consumer qualification at
short/32K/128K geometry, and exact historical all-kept generation/count replay.
V2 recurrence receives its existing independent CPU/CUDA gate. All arms use
identical per-model prompts, seed 42, model/runtime, limits and scoring.

Run one GPU worker, resumable immutable shards, a fresh idle-device check before
each launch, and nvidia-smi health checks every 900 seconds. Retain failed
attempts. No GPU-hour ceiling. Freeze source/input hashes and source bindings;
no raw prompts, gold, token IDs or generations in Git. Store private evidence
under the campaign's private directories, sanitized records/receipts separately.

Report accuracy, parsing/EOS/cap rates, N, C, T, N/request, C/request, pooled
N/C, actual kept/eligible/skipped tile counts, sparsity, discovery work and GPU
seconds. Report paired item-bootstrap intervals; this single-seed campaign does
not estimate seed variability. Timing is instrumented accounting, with no clean
latency or speedup claim. Score only complete source-bound panels and close with
cross-arm fairness, coverage, private-token-hash and immutable-source audits.
