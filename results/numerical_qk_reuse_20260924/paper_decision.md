# Integrated native M1/M3: negative four-question smoke

The numerical-cache system is connected and avoids current QK on valid reuse, but this frozen period8 configuration does not support a quality–generation-speed paper claim. Complete the bounded first delivery; do not expand this result into the 240-request matrix.

Four preselected AIME26 development questions, seed42, native adaptive stopping, thinkingON, max8192, natural EOS. All sixteen attempt0 outputs are retained. The frozen channel-aware scorer was not changed after outcomes.

| Arm | Correct | Output cap / unparsed | Mean request wall | Mean decoder calls | Mean canvases |
|---|---:|---:|---:|---:|---:|
| Native dense | 3/4 | 1/4 / 1/4 | 41.3s | 271.5 | 25.0 |
| Fresh Junyu T | 3/4 | 0/4 / 0/4 | 39.9s | 234.8 | 21.0 |
| M1 | 0/4 | 3/4 / 4/4 | 254.2s | 1247.0 | 30.8 |
| M3 R2 | 0/4 | 3/4 / 4/4 | 166.0s | 1135.5 | 25.8 |

These are complete-budget requests, not necessarily completed answers. Smaller M3 wall than M1 is an observed system result, not an isolated selector-cost estimate: their trajectories, outputs and incurred compilation differ. Cold/JIT/setup is included. No clean warm PROFILE was run. Dense timing-only repeats have mean wall40.336s and initial-prefill-excluded device timeline40.188s; their tokens/calls match original quality receipts. Paired exploratory ratios and question-bootstrap intervals are in `smoke_report.md`. They do not establish speed or quality equivalence on four questions.

## What is implemented and verified

- Own branch from Junyu `053441c6...`; newest fetched peer refs matched the pinned identities. Private pinned native kernel/ATen rebuild; no peer/shared environment modifications.
- Per-query-head Q128 × KV64 geometry, Gaussian32 current-V selection, Junyu sequential retained-state/T rule. All30 native decoder attention layers are hooked. This is not the historical five-GLOBAL-layer F consumer.
- M1 uses genuinely observed transformed scores at calls0,8,16,... with current projected V for selection, and the same historical scores with current original V for output. M3 holds the M1 bitmap for2 actual calls and uses the same output executor/precision. No sampler/SC/EOS changes.
- Six GPU mathematical tests, two actual dispatch/lifecycle tests, and twelve fresh-history Junyu comparisons passed frozen numerical gates. The current-QK producer is replaced by a throwing spy during GPU reuse qualification; the cached consumer has no Q/K input. All eight natural M1/M3 traces obey the two clocks, with zero current-QK elements on ordinary reuse and no unsupported-mask refreshes.
- M1/M3 reused86.9% of target score elements. Actual physical full-PV skip: T30.9%, M159.5%, M357.0%. M3 held67,680 of136,260 attention-call decisions, avoiding projected-V/router work on those calls. M1 still reads/projects current V every call. These quantities are distinct and are not measured DRAM-byte savings.
- Same-stream DynamicCache lifecycle is qualified within this scope. CUDA Graph replay, asynchronous overlap and Haowei's alternative executor are NOT qualified/integrated. Peak score cache stays below the2GiB guard.

Pinned T_s50 thresholds were unchanged. T and numerical-reuse arms nevertheless achieve different physical sparsities; this is not a matched-work isolation of stale-output error. Installed native dense and Junyu also differ in LOCAL support: native SDPA attends its supplied truncated prefix+canvas, whereas Junyu applies a query-relative lower-window bound. T/M1/M3 preserve Junyu's rule. Do not attribute every dense/T difference to this new cache.

## Actual stopping diagnostic

One frozen question, aime26/2, was run separately. The exact-stop observer passed same-load dense token/call parity. Diagnostic tokens and calls match each arm's original primary request. No diagnostic timing or quality replaces the primary table.

| Arm | Calls | Stable AND not confident | Mean accepted /256 | Mean renoised /256 |
|---|---:|---:|---:|---:|
| Dense |146|1|129.71|126.29|
| T |208|6|122.84|133.16|
| M1 |908|670|13.93|242.07|
| M3 |1289|855|22.33|233.67|

Native stop is the conjunction of stable predictions and mean-entropy confidence; native ST is1 in this installed stack. M1/M3 frequently satisfy stability while failing confidence on this question, and accept far fewer positions. This describes their actual paths; it does not prove that confidence distortion alone caused the quality failure, nor justify threshold relaxation. Full joint flags, denominators and per-canvas counts are in `stop_summary.md/json`.

## Scope, accounting and next decision

True output-ready burst events were not available: TBT=N/A. CUDA-event generation spans exclude only initial prefill and include subsequent commit/encoder work and host gaps. CPU generation boundary, warm full-forward phase PROFILE, isolated producer critical cost and same-support current-output comparison are NOT_RUN; no component sum is presented as measured E2E.

Total charged GPU-process wall3056.85s including loading/JIT/tests/failures/diagnostics. Measured process CPU2881.48s excludes explicitly noted partially metered build/analysis work. Own remote directory662MiB; roughly60GiB free; all own model processes exited0 and GPU is released. Failures and raw outputs remain under the private remote run paths with hashes. Only redacted records and source/configs are committed.

R3, full30×2seed expansion, sparsity curve, clean warm optimization and second-model generation are NOT_RUN. I-DLM preflight found no supported checkpoint/adapter in the inspected pinned environment; no stride or verification semantics were invented. Four questions/one seed cannot establish noninferiority. No new scientific contribution is claimed from integration alone.

The next unique diagnostic, before more natural generation, is the v5 same-state/same-cache/same-support comparison of cached-score output versus `routing_only_current_output`. It can distinguish stale final-output error from changed routing support, with current QK cost charged. That experiment is not yet executed and is not an online method or oracle upper bound. Preserve this round's failures; do not silently retune the score period, thresholds, masks or scorer to erase them.
