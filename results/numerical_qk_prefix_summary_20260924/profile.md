# Corrected direct measurement (repaired harness)

`scripts/preqk_full_forward_profile.py` after the v8 repairs: ONE armed step
feeds both the per-layer and the full-step capture, the intended prefix is
asserted, the production causal T is installed in every router, RNG is
restored around each timed series, replay inputs are digested twice to prove
they are identical, counters are reported as per-repetition deltas, and
optional telemetry is cleared between replays.

Captured state: `aime26/14`, armed step 91, LOCAL layer 0 at a
**saturated** prefix 1023 (nk 1279), GLOBAL layer 5 at prefix
1923 (nk 2179). Sensitivity: genuinely nonuniform causal T,
min 1.0, max 3.625, spread 2.625.

## Complete `Attention.__call__` (preparation, planning, lease, guards)

| arm | LOCAL layer 0 (D=256) | GLOBAL layer 5 (D=512) |
|---|---|---|
| native_dense | dense 0.074 | dense 0.317 |
| cached_scores | anchor 1.604  ordinary 1.239 | anchor 2.330  ordinary 2.090 |
| routing_only_current_output | anchor 1.609  ordinary 1.545 | anchor 2.336  ordinary 2.266 |
| historical_route_preqk_current_output | anchor 1.622  ordinary 1.392 | anchor 2.545  ordinary 2.097 |
| preqk_summary_selector | anchor 1.663  ordinary 0.990 | anchor 2.345  ordinary 1.995 |

The summary selector is **29% faster on the local-layer ordinary call**
(1.392 -> 0.990 ms) and unchanged on the global layer, which is correct: the
default enables it on LOCAL layers only.

## Complete denoising step (30 decoder layers + sampler)

| arm | anchor (ms) | ordinary (ms) |
|---|---|---|
| native_dense | 183.24 | -- |
| cached_scores | 213.15 | 205.07 |
| historical_route_preqk_current_output | 212.04 | 206.20 |
| preqk_summary_selector | 214.46 | 206.23 |
| routing_only_current_output | 212.40 | 214.96 |

**The per-call gain does not reach the step.** 206.20 -> 206.23 ms, with
minima 204.39 and 204.36: unchanged. The summary path is definitely active
during these steps -- the per-repetition counter delta for one ordinary step
shows 25 summary hits serving 375 prefix tiles across the
30 attention calls (25 local layers of 30).

The whole numerical stack sits only ~23 ms above dense per step (206 vs 183),
so there was never enough selector time at the step level for a 29% per-call
reduction on 25 layers to show up as the ~10 ms that naive multiplication
predicts. Reporting the discrepancy rather than the multiplication: see
`decision.md`, question 5.

## Scope

One captured state, one id, two instrumented reference layers. Per-call and
per-step figures are directly measured; nothing here is a component sum or a
30x extrapolation. Request-level behaviour is a separate question that this
harness has still not reproduced (v7's ~430 ms per real forward vs ~206 ms
replayed).

---
**CORRECTED (v9, 2026-09-25) — appended; the text above is preserved as recorded.**
- The full-step row labeled `native_dense` (183.24 ms) ran `_install_dense` with
  `attention_override=None`, which the BLASST registry dispatcher sends to
  `dense_eager_attention_forward`, NOT the untouched native SDPA function. It is a
  same-legal-mask eager dense row. v9's dispatch spy proves this (30/30 calls in
  `dense_eager`) — see `results/numerical_qk_request_timing_20260924/measurement_contract_and_dispatch.md`.
- v8's step replay restored RNG once per series, shared the native sampler /
  stopping criterion / capture-time T observer across repetitions, and copied
  fixtures inside the timer. Superseded by `scripts/v9_step_replay_profile.py`.
- The inference "206 − 183 ≈ 23 ms, so there was never room for a ~10 ms gain"
  is WITHDRAWN: the 183 ms baseline is mislabeled, the two contexts need not add,
  and replay-state/launch-overlap effects were unresolved. The L2-residency
  explanation remains an untested hypothesis. The raw v8 numbers stand as measured.
- `real_state_qualify.json` is the 512-token trajectory check, not the full-budget
  smoke receipt; the latter is now `results/numerical_qk_request_timing_20260924/v8_smoke_receipt_verification.json`.
