# v14 method contract: CVM-T (cached value-risk margin + live temporal protection)

A new, explicitly named simplification motivated by the cost failure of the frozen M1/M3 path (v13). It is NOT
the unchanged M1, and it is NOT specified by the meeting. The original M1/M3 results keep their names.

## Source facts (reused, unchanged)
- Junyu's value-direction risk: centered change of the running retained projected output (Gaussian32 sketch),
  weighted by the row sensitivity s_i. Production path: FIXED tf32x3_register + TMA router (squared-risk
  domain: drop iff max_rows rho^2 s^2 < tau^2), Q128 x KV64 decision shared by two Q64 CTAs (cluster decide()).
- T: u <- 0.5u + 0.5*flip(argmax), s = clamp(1 + 3u, 1, 4); completed iterations only; reset each canvas;
  the first two iterations have s = 1.
- Native adaptive sampler and stopping are unchanged. LOCAL layers are native in every arm; GLOBAL layers
  [5, 11, 17, 23, 29] only.

## New approximations (the candidate)
1. **Private v5 kernel** (`csrc/value_direction_v5.{cu,h}`, `torch_bridge_v5.cpp`, ATen `value_direction_hopper_v5`,
   ABI 5; Junyu's v4 files and binary untouched):
   - `protect_tile`: tiles j >= (prefix // 64) are never skipped. That covers the current canvas and the partial
     prefix/canvas edge tile. It is applied inside `decide()`, so retained state, P publication and PV all agree.
   - optional export of the UNWEIGHTED `log_rho[i,J]` (FIXED: 0.5*log rho^2) per owned row, before log s_i and
     before the tile max, with the scan's first-support (+inf) and inactive (-inf) sentinels. Layout [B,H,KT,Q],
     FP32. It piggybacks on the real anchor; there is no extra QK producer.
2. **True observation (anchor):** A8 clock (canvas step % 8 == 0), plus every new canvas or cache epoch/identity
   change. It runs Junyu's fresh value-aware attention on CURRENT Q/K/V (v5, protect on, export on). Its output is
   the anchor step's attention output.
3. **Ordinary step (CVM_T):** one planner launch, then the v11 preselected-support consumer:
   `restore[I,J] = any_{valid i in I} log s_i(t) + log_rho_a[i,J] >= log tau` (equality retained), then
   `skipped &= ~restore` on prunable prefix tiles only. Keep is add-only within an anchor epoch and reset at the
   next true anchor. Row-specific s is never averaged. The output is CURRENT QK/softmax/V over the retained keys.
   Ordinary steps do not read scores, project V, compute a selector dot, or write a discarded PV.
4. **Matched controls:** T_P (fresh v5 kernel every step, same mandatory set); B8_P (same anchors, same export,
   same consumer, bitmap held with no restore). T_G_original = the v13 global_T reference. All T-using arms use
   the exact T fast path.
5. **Structural fallback:** if prefix // 64 == 0 (no complete immutable-prefix tile), the call uses the original
   native SDPA. This is not a tuned threshold.

## Explicit limits
- Cached risk depends on the anchor's QK, retained state, reference and value geometry: it is stale between
  anchors even though prefix V is immutable.
- Live T does not detect every failure (stable-but-unconfident states).
- Add-only support is not guaranteed to reduce output error.
- s in [1,4] bounds the controller, not the attention error. A margin > log 4 is only ineligible for restoration,
  not certified safe.
- The consumer uses fresh-T retained-path arithmetic, which is a numerically distinct variant from the old Triton
  consumer (v11: 60/660 max-abs envelope failures; not erased).
- No novelty from caching, T, or pre-QK alone.

## T fast path (exact)
`State(fast_t=True)` (method T, no diagnostics): argmax on the same processed logits, the same flip EMA, the same
clamp; `t_history` replaces the `margin is not None` sentinel. The weight sequence is bit-identical to the frozen
observer across resets, partial canvases and ties (tests).
