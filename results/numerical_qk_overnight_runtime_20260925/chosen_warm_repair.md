# v10 CP2: the ONE warm repair — production telemetry + guard glue

Commits `947cf49` (telemetry flag) and `48ad289` (fused guard, no memset for unused counters). Flags:
`telemetry='minimal'`, `guard_mode='fused'`. Defaults stay `full`/`separate`, so every older configuration is
unchanged. Family: "make optional per-layer instrumentation truly optional and consolidate its tensor glue".
MoE, backend, CUDA graphs, T formula, thresholds and the sampler were all left untouched.

## What changed (per decoder attention call)
1. `telemetry='minimal'`: no per-call metadata dict and no two on-device tile-count reductions. These are
   audit data and never select support. Host counters, cache/phase ownership, summary accounting, memory
   budgets and reproducibility fields are kept. Per-tile skip statistics are reported as
   `N/A (use a token-matched full audit run)`, never as zeros; `router.records()` returns None.
2. `guard_mode='fused'`: one Triton pass (`_guard`, 64 programs, each writes its own ok slot, so no
   zero-init) followed by one `_assert_async`. It covers exactly the union of the previous three chains:
   non-finite attention output (torch.isfinite), invalid score rows, and malformed routed cached-score tiles
   (ordinary path). It is still asynchronous and still issued before the output is returned, as before.
3. The preqk consumer's one-element `counters` stand-in, never written when trace=False, uses `empty` instead
   of `zeros` (one memset less). trace=True still zero-initializes.

## Parity / guard evidence (`tests/test_v10_minimal_telemetry.py`, 12 passed on H100)
- Full vs minimal and separate vs fused: attention outputs are bit-identical over 10 steps (2 anchors) on 2
  layers, for both static and generic kernels. Counters are identical (calls, refreshes, summary
  builds/hits/misses, peak bytes, QK elements).
- The same number of guard calls in full vs minimal telemetry; the lean router uses 20 assert chains vs the
  reference's larger count.
- `guard_flags` equals the torch reference for clean input, NaN at the first element, +inf at the last,
  -inf mid-tensor, an invalid row at the last element, an invalid tile at the last element, and the no-tiles
  variant.

## Measured benefit (same captured state, per-rep restored replay, median of 10; O/O2 and P/P2 orders)
| ms | canvas 1 | canvas 6 |
|---|---:|---:|
| P ordinary -> O ordinary | 153.6 -> 150.6 (-3.0) | 166.9 -> 163.7 (-3.2) |
| P anchor -> O anchor | 162.4 -> 160.2 (-2.2) | 168.9 -> 166.1 (-2.8) |
Reproduced by the O2/P2 repeats within ±0.4 ms. This is about -2% per step, roughly 10% of the ~30 ms
sparse-native gap. It is below the ~5% trigger, so the 8-ID expansion was NOT run. Request-level effect:
`complete_request_results.*`.

## Candidates NOT taken this round
- The T-only fast path: T accounts for ~1.6 ms of device time per step; left for later.
- Restructuring `_route` (the largest measured item): the per-tile block log-mass/mu computation is
  independent across tiles and could run on a tile-parallel grid ahead of the sequential risk scan (the
  existing STORE/LOAD summary split already proves bit identity for that separation). It is a new kernel
  design, outside this round's permitted repair families, so it is proposed, not implemented.

---
**CORRECTION (v11, 2026-09-25), appended; the text above is preserved as recorded.** In the v10 CP3 *panel*,
`scripts/v10_request_runs.py::arm_config` did not copy the arm's `guard_mode` into the run config. So panel arm O
ran `telemetry='minimal'` with the DEFAULT `guard_mode='separate'` (verified in the private config
`cp3/private/configs/v10cp3.O.json`: no guard_mode key). The panel's O/P ratios (geometric 0.992) therefore
measure minimal telemetry alone, not telemetry plus the fused guard. The same-state step-replay numbers
(-3.0 ms/step) DID use `guard_mode='fused'` (the replay script passed it) and stand as stated. Fixed in v11:
arm_config now copies guard_mode/consumer/support_build/collect.
