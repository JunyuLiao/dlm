# Directly measured full-call and full-forward profile

`scripts/preqk_full_forward_profile.py`, one captured real state from
`aime26/2` (**prefix=109, nk=365, nq=256**; local layer 0 D=256 GQA 16/8,
global layer 5 D=512 GQA 16/2), 5 warmup + 20 timed repetitions, router
state reset outside every timed region so repeats cannot drift between
refresh and reuse. Raw: `full_forward_profile.json`.

## 1. Complete native denoising step (all 30 decoder layers + sampler)

This is a directly measured step, **not** a sum of components and not a
two-layer figure scaled by 30. It is the unit that per-canvas call counts
multiply into request wall time.

| arm | anchor (ms) | ordinary (ms) | first observation (ms) |
|---|---|---|---|
| native_dense | 164.96 | -- | 205.3 |
| fresh_junyu_T | 178.54 | -- | 2473.0 |
| cached_scores | 212.05 | 200.43 | 235.0 |
| historical_route_preqk_current_output | 209.90 | 203.09 | 236.2 |
| routing_only_current_output | 210.50 | 210.17 | 5479.7 |
| M3_held | 212.66 | 203.00 | 239.0 |

**The measured result.** On an ordinary (non-anchor) step the new pre-QK
consumer takes **203.09 ms** against `routing_only_current_output`'s
**210.17 ms** -- a directly measured saving of **7.07 ms
(3.4%)** per full forward. The cheapest numerical arm is
`cached_scores` at **200.43 ms**, but that is the configuration that
scores 0/4: it reuses stale scores and never computes current QK at all. The
pre-QK consumer therefore recovers most of the distance to that floor
(73% of the 9.74 ms gap) while keeping
current-score output.

**It is still slower than dense.** `native_dense` runs the same step in
**164.96 ms**, so the optimized arm is **1.23x** a dense forward, and
fresh Junyu T is 1.08x. Removing the discarded PV and the
dropped-tile QK did not close that gap, and this profile alone does not
decide end-to-end: that depends on how many forwards each arm needs.

## 2. Complete `Attention.__call__` (preparation, planning, lease, guards)

| arm | layer 0 (local, D=256) | layer 5 (global, D=512) |
|---|---|---|
| native_dense | dense 0.059 | dense 0.072 |
| cached_scores | anchor 1.104  ordinary 0.762 | anchor 0.999  ordinary 0.774 |
| historical_route_preqk_current_output | anchor 1.110  ordinary 1.240 | anchor 1.001  ordinary 0.844 |
| routing_only_current_output | anchor 1.468  ordinary 1.094 | anchor 1.511  ordinary 1.003 |
| M3_held | anchor 1.105  held 0.821 | anchor 0.995  held 0.837 |

**This is where the remaining cost is, and it is not uniform.** At the
global layer (D=512, GQA 16/2) the pre-QK consumer's ordinary call is
0.844 ms against
routing_only's 1.003 ms -- a clear win. At the local layer
(D=256, GQA 16/8) it is 1.240 ms against
1.094 ms -- a clear **loss**.

The reason is structural and worth stating plainly: `_preqk_pv` inherits
`_pv`'s 16-row query program, so the eight programs covering one Q128 tile
each load that tile's K independently -- eight redundant K loads per retained
tile, against one cuBLAS matmul in the materialized path. That redundancy is
paid on every *retained* tile, while the saving only accrues on *dropped*
tiles. At this captured state (prefix=109, only 6 KV tiles, a 1024 window
that never binds) few tiles are dropped, so at D=256 the redundancy wins and
at D=512 -- where the materialized path is relatively more expensive -- the
skipping wins. The 30-layer step is net faster because the global layers and
the removed discarded PV more than pay for the local-layer regression.

## 3. Materialized current-QK elements (from the run counters)

- `routing_only_current_output`: 5.723e+09
- `historical_route_preqk_current_output`: 3.815e+09
- `cached_scores`: 3.815e+09

The pre-QK arm now materializes exactly as much current QK as `cached_scores`
-- i.e. **only at the period-8 anchors** -- a 33% reduction against
`routing_only_current_output`, which re-observed the whole compact domain on
every step. Note precisely what this counter is: *materialized* score
elements. The QK the pre-QK kernel computes inside retained tiles is
additional and is bounded by the retained bitmap; it is reported through the
kernel's own tile counters, not here.

## Limits

- One captured state, one id, short context (prefix=109). A late-answer state
  with a saturated 1024-token local window and many more KV tiles would shift
  both the drop fraction and the K-reload penalty; this does not predict it.
- `fresh_junyu_T` was measured only at the anchor phase (it has no score-cache
  phases); its first observation includes the compiled-library load.
- Per-forward timings do not establish end-to-end anything. The bounded
  generation panel measures that separately.
