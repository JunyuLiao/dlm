# Directly measured full-call, full-forward and scaling profile

`scripts/preqk_full_forward_profile.py` (two captured states) and
`scripts/preqk_scaling_sweep.py` (context sweep). 5 warmup + 20 timed
repetitions (12 for the second capture), router state reset outside every
timed region so a repeat cannot drift between refresh and reuse. Raw:
`full_forward_profile.json`, `full_forward_profile_long.json`,
`scaling_sweep.json`.

> **Capture limitation, stated up front.** Both captures recorded the FIRST
> attention call of the run, so both are short-context states (prefix 109 and
> 131, nk 365 and 387) even though the second was launched at a later step.
> Full answers reach far longer global contexts, so the denoising-step table
> below describes an early state only. The scaling sweep in section 3 is what
> covers real answer lengths.

## 1. Complete native denoising step (30 decoder layers + sampler)

Directly measured, not a component sum and not a two-layer figure scaled by 30.

| arm | capture A anchor | A ordinary | capture B anchor | B ordinary |
|---|---|---|---|---|
| native_dense | 164.96 | -- | 174.40 | -- |
| fresh_junyu_T | 178.54 | -- | -- | -- |
| cached_scores | 212.05 | 200.43 | 204.33 | 197.66 |
| historical_route_preqk_current_output | 209.90 | 203.09 | 203.87 | 198.74 |
| routing_only_current_output | 210.50 | 210.17 | 205.19 | 207.16 |
| M3_held | 212.66 | 203.00 | -- | -- |

On an ordinary step the pre-QK consumer beats `routing_only_current_output`
in both captures (203.09 vs 210.17 ms, and
198.74 vs 207.16 ms) and sits within ~1 ms of the
`cached_scores` floor (197.66 ms) -- the floor being the 0/4
configuration that never computes current QK at all. Dense remains fastest
(174.40 ms).

## 2. Complete `Attention.__call__`, capture B (the cleaner run)

| arm | layer 0 (local D=256) | layer 5 (global D=512) |
|---|---|---|
| cached_scores | anchor 1.096  ordinary 0.754 | anchor 1.024  ordinary 0.811 |
| historical_route_preqk_current_output | anchor 1.101  ordinary 0.820 | anchor 1.019  ordinary 0.815 |
| routing_only_current_output | anchor 1.101  ordinary 1.074 | anchor 1.032  ordinary 0.982 |
| native_dense | dense 0.060 | dense 0.082 |

**Correction to an earlier reading of capture A.** Capture A appeared to show
the pre-QK ordinary call *losing* at the local layer (1.240 vs 1.094 ms). That
measurement carried a 1527 ms first observation, i.e. Triton specialization
leaking into a short repetition series. Capture B, warmed, shows the pre-QK
call ahead at BOTH layers (0.820 vs 1.074 local, 0.815 vs 0.982 global). The
earlier 'local-layer regression' claim is withdrawn; there is no such
regression in the warmed measurement.

## 3. Scaling across real context lengths (attention only, 50% dropped)

A numerical step pays the selector **and** its consumer, so the ratio that
decides the method is `preqk + route` against `dense`.

| geometry | nk | dense | observe+PV | preqk | route_only | preqk+route | vs dense |
|---|---|---|---|---|---|---|---|
| local_D256_GQA16_8 | 384 | 0.146 | 0.327 | 0.130 | 0.152 | 0.282 | **1.93x** |
| local_D256_GQA16_8 | 1279 | 0.435 | 1.177 | 0.379 | 1.146 | 1.525 | **3.51x** |
| global_D512_GQA16_2 | 384 | 0.362 | 0.322 | 0.223 | 0.158 | 0.381 | **1.05x** |
| global_D512_GQA16_2 | 1279 | 1.229 | 1.549 | 0.620 | 1.046 | 1.666 | **1.36x** |
| global_D512_GQA16_2 | 2048 | 1.786 | 2.204 | 0.927 | 0.789 | 1.716 | **0.96x** |
| global_D512_GQA16_2 | 4096 | 3.559 | 4.426 | 1.749 | 1.440 | 3.189 | **0.90x** |
| global_D512_GQA16_2 | 8192 | 6.433 | 8.873 | 3.353 | 2.848 | 6.202 | **0.96x** |

Two findings, both measured:

1. **The pre-QK consumer scales well.** Against the materialized path it wins
   by more as context grows (at the local geometry `observe+PV` goes 0.327 ->
   1.177 ms while `preqk` stays 0.130 -> 0.379; at global 8192 it is 3.353 vs
   8.873). The earlier worry that its 16-row query program would reload K and
   lose at long context is not what the data shows.
2. **`route_only` is now the limiting cost, and it is worst exactly where the
   model spends most of its layers.** At the saturated local geometry
   (nk=1279, which is where 25 of 30 layers live, capped by
   `sliding_window-1 + canvas`) the selector alone costs 1.146 ms against
   0.435 ms for the *entire* dense attention of that layer -- 2.6x dense, paid
   on every decision-refresh call. That pushes `preqk+route` to **3.51x dense**
   on local layers, while on global layers at >=2048 keys it reaches
   **0.90-0.96x dense**, i.e. finally at parity.

The method is therefore dominated by selector cost on sliding layers, not by
the output consumer that this round rebuilt.
