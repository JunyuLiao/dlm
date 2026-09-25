# v12 scope and method contract (GLOBAL-only, native LOCAL)

This is a new, explicitly labeled scope experiment; it is not a Fan instruction. The temporal methods act only on
the five GLOBAL layers (5, 11, 17, 23, 29). All 25 LOCAL layers call the ORIGINAL native SDPA with the original
inputs in every arm; the bounded dispatch probe checks this per step. GLOBAL legality is the model's own
unrestricted domain, and a routed call carrying a mask fails explicitly. v11's all-layer legacy-mask results are
context only, not pooled with these arms.

| arm | GLOBAL attention | decision clock | output arithmetic |
|---|---|---|---|
| D_native | native SDPA | - | native |
| T_G | fresh Junyu T (routing from current QK every step) | every step | Junyu fused kernel |
| G1_split | M1: A8 historical scores + CURRENT projected V + live T/ref, exact GLOBAL prefix summaries | every step | anchor: Triton fused PV; ordinary: Hopper support consumer |
| G3_split | same as G1 | every 2nd step (held = last G1 map) | same |
| B8_G | same anchors and anchor selector (legacy recompute, no summaries) | only at A8 observations / canvas reset | same |

B8_G is a simple frozen value-aware bitmap baseline. It is not the historical G75 mass ranking and not M2.
The threshold is T_s50 GLOBAL (-3.137). Achieved skip fractions are reported rather than assumed; the first
4 canvases of /2 skipped only about 3.5% of GLOBAL tiles.

GLOBAL summary qualification (real /2 trajectory, 4 canvases): 205 decisions bit-identical with summaries on vs
off, 35 builds, 170 hits, 0 misses, 35.7 MB peak.
