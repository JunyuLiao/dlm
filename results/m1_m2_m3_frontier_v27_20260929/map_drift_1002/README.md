# Map drift between re-decisions within a canvas (2026-10-02)

`scripts/v27_map_drift_diag.py` on dllm with the E5 frozen configuration `M3_R6_A64_fused_dp_async_m1ln2_c0_fa4`,
LongBench-v2 64K and 32K items 0–3, seed 404 (diagnostic, no timing claims). For each GLOBAL layer and canvas, the first
decision's kept prefix map is compared with every later decision (calls 8, 14, …) of the same canvas.

| request | re-decisions | changed tiles / first kept | kept-set Jaccard |
|---|---:|---:|---:|
| 64K item 0 | 40 | 0.62 | 0.62 |
| 64K item 1 | 65 | 0.71 | 0.59 |
| 64K item 2 | 295 | 0.52 | 0.66 |
| 64K item 3 | 175 | 0.50 | 0.66 |
| 32K item 0 | 170 | 0.52 | 0.66 |
| 32K item 1 | 50 | 0.44 | 0.70 |
| 32K item 2 | 65 | 0.61 | 0.62 |
| 32K item 3 | 100 | 0.63 | 0.61 |

Every re-decision changed the map. A follow-up debug run showed the prefix risk table (checksum), the V reference
scale and the threshold are identical across decisions of one canvas; what changes is the per-position query
sensitivity T = clamp(1 + 3·EMA(argmax flips, γ = 0.5), 1, 4) set every step by `NativeReuseState('T')`
(bound parent config `beta=3.0, gamma=0.5, fast_t=True`). This corrects the 2026-10-01 statements that T = 1 and that
re-decisions barely change the map. Raw rows: `drift_e5_items0-3.jsonl`.
