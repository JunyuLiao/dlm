# Headroom and cost boundary

The geometry receipts isolate **one complete native attention call** at GLOBAL layer 5 and LOCAL layer 0 on real same-history M3 method-path QKV. The fixed-support coarse call is measured only for GLOBAL; support preparation and route costs are excluded. [Opportunity table](geometry_opportunity.md) and [raw scalar rows](geometry_opportunity.csv) give three accepted samples per state and both host identities. No fine-geometry consumer exists in these measurements, so no fine-geometry latency or full-request speedup is measured.

| Host | Task | GLOBAL native attention event ms | GLOBAL coarse fixed support event ms | LOCAL native attention event ms |
|---|---|---:|---:|---:|
| mpk | aime26 | 0.286 | 0.228 | 0.090 |
| mpk | longbench_v2 | 4.299 | 1.487 | 0.070 |
| mpk | ruler4k | 1.329 | 0.494 | 0.071 |
| dllm | aime26 | 0.325 | 0.269 | 0.057 |
| dllm | longbench_v2 | 5.365 | 1.683 | 0.064 |
| dllm | ruler4k | 1.282 | 0.424 | 0.066 |

These are medians of the per-state 3-repetition medians. They are descriptive across the reached early/later/last calls, not paired request estimators. For scale only, the v20 published mpk LongBench 16-call complete decoder-forward sum/16 was native **163.494 ms** and M3-R3 **155.474 ms** on a different captured state/measurement series; see [metric boundaries](metric_boundaries.md). A layer attention event of ~1–5 ms must not be subtracted directly from that full-forward event as if the clocks, context, and overlap were identical. CUDA event spans can contain host launch gaps and are not GPU-active time.

| Host | Task | Toy `5 × (GLOBAL native − coarse fixed)` ms | Toy `25 × LOCAL native` ms |
|---|---|---:|---:|
| mpk | aime26 | 0.290 | 2.257 |
| mpk | longbench_v2 | 14.061 | 1.752 |
| mpk | ruler4k | 4.171 | 1.763 |
| dllm | aime26 | 0.281 | 1.426 |
| dllm | longbench_v2 | 18.412 | 1.590 |
| dllm | ruler4k | 4.294 | 1.643 |

These products are arithmetic illustrations only: the first removes no selector/metadata overhead and assumes five layers share layer-5 prices; the second is the impossible reference of eliminating every LOCAL attention call at layer-0 price. They are **not** measured savings, attention-share estimates, or rigorous full-forward bounds.
A toy architecture count is five GLOBAL and twenty-five LOCAL decoder layers. Multiplying these two representative layer prices by 5 or 25 only gives a **planning extrapolation**: layer shapes, K/V extents, cache states, masks and launch overlap vary, and no all-layer timing or rigorous request bound was measured. The old cross-head KV32 path additionally needs bitmap addressing, selector state, projection/summary, layout and guard costs. The LOCAL opportunity is more constrained: its current-score-oracle error/work table is not a qualified deployed LOCAL policy.

The operational budget remains `net saved = removed dense attention work − sparse consumer overhead − selection/summary/export − layout/control/guard`. v20 natural LongBench R3 made **2514** calls versus native **2360** (1.0653×), so under an intentionally equal-call-cost approximation the successor needs per-call cost below **2360/2514 = 0.9387** merely to offset call inflation. This is not a forecast: prefill, commits, contexts, output canvases, sampler/host work, and changed trajectories decide full-request time. [Prior budget](refresh_cost_budget.md) and [natural adaptive panel](../fan_m1_m3_multidataset_20260927/adaptive_panel_report.md) retain those separate boundaries.

Decision gate: one candidate geometry requires a qualified complete attention consumer with actual QK/PV/KV-load counters and a matched-output numerical envelope, followed by direct full-forward/denoising-step and clean adaptive-request timing. A cheaper Torch logical support is an opportunity signal only.
