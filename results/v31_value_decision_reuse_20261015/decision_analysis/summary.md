# Paired item-level decision analysis

eager final-GLOBAL-layer same-state diagnostics on previously examined prompts.

Only sparse frames enter this analysis. Repetitions do not increase sample size. 
The numerical-floor row measures a different reference and is excluded from the treatment table.

| Workload | Variant | Items | Frames | Top1 disagree rows | Acceptance disagree rows | KL delta, item mean | Top1 better / worse items |
|---|---|---:|---:|---:|---:|---:|---|
| aime26 | correction_direction_guard | 5 | 35 | 24 | 12 | -2.27848e-06 | 2 / 1 |
| aime26 | correction_moments | 5 | 35 | 28 | 13 | -5.34642e-06 | 3 / 1 |
| aime26 | correction_norm_guard | 5 | 35 | 26 | 16 | +0 | 0 / 0 |
| aime26 | correction_residual | 5 | 35 | 19 | 7 | -4.19609e-06 | 3 / 0 |
| aime26 | full_cgate_swap8 | 5 | 35 | 23 | 15 | -2.82453e-06 | 3 / 1 |
| aime26 | full_swap8 | 5 | 35 | 21 | 14 | -2.68828e-06 | 3 / 1 |
| aime26 | full_swap8_direction_guard | 5 | 35 | 20 | 11 | -3.96597e-06 | 3 / 0 |
| aime26 | mass_anchor | 5 | 35 | 26 | 16 | +0 | 0 / 0 |
| aime26 | native_control | 5 | 35 | 28 | 13 | +2.69032e-06 | 1 / 3 |
| aime26 | online_swap8 | 5 | 35 | 21 | 14 | -2.68828e-06 | 3 / 1 |
| aime26 | shuffled_cgate_swap8 | 5 | 35 | 22 | 17 | -2.67977e-06 | 3 / 1 |
| longbench_v2 | correction_direction_guard | 8 | 99 | 792 | 125 | -0.00360441 | 8 / 0 |
| longbench_v2 | correction_moments | 8 | 99 | 573 | 105 | -0.00517512 | 8 / 0 |
| longbench_v2 | correction_norm_guard | 8 | 99 | 1055 | 136 | -9.86482e-05 | 1 / 2 |
| longbench_v2 | correction_residual | 8 | 99 | 711 | 132 | -0.00425834 | 8 / 0 |
| longbench_v2 | full_cgate_swap8 | 8 | 99 | 1034 | 133 | -0.000438765 | 7 / 1 |
| longbench_v2 | full_swap8 | 8 | 99 | 1035 | 132 | -0.000429215 | 5 / 2 |
| longbench_v2 | full_swap8_direction_guard | 8 | 99 | 803 | 130 | -0.00367135 | 8 / 0 |
| longbench_v2 | mass_anchor | 8 | 99 | 1068 | 134 | +0 | 0 / 0 |
| longbench_v2 | native_control | 8 | 99 | 1040 | 126 | +4.21186e-06 | 5 / 3 |
| longbench_v2 | online_swap8 | 8 | 99 | 1054 | 127 | -0.000365378 | 6 / 2 |
| longbench_v2 | shuffled_cgate_swap8 | 8 | 99 | 1023 | 137 | -0.000428661 | 5 / 2 |
| ruler128k | correction_direction_guard | 5 | 14 | 12 | 8 | -0.000832582 | 2 / 0 |
| ruler128k | correction_moments | 5 | 14 | 9 | 4 | -0.0018825 | 2 / 1 |
| ruler128k | correction_norm_guard | 5 | 14 | 24 | 5 | -4.32015e-05 | 0 / 0 |
| ruler128k | correction_residual | 5 | 14 | 10 | 6 | -0.00162588 | 2 / 0 |
| ruler128k | full_cgate_swap8 | 5 | 14 | 23 | 2 | -0.000870765 | 1 / 1 |
| ruler128k | full_swap8 | 5 | 14 | 23 | 2 | -0.000788224 | 1 / 1 |
| ruler128k | full_swap8_direction_guard | 5 | 14 | 15 | 6 | -0.00132172 | 2 / 0 |
| ruler128k | mass_anchor | 5 | 14 | 24 | 4 | +0 | 0 / 0 |
| ruler128k | native_control | 5 | 14 | 23 | 4 | +2.14562e-05 | 1 / 0 |
| ruler128k | online_swap8 | 5 | 14 | 25 | 2 | -0.000524169 | 0 / 1 |
| ruler128k | shuffled_cgate_swap8 | 5 | 14 | 22 | 2 | -0.000785056 | 1 / 0 |
