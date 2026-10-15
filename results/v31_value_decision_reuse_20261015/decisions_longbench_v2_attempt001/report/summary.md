# Same-state final-layer decision diagnostic

not issued by this report: interpret all workloads together before native development; no task-accuracy claim

| Workload | Arm | Frames | Top-1 disagreement | Acceptance disagreement | Mean KL |
|---|---|---:|---:|---:|---:|
| longbench_v2_0shot_think | correction_direction_guard | 99 | 0.031250 | 0.004932 | 0.003494 |
| longbench_v2_0shot_think | correction_moments | 99 | 0.022609 | 0.004143 | 0.001719 |
| longbench_v2_0shot_think | correction_norm_guard | 99 | 0.041627 | 0.005366 | 0.007556 |
| longbench_v2_0shot_think | correction_residual | 99 | 0.028054 | 0.005208 | 0.002737 |
| longbench_v2_0shot_think | full_cgate_swap8 | 99 | 0.040799 | 0.005248 | 0.007161 |
| longbench_v2_0shot_think | full_swap8 | 99 | 0.040838 | 0.005208 | 0.007171 |
| longbench_v2_0shot_think | full_swap8_direction_guard | 99 | 0.031684 | 0.005129 | 0.003418 |
| longbench_v2_0shot_think | mass_anchor | 99 | 0.042140 | 0.005287 | 0.007675 |
| longbench_v2_0shot_think | native_control | 99 | 0.041035 | 0.004972 | 0.007682 |
| longbench_v2_0shot_think | online_swap8 | 99 | 0.041588 | 0.005011 | 0.007245 |
| longbench_v2_0shot_think | same_mask_numerical_floor_vs_native | 99 | 0.016967 | 0.000710 | 0.000336 |
| longbench_v2_0shot_think | shuffled_cgate_swap8 | 99 | 0.040365 | 0.005406 | 0.007172 |
