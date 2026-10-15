# Same-state final-layer decision diagnostic

not issued: instrumentation qualification on one LongBench item is not the multi-workload decision gate

| Workload | Arm | Frames | Top-1 disagreement | Acceptance disagreement | Mean KL |
|---|---|---:|---:|---:|---:|
| longbench_v2_0shot_think | correction_direction_guard | 12 | 0.016602 | 0.003906 | 0.000920 |
| longbench_v2_0shot_think | correction_moments | 12 | 0.012044 | 0.005208 | 0.000556 |
| longbench_v2_0shot_think | correction_norm_guard | 12 | 0.018229 | 0.004557 | 0.001306 |
| longbench_v2_0shot_think | correction_residual | 12 | 0.015299 | 0.003906 | 0.000856 |
| longbench_v2_0shot_think | full_cgate_swap8 | 12 | 0.017904 | 0.003581 | 0.001262 |
| longbench_v2_0shot_think | full_swap8 | 12 | 0.017578 | 0.003906 | 0.001267 |
| longbench_v2_0shot_think | mass_anchor | 12 | 0.018229 | 0.004557 | 0.001306 |
| longbench_v2_0shot_think | native_control | 12 | 0.019206 | 0.003581 | 0.001310 |
| longbench_v2_0shot_think | online_swap8 | 12 | 0.017904 | 0.004557 | 0.001263 |
| longbench_v2_0shot_think | same_mask_numerical_floor_vs_native | 12 | 0.007161 | 0.001628 | 0.000218 |
