# Same-state final-layer decision diagnostic

not issued by this report: interpret all workloads together before native development; no task-accuracy claim

| Workload | Arm | Frames | Top-1 disagreement | Acceptance disagreement | Mean KL |
|---|---|---:|---:|---:|---:|
| math500 | correction_direction_guard | 75 | 0.005104 | 0.001094 | 0.000167 |
| math500 | correction_moments | 75 | 0.005052 | 0.001615 | 0.000154 |
| math500 | correction_norm_guard | 75 | 0.005573 | 0.001042 | 0.000183 |
| math500 | correction_residual | 75 | 0.005052 | 0.000990 | 0.000153 |
| math500 | full_cgate_swap8 | 75 | 0.005104 | 0.000729 | 0.000180 |
| math500 | full_swap8 | 75 | 0.005260 | 0.000990 | 0.000177 |
| math500 | full_swap8_direction_guard | 75 | 0.005417 | 0.000885 | 0.000165 |
| math500 | mass_anchor | 75 | 0.005573 | 0.001042 | 0.000183 |
| math500 | native_control | 75 | 0.005990 | 0.001354 | 0.000185 |
| math500 | online_swap8 | 75 | 0.005260 | 0.000990 | 0.000177 |
| math500 | same_mask_numerical_floor_vs_native | 75 | 0.005104 | 0.000937 | 0.000132 |
| math500 | shuffled_cgate_swap8 | 75 | 0.005208 | 0.000677 | 0.000177 |
