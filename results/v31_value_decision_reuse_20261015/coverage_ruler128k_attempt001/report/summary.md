# Same-state final-layer decision diagnostic

not issued by this report: interpret all workloads together before native development; no task-accuracy claim

| Workload | Arm | Frames | Top-1 disagreement | Acceptance disagreement | Mean KL |
|---|---|---:|---:|---:|---:|
| ruler128k_v33ofc_subset | correction_direction_guard | 42 | 0.001860 | 0.003162 | 0.003169 |
| ruler128k_v33ofc_subset | correction_moments | 42 | 0.001581 | 0.003906 | 0.001946 |
| ruler128k_v33ofc_subset | correction_norm_guard | 42 | 0.002511 | 0.004836 | 0.004216 |
| ruler128k_v33ofc_subset | correction_residual | 42 | 0.001767 | 0.003627 | 0.003034 |
| ruler128k_v33ofc_subset | full_cgate_swap8 | 42 | 0.002232 | 0.004929 | 0.004133 |
| ruler128k_v33ofc_subset | full_swap8 | 42 | 0.002325 | 0.004743 | 0.004155 |
| ruler128k_v33ofc_subset | full_swap8_direction_guard | 42 | 0.001860 | 0.003441 | 0.003046 |
| ruler128k_v33ofc_subset | mass_anchor | 42 | 0.002511 | 0.004836 | 0.004320 |
| ruler128k_v33ofc_subset | native_control | 42 | 0.002511 | 0.004650 | 0.004354 |
| ruler128k_v33ofc_subset | online_swap8 | 42 | 0.002232 | 0.004743 | 0.004180 |
| ruler128k_v33ofc_subset | same_mask_numerical_floor_vs_native | 42 | 0.000000 | 0.000372 | 0.000032 |
| ruler128k_v33ofc_subset | shuffled_cgate_swap8 | 42 | 0.002325 | 0.004929 | 0.004126 |
