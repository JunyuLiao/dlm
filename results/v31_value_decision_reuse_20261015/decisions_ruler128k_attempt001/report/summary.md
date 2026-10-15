# Same-state final-layer decision diagnostic

not issued by this report: interpret all workloads together before native development; no task-accuracy claim

| Workload | Arm | Frames | Top-1 disagreement | Acceptance disagreement | Mean KL |
|---|---|---:|---:|---:|---:|
| ruler128k_v33ofc_subset | correction_direction_guard | 14 | 0.003348 | 0.002232 | 0.002636 |
| ruler128k_v33ofc_subset | correction_moments | 14 | 0.002511 | 0.001116 | 0.001339 |
| ruler128k_v33ofc_subset | correction_norm_guard | 14 | 0.006696 | 0.001395 | 0.003727 |
| ruler128k_v33ofc_subset | correction_residual | 14 | 0.002790 | 0.001674 | 0.001783 |
| ruler128k_v33ofc_subset | full_cgate_swap8 | 14 | 0.006417 | 0.000558 | 0.002842 |
| ruler128k_v33ofc_subset | full_swap8 | 14 | 0.006417 | 0.000558 | 0.002928 |
| ruler128k_v33ofc_subset | full_swap8_direction_guard | 14 | 0.004185 | 0.001674 | 0.002107 |
| ruler128k_v33ofc_subset | mass_anchor | 14 | 0.006696 | 0.001116 | 0.003803 |
| ruler128k_v33ofc_subset | native_control | 14 | 0.006417 | 0.001116 | 0.003825 |
| ruler128k_v33ofc_subset | online_swap8 | 14 | 0.006975 | 0.000558 | 0.003218 |
| ruler128k_v33ofc_subset | same_mask_numerical_floor_vs_native | 14 | 0.000837 | 0.000000 | 0.000024 |
| ruler128k_v33ofc_subset | shuffled_cgate_swap8 | 14 | 0.006138 | 0.000558 | 0.002933 |
