# v23 bootstrap6 reduction

First-generation outputs only were scored for quality; warm-repeat wall times are timing measurements, not independent samples; all intervals below are exploratory question-cluster bootstraps on an exposed development panel, not noninferiority tests.

## aime26

Complete blocks used: 8

| arm | n | strict_correct | capped | unparsed | total_calls | total_canvases | calls/canvas | total_out_tok | router A/D/H | warm_accepted | sum_warm_wall_s | amortized_ms/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_native | 8 | 4/8 | 2 | 2 | 1961 | 168 | 11.67 | 42289 | n/a | 8 | 285.7 | 145.7 |
  - per-host mean warm wall (s): 149.165.151.254: 38.95, 149.165.159.64: 32.47
| T_scope | 8 | 6/8 | 2 | 2 | 2020 | 162 | 12.47 | 40525 | n/a | 8 | 298.1 | 147.6 |
  - per-host mean warm wall (s): 149.165.151.254: 41.36, 149.165.159.64: 33.16
| M3_R3_A8_incumbent | 8 | 6/8 | 2 | 1 | 1855 | 173 | 10.72 | 43543 | 1530/2090/5655 | 8 | 275 | 148.2 |
  - per-host mean warm wall (s): 149.165.151.254: 35.76, 149.165.159.64: 32.98
| M1_boot | 8 | 6/8 | 2 | 2 | 1847 | 156 | 11.84 | 39164 | 600/7075/0 | 8 | 278.8 | 150.9 |
  - per-host mean warm wall (s): 149.165.151.254: 35.09, 149.165.159.64: 34.6
| M3_boot | 8 | 6/8 | 2 | 1 | 1967 | 164 | 11.99 | 41306 | 655/2040/5500 | 8 | 292.7 | 148.8 |
  - per-host mean warm wall (s): 149.165.151.254: 36.64, 149.165.159.64: 36.52
| B_boot | 8 | 5/8 | 2 | 2 | 1931 | 169 | 11.43 | 42583 | 605/0/7360 | 8 | 285.7 | 148 |
  - per-host mean warm wall (s): 149.165.151.254: 38.5, 149.165.159.64: 32.94

| contrast (arm1/arm2) | n | warm geo ratio | min | max | calls geo ratio | arm1✓arm2✗ | arm2✓arm1✗ | 95% CI (log warm ratio, exp) |
|---|---|---|---|---|---|---|---|---|
| M3_boot/D_native | 8 | 1.042 | 0.4691 | 1.547 | 1.022 | 2 | 0 | [0.8841, 1.269] (n_q=4) |
| M3_boot/M3_R3_A8_incumbent | 8 | 1.017 | 0.5352 | 2.146 | 1.017 | 0 | 0 | [0.7201, 1.286] (n_q=4) |
| M3_boot/B_boot | 8 | 0.9839 | 0.6733 | 1.167 | 0.9772 | 1 | 0 | [0.9103, 1.079] (n_q=4) |
| M3_boot/T_scope | 8 | 1.031 | 0.5975 | 2.17 | 1.021 | 0 | 0 | [0.9846, 1.101] (n_q=4) |
| M1_boot/D_native | 8 | 0.975 | 0.672 | 1.562 | 0.9457 | 2 | 0 | [0.8468, 1.085] (n_q=4) |
| B_boot/D_native | 8 | 1.059 | 0.4948 | 2.298 | 1.045 | 2 | 1 | [0.8806, 1.382] (n_q=4) |
| M3_R3_A8_incumbent/D_native | 8 | 1.024 | 0.6627 | 2.34 | 1.004 | 2 | 0 | [0.7623, 1.733] (n_q=4) |
| T_scope/D_native | 8 | 1.011 | 0.4907 | 2.589 | 1.001 | 2 | 0 | [0.871, 1.158] (n_q=4) |
| M1_boot/M3_boot | 8 | 0.9361 | 0.5342 | 1.433 | 0.9257 | 0 | 0 | [0.6822, 1.196] (n_q=4) |

## longbench_v2

Complete blocks used: 12

| arm | n | strict_correct | capped | unparsed | total_calls | total_canvases | calls/canvas | total_out_tok | router A/D/H | warm_accepted | sum_warm_wall_s | amortized_ms/call |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_native | 12 | 6/12 | 0 | 0 | 2360 | 140 | 16.86 | 34130 | n/a | 12 | 442.9 | 187.7 |
  - per-host mean warm wall (s): 149.165.151.254: 40.37, 149.165.159.64: 33.44
| T_scope | 12 | 6/12 | 0 | 0 | 2223 | 128 | 17.37 | 31468 | n/a | 12 | 411.5 | 185.1 |
  - per-host mean warm wall (s): 149.165.151.254: 33.98, 149.165.159.64: 34.6
| M3_R3_A8_incumbent | 12 | 5/12 | 0 | 0 | 2604 | 145 | 17.96 | 35484 | 1940/3070/8010 | 12 | 460.6 | 176.9 |
  - per-host mean warm wall (s): 149.165.151.254: 41.65, 149.165.159.64: 35.11
| M1_boot | 12 | 5/12 | 0 | 0 | 2623 | 147 | 17.84 | 36315 | 1140/10505/0 | 12 | 485.3 | 185 |
  - per-host mean warm wall (s): 149.165.151.254: 37.96, 149.165.159.64: 42.93
| M3_boot | 12 | 6/12 | 0 | 0 | 2415 | 143 | 16.89 | 35312 | 995/2685/6965 | 12 | 431.5 | 178.7 |
  - per-host mean warm wall (s): 149.165.151.254: 34.83, 149.165.159.64: 37.09
| B_boot | 12 | 4/12 | 0 | 0 | 2524 | 144 | 17.53 | 35485 | 1060/0/10120 | 12 | 439.9 | 174.3 |
  - per-host mean warm wall (s): 149.165.151.254: 36.58, 149.165.159.64: 36.74

| contrast (arm1/arm2) | n | warm geo ratio | min | max | calls geo ratio | arm1✓arm2✗ | arm2✓arm1✗ | 95% CI (log warm ratio, exp) |
|---|---|---|---|---|---|---|---|---|
| M3_boot/D_native | 12 | 0.9901 | 0.5347 | 1.932 | 1.04 | 0 | 0 | [0.9438, 1.051] (n_q=6) |
| M3_boot/M3_R3_A8_incumbent | 12 | 0.9997 | 0.406 | 1.999 | 0.9936 | 1 | 0 | [0.8289, 1.22] (n_q=6) |
| M3_boot/B_boot | 12 | 0.9968 | 0.7696 | 1.989 | 0.9755 | 2 | 0 | [0.9438, 1.09] (n_q=6) |
| M3_boot/T_scope | 12 | 1.095 | 0.7341 | 2.37 | 1.145 | 0 | 0 | [0.9339, 1.298] (n_q=6) |
| M1_boot/D_native | 12 | 1.058 | 0.5351 | 1.661 | 1.069 | 0 | 1 | [0.8521, 1.3] (n_q=6) |
| B_boot/D_native | 12 | 0.9934 | 0.6947 | 2.005 | 1.066 | 0 | 2 | [0.9055, 1.09] (n_q=6) |
| M3_R3_A8_incumbent/D_native | 12 | 0.9904 | 0.5528 | 1.864 | 1.047 | 0 | 1 | [0.796, 1.226] (n_q=6) |
| T_scope/D_native | 12 | 0.9041 | 0.3923 | 1.248 | 0.9081 | 0 | 0 | [0.7581, 1.05] (n_q=6) |
| M1_boot/M3_boot | 12 | 1.068 | 0.4816 | 1.66 | 1.028 | 0 | 1 | [0.8636, 1.3] (n_q=6) |

