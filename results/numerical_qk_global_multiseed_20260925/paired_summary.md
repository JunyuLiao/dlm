# v13 paired summary (v13_global_multiseed_all30)

Scope: GLOBAL-only temporal methods (layers 5,11,17,23,29); LOCAL native in all arms. Questions: 30; generation seeds: [17, 29]; arms: B8_G, D_native, G1, G3, T_G.
Executions: 600/600 planned (complete).
Quality = attempt 0 (cap/unparsed/failure = incorrect). Time = accepted single warm API wall.
Ratios are candidate/reference: < 1 means less time. Speedup = 1/ratio.

## Quality (attempt 0)

| arm | seed 17 | seed 29 | combined | mean per question | caps | failures | not executed |
|---|---:|---:|---:|---:|---:|---:|---:|
| B8_G | 17/30 | 17/30 | 34/60 | 0.567 | 20 | 0 | 0 |
| D_native | 16/30 | 16/30 | 32/60 | 0.533 | 24 | 0 | 0 |
| G1 | 14/30 | 16/30 | 30/60 | 0.500 | 26 | 0 | 0 |
| G3 | 16/30 | 18/30 | 34/60 | 0.567 | 25 | 0 | 0 |
| T_G | 17/30 | 17/30 | 34/60 | 0.567 | 23 | 0 | 0 |

## Paired comparisons (all questions)

| pair | quality diff / question [95% CI] | cand correct | ref correct | disagreements (cand-only/ref-only) | geometric time ratio [95% CI] | summed time ratio [95% CI] | call factor | time/call factor |
|---|---|---:|---:|---|---|---|---:|---:|
| B8_G/D_native | +0.033 [-0.083, 0.167] | 34 | 32 | 7/5 | 0.949 [0.851, 1.052] | 0.927 [0.856, 0.997] | 0.907 | 1.022 |
| G1/B8_G | -0.067 [-0.217, 0.067] | 30 | 34 | 6/10 | 1.109 [1.016, 1.208] | 1.130 [1.064, 1.207] | 1.104 | 1.023 |
| G1/D_native | -0.033 [-0.167, 0.100] | 30 | 32 | 6/8 | 1.052 [0.957, 1.154] | 1.048 [0.972, 1.131] | 1.002 | 1.046 |
| G1/G3 | -0.067 [-0.200, 0.067] | 30 | 34 | 5/9 | 0.964 [0.898, 1.040] | 0.977 [0.912, 1.048] | 0.960 | 1.018 |
| G1/T_G | -0.067 [-0.167, 0.033] | 30 | 34 | 3/7 | 1.118 [1.017, 1.226] | 1.102 [1.019, 1.191] | 1.080 | 1.021 |
| G3/B8_G | +0.000 [-0.117, 0.117] | 34 | 34 | 4/4 | 1.150 [1.039, 1.283] | 1.156 [1.055, 1.277] | 1.150 | 1.005 |
| G3/D_native | +0.033 [-0.050, 0.117] | 34 | 32 | 5/3 | 1.091 [1.011, 1.184] | 1.072 [1.001, 1.152] | 1.043 | 1.027 |
| G3/T_G | +0.000 [-0.083, 0.083] | 34 | 34 | 5/5 | 1.160 [1.071, 1.259] | 1.127 [1.041, 1.227] | 1.124 | 1.003 |
| T_G/D_native | +0.033 [-0.100, 0.183] | 34 | 32 | 9/7 | 0.941 [0.868, 1.019] | 0.951 [0.885, 1.019] | 0.928 | 1.025 |

## Robustness

| pair | seed | geometric | summed | quality diff |
|---|---|---:|---:|---:|
| B8_G/D_native | 17 | 0.953 | 0.924 | +0.033 |
| B8_G/D_native | 29 | 0.945 | 0.930 | +0.033 |
| G1/B8_G | 17 | 1.175 | 1.169 | -0.100 |
| G1/B8_G | 29 | 1.046 | 1.091 | -0.033 |
| G1/D_native | 17 | 1.119 | 1.080 | -0.067 |
| G1/D_native | 29 | 0.989 | 1.015 | +0.000 |
| G1/G3 | 17 | 1.043 | 1.023 | -0.067 |
| G1/G3 | 29 | 0.891 | 0.933 | -0.067 |
| G1/T_G | 17 | 1.213 | 1.152 | -0.100 |
| G1/T_G | 29 | 1.031 | 1.053 | -0.033 |
| G3/B8_G | 17 | 1.126 | 1.143 | -0.033 |
| G3/B8_G | 29 | 1.174 | 1.169 | +0.033 |
| G3/D_native | 17 | 1.073 | 1.056 | +0.000 |
| G3/D_native | 29 | 1.109 | 1.087 | +0.067 |
| G3/T_G | 17 | 1.163 | 1.126 | -0.033 |
| G3/T_G | 29 | 1.157 | 1.128 | +0.033 |
| T_G/D_native | 17 | 0.923 | 0.938 | +0.033 |
| T_G/D_native | 29 | 0.959 | 0.964 | +0.033 |

| pair | stratum | questions | geometric | summed | cand correct | ref correct |
|---|---|---:|---:|---:|---:|---:|
| B8_G/D_native | development | 6 | 0.912 | 0.852 | 10 | 6 |
| B8_G/D_native | reserved | 24 | 0.958 | 0.945 | 24 | 26 |
| G1/B8_G | development | 6 | 1.098 | 1.195 | 7 | 10 |
| G1/B8_G | reserved | 24 | 1.111 | 1.116 | 23 | 24 |
| G1/D_native | development | 6 | 1.001 | 1.018 | 7 | 6 |
| G1/D_native | reserved | 24 | 1.065 | 1.054 | 23 | 26 |
| G1/G3 | development | 6 | 0.914 | 0.904 | 7 | 7 |
| G1/G3 | reserved | 24 | 0.977 | 0.995 | 23 | 27 |
| G1/T_G | development | 6 | 1.216 | 1.209 | 7 | 8 |
| G1/T_G | reserved | 24 | 1.095 | 1.080 | 23 | 26 |
| G3/B8_G | development | 6 | 1.201 | 1.321 | 7 | 10 |
| G3/B8_G | reserved | 24 | 1.138 | 1.122 | 27 | 24 |
| G3/D_native | development | 6 | 1.095 | 1.125 | 7 | 6 |
| G3/D_native | reserved | 24 | 1.090 | 1.060 | 27 | 26 |
| G3/T_G | development | 6 | 1.330 | 1.337 | 7 | 8 |
| G3/T_G | reserved | 24 | 1.121 | 1.086 | 27 | 26 |
| T_G/D_native | development | 6 | 0.823 | 0.842 | 8 | 6 |
| T_G/D_native | reserved | 24 | 0.973 | 0.976 | 26 | 26 |

### Leave-one-question-out (range over dropped question)

| pair | geometric min..max | summed min..max | question whose removal changes summed most |
|---|---|---|---|
| B8_G/D_native | 0.932..0.973 | 0.916..0.949 | aime26/23 (0.949) |
| G1/B8_G | 1.089..1.133 | 1.115..1.141 | aime26/23 (1.115) |
| G1/D_native | 1.031..1.078 | 1.035..1.069 | aime26/12 (1.069) |
| G1/G3 | 0.944..0.979 | 0.965..0.997 | aime26/25 (0.997) |
| G1/T_G | 1.096..1.144 | 1.088..1.121 | aime26/21 (1.121) |
| G3/B8_G | 1.124..1.167 | 1.128..1.174 | aime26/23 (1.128) |
| G3/D_native | 1.065..1.106 | 1.055..1.088 | aime26/7 (1.055) |
| G3/T_G | 1.136..1.182 | 1.101..1.146 | aime26/23 (1.101) |
| T_G/D_native | 0.926..0.960 | 0.937..0.972 | aime26/23 (0.972) |
