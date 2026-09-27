# Adaptive panel: full 700-execution report

All 700 planned executions have ledger rows; all 50 question-seed blocks have seven successful first outputs and seven accepted warm pairs (350 accepted warm requests). Quality uses first outputs only. RULER covers all 13 pinned tasks at seeds 101/202; AIME and LongBench-v2 each cover six pinned questions at both seeds. No answer, gold, prompt or raw completion is included.

Accepted warm whole-request wall includes prefill, all decoder/commit/sampler/output work and excludes model load. Absolute means are shown by host only. The optional CUDA event span begins at the first actual encoder-forward end and ends after generate; it includes host gaps and later encoder/commit work, and is not synchronized prefill-excluded generation wall. Paired method/reference ratio <1 is faster. Per-call wall is amortized request wall divided by actual calls, not direct forward latency.

Protocol v20_fan_4a8f6a589714623e; panel SHA-256 99e1d287867b0906b8c5a2715d4cf962ec31e6d1f473c1fe0ce2df5891f1aa17; binding SHA-256 ee06c936e500fecfd219c6a9e277f99e77baed7b39d405ad96625f94e50734b5; redacted source SHA-256 25f5483e6650c118744fd642ecd3ec62d3178514eb8e8b2a0d730f1227ee9d5f.

## ruler4k (13 questions × 2 seeds)

| Arm | Task score mean | Correct | Official RULER task macro | Strict EOS | Task at cap | Parsed | EOS wrong | Output cap | Canvas cap | Native stop | Calls total | Calls/request | Calls/canvas pooled; per-request med/p90 | Tokens total; per-canvas pooled/med/p90 | A/D/H layer calls | First fail/missing; warm accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| B_A8_matched | 0.923 | 24/26 | 0.923 | 24 | 0 | 26/26 | 0 | 2 | 0 | 26 | 117 | 4.50 | 4.50; 3.50/9.50 | 1046; 40.23/28.00/79.00 | 145/0/440 | 0/0; 26/26 |
| D_matched | 0.923 | 24/26 | 0.923 | 24 | 0 | 26/26 | 0 | 2 | 0 | 26 | 104 | 4.00 | 4.00; 4.00/6.50 | 1051; 40.42/27.50/79.00 | 520/0/0 | 0/0; 26/26 |
| D_native | 0.923 | 24/26 | 0.923 | 24 | 0 | 26/26 | 0 | 2 | 0 | 26 | 99 | 3.81 | 3.81; 4.00/5.00 | 1049; 40.35/28.00/79.00 | N/A/N/A/N/A | 0/0; 26/26 |
| M1_R1_A8_current_output | 0.923 | 24/26 | 0.923 | 24 | 0 | 26/26 | 0 | 2 | 0 | 26 | 106 | 4.08 | 4.08; 4.00/6.50 | 1047; 40.27/28.00/79.00 | 140/390/0 | 0/0; 26/26 |
| M3_R2_A8_current_output | 0.923 | 24/26 | 0.923 | 24 | 0 | 26/26 | 0 | 2 | 0 | 26 | 114 | 4.38 | 4.38; 3.50/8.50 | 1046; 40.23/28.00/79.00 | 145/170/255 | 0/0; 26/26 |
| M3_R3_A8_current_output | 0.923 | 24/26 | 0.923 | 24 | 0 | 26/26 | 0 | 2 | 0 | 26 | 110 | 4.23 | 4.23; 3.50/7.50 | 1046; 40.23/28.00/79.00 | 145/85/320 | 0/0; 26/26 |
| T_scope | 0.923 | 24/26 | 0.923 | 24 | 0 | 26/26 | 0 | 2 | 0 | 26 | 98 | 3.77 | 3.77; 3.50/5.00 | 1047; 40.27/28.00/79.00 | N/A/N/A/N/A | 0/0; 26/26 |

Absolute accepted-warm timing, separately by host:

| Arm | Host | Accepted warm requests | Whole-request wall mean (s) | CUDA first-encoder-end to finish span mean (s) |
|---|---|---:|---:|---:|
| B_A8_matched | 149.165.151.254 | 13 | 1.216 | 0.812 |
| B_A8_matched | 149.165.159.64 | 13 | 0.853 | 0.467 |
| D_matched | 149.165.151.254 | 13 | 1.002 | 0.600 |
| D_matched | 149.165.159.64 | 13 | 0.881 | 0.494 |
| D_native | 149.165.151.254 | 13 | 0.993 | 0.591 |
| D_native | 149.165.159.64 | 13 | 0.844 | 0.458 |
| M1_R1_A8_current_output | 149.165.151.254 | 13 | 1.127 | 0.722 |
| M1_R1_A8_current_output | 149.165.159.64 | 13 | 0.844 | 0.456 |
| M3_R2_A8_current_output | 149.165.151.254 | 13 | 1.236 | 0.833 |
| M3_R2_A8_current_output | 149.165.159.64 | 13 | 0.826 | 0.440 |
| M3_R3_A8_current_output | 149.165.151.254 | 13 | 1.162 | 0.759 |
| M3_R3_A8_current_output | 149.165.159.64 | 13 | 0.835 | 0.449 |
| T_scope | 149.165.151.254 | 13 | 0.994 | 0.589 |
| T_scope | 149.165.159.64 | 13 | 0.866 | 0.463 |

Same-question/seed/GPU paired request ratios and question-cluster exploratory intervals:

| Method/reference | Pairs | Geometric ratio | 95% exploratory cluster interval | Paired score delta; interval | .254 total ratio / calls ratio / amortized wall-call | .64 total ratio / calls ratio / amortized wall-call |
|---|---:|---:|---|---|---|---|
| B_A8_matched/D_native | 26 | 1.068 | 0.956..1.213 | 0.000; 0.000..0.000 | n=13: 1.224/1.333/0.918 | n=13: 1.011/1.000/1.011 |
| B_A8_matched/T_scope | 26 | 1.064 | 0.977..1.175 | 0.000; 0.000..0.000 | n=13: 1.223/1.358/0.900 | n=13: 0.985/1.000/0.985 |
| D_matched/D_native | 26 | 1.014 | 0.949..1.074 | 0.000; 0.000..0.000 | n=13: 1.009/1.019/0.991 | n=13: 1.043/1.089/0.958 |
| D_matched/T_scope | 26 | 1.010 | 0.958..1.072 | 0.000; 0.000..0.000 | n=13: 1.009/1.038/0.972 | n=13: 1.017/1.089/0.934 |
| D_matched/B_A8_matched | 26 | 0.950 | 0.861..1.047 | 0.000; 0.000..0.000 | n=13: 0.825/0.764/1.079 | n=13: 1.032/1.089/0.948 |
| M1_R1_A8_current_output/D_native | 26 | 1.054 | 0.982..1.131 | 0.000; 0.000..0.000 | n=13: 1.135/1.167/0.972 | n=13: 1.000/0.956/1.047 |
| M1_R1_A8_current_output/T_scope | 26 | 1.050 | 0.999..1.117 | 0.000; 0.000..0.000 | n=13: 1.134/1.189/0.954 | n=13: 0.975/0.956/1.020 |
| M1_R1_A8_current_output/B_A8_matched | 26 | 0.987 | 0.917..1.050 | 0.000; 0.000..0.000 | n=13: 0.927/0.875/1.059 | n=13: 0.990/0.956/1.036 |
| M3_R2_A8_current_output/D_native | 26 | 1.075 | 0.976..1.189 | 0.000; 0.000..0.000 | n=13: 1.245/1.333/0.934 | n=13: 0.979/0.933/1.049 |
| M3_R2_A8_current_output/T_scope | 26 | 1.071 | 0.989..1.171 | 0.000; 0.000..0.000 | n=13: 1.244/1.358/0.916 | n=13: 0.954/0.933/1.023 |
| M3_R2_A8_current_output/B_A8_matched | 26 | 1.007 | 0.969..1.044 | 0.000; 0.000..0.000 | n=13: 1.017/1.000/1.017 | n=13: 0.969/0.933/1.038 |
| M3_R3_A8_current_output/D_native | 26 | 1.048 | 0.954..1.157 | 0.000; 0.000..0.000 | n=13: 1.170/1.241/0.943 | n=13: 0.989/0.956/1.035 |
| M3_R3_A8_current_output/T_scope | 26 | 1.044 | 0.973..1.136 | 0.000; 0.000..0.000 | n=13: 1.169/1.264/0.925 | n=13: 0.964/0.956/1.009 |
| M3_R3_A8_current_output/B_A8_matched | 26 | 0.981 | 0.944..1.011 | 0.000; 0.000..0.000 | n=13: 0.956/0.931/1.027 | n=13: 0.978/0.956/1.024 |
| T_scope/D_native | 26 | 1.004 | 0.943..1.063 | 0.000; 0.000..0.000 | n=13: 1.001/0.981/1.020 | n=13: 1.026/1.000/1.026 |
| T_scope/B_A8_matched | 26 | 0.940 | 0.838..1.025 | 0.000; 0.000..0.000 | n=13: 0.818/0.736/1.111 | n=13: 1.015/1.000/1.015 |

Per-canvas call histogram, call positions 0/1/2+, terminations, output-token distribution and per-host timing values are preserved in the CSV and byte-exact redacted JSON. The intervals cluster by question with both seeds kept together; they are exploratory, not a noninferiority claim.

## aime26 (6 questions × 2 seeds)

| Arm | Task score mean | Correct | Official RULER task macro | Strict EOS | Task at cap | Parsed | EOS wrong | Output cap | Canvas cap | Native stop | Calls total | Calls/request | Calls/canvas pooled; per-request med/p90 | Tokens total; per-canvas pooled/med/p90 | A/D/H layer calls | First fail/missing; warm accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| B_A8_matched | 0.583 | 7/12 | N/A | 7 | 0 | 8/12 | 0 | 5 | 0 | 291 | 3676 | 306.33 | 12.63; 11.09/16.49 | 73653; 253.10/254.05/256.00 | 2935/0/15445 | 0/0; 12/12 |
| D_matched | 0.583 | 7/12 | N/A | 7 | 0 | 8/12 | 1 | 4 | 0 | 263 | 3274 | 272.83 | 12.45; 11.78/14.29 | 66405; 252.49/251.88/256.00 | 16370/0/0 | 0/0; 12/12 |
| D_native | 0.417 | 5/12 | N/A | 5 | 0 | 7/12 | 2 | 5 | 0 | 283 | 3600 | 300.00 | 12.72; 11.18/17.11 | 71487; 252.60/251.89/256.00 | N/A/N/A/N/A | 0/0; 12/12 |
| M1_R1_A8_current_output | 0.583 | 7/12 | N/A | 7 | 0 | 8/12 | 1 | 4 | 0 | 269 | 3595 | 299.58 | 13.36; 12.83/16.62 | 67944; 252.58/254.13/256.00 | 2845/15130/0 | 0/0; 12/12 |
| M3_R2_A8_current_output | 0.500 | 6/12 | N/A | 6 | 0 | 7/12 | 1 | 5 | 1 | 293 | 3883 | 323.58 | 13.21; 13.01/16.12 | 74370; 252.96/254.32/256.00 | 3085/7010/9320 | 0/0; 12/12 |
| M3_R3_A8_current_output | 0.417 | 5/12 | N/A | 5 | 1 | 7/12 | 1 | 6 | 0 | 300 | 3866 | 322.17 | 12.89; 11.12/16.41 | 75994; 253.31/254.92/256.00 | 3050/4470/11810 | 0/0; 12/12 |
| T_scope | 0.667 | 8/12 | N/A | 8 | 0 | 8/12 | 0 | 4 | 1 | 270 | 3686 | 307.17 | 13.60; 12.00/17.19 | 68173; 251.56/248.81/256.00 | N/A/N/A/N/A | 0/0; 12/12 |

Absolute accepted-warm timing, separately by host:

| Arm | Host | Accepted warm requests | Whole-request wall mean (s) | CUDA first-encoder-end to finish span mean (s) |
|---|---|---:|---:|---:|
| B_A8_matched | 149.165.151.254 | 6 | 55.564 | 55.413 |
| B_A8_matched | 149.165.159.64 | 6 | 35.167 | 35.028 |
| D_matched | 149.165.151.254 | 6 | 52.319 | 52.167 |
| D_matched | 149.165.159.64 | 6 | 27.650 | 27.511 |
| D_native | 149.165.151.254 | 6 | 54.415 | 54.258 |
| D_native | 149.165.159.64 | 6 | 33.592 | 33.453 |
| M1_R1_A8_current_output | 149.165.151.254 | 6 | 53.314 | 53.163 |
| M1_R1_A8_current_output | 149.165.159.64 | 6 | 37.810 | 37.669 |
| M3_R2_A8_current_output | 149.165.151.254 | 6 | 63.169 | 63.015 |
| M3_R2_A8_current_output | 149.165.159.64 | 6 | 33.687 | 33.548 |
| M3_R3_A8_current_output | 149.165.151.254 | 6 | 56.025 | 55.874 |
| M3_R3_A8_current_output | 149.165.159.64 | 6 | 40.119 | 39.980 |
| T_scope | 149.165.151.254 | 6 | 53.240 | 53.089 |
| T_scope | 149.165.159.64 | 6 | 37.289 | 37.150 |

Same-question/seed/GPU paired request ratios and question-cluster exploratory intervals:

| Method/reference | Pairs | Geometric ratio | 95% exploratory cluster interval | Paired score delta; interval | .254 total ratio / calls ratio / amortized wall-call | .64 total ratio / calls ratio / amortized wall-call |
|---|---:|---:|---|---|---|---|
| B_A8_matched/D_native | 12 | 1.057 | 0.887..1.407 | 0.167; 0.000..0.500 | n=6: 1.021/1.004/1.017 | n=6: 1.047/1.048/0.999 |
| B_A8_matched/T_scope | 12 | 1.045 | 0.878..1.287 | -0.083; -0.250..0.000 | n=6: 1.044/1.038/1.006 | n=6: 0.943/0.942/1.001 |
| D_matched/D_native | 12 | 0.915 | 0.858..0.979 | 0.167; 0.000..0.500 | n=6: 0.961/0.961/1.000 | n=6: 0.823/0.829/0.993 |
| D_matched/T_scope | 12 | 0.905 | 0.776..1.021 | -0.083; -0.250..0.000 | n=6: 0.983/0.993/0.989 | n=6: 0.742/0.746/0.995 |
| D_matched/B_A8_matched | 12 | 0.866 | 0.620..1.089 | 0.000; 0.000..0.000 | n=6: 0.942/0.957/0.984 | n=6: 0.786/0.791/0.994 |
| M1_R1_A8_current_output/D_native | 12 | 1.042 | 0.986..1.097 | 0.167; 0.000..0.500 | n=6: 0.980/0.933/1.051 | n=6: 1.126/1.102/1.022 |
| M1_R1_A8_current_output/T_scope | 12 | 1.030 | 0.916..1.148 | -0.083; -0.250..0.000 | n=6: 1.001/0.964/1.039 | n=6: 1.014/0.991/1.023 |
| M1_R1_A8_current_output/B_A8_matched | 12 | 0.986 | 0.740..1.195 | 0.000; 0.000..0.000 | n=6: 0.960/0.929/1.033 | n=6: 1.075/1.052/1.022 |
| M3_R2_A8_current_output/D_native | 12 | 1.096 | 0.913..1.325 | 0.083; 0.000..0.250 | n=6: 1.161/1.128/1.029 | n=6: 1.003/1.001/1.001 |
| M3_R2_A8_current_output/T_scope | 12 | 1.083 | 0.939..1.301 | -0.167; -0.333..0.000 | n=6: 1.186/1.166/1.018 | n=6: 0.903/0.901/1.003 |
| M3_R2_A8_current_output/B_A8_matched | 12 | 1.036 | 0.866..1.237 | -0.083; -0.250..0.000 | n=6: 1.137/1.124/1.012 | n=6: 0.958/0.956/1.002 |
| M3_R3_A8_current_output/D_native | 12 | 1.135 | 0.960..1.368 | 0.000; -0.250..0.250 | n=6: 1.030/1.003/1.026 | n=6: 1.194/1.184/1.009 |
| M3_R3_A8_current_output/T_scope | 12 | 1.122 | 0.944..1.381 | -0.250; -0.417..-0.083 | n=6: 1.052/1.037/1.015 | n=6: 1.076/1.065/1.010 |
| M3_R3_A8_current_output/B_A8_matched | 12 | 1.073 | 0.888..1.281 | -0.167; -0.333..0.000 | n=6: 1.008/0.999/1.009 | n=6: 1.141/1.130/1.009 |
| T_scope/D_native | 12 | 1.011 | 0.905..1.118 | 0.250; 0.000..0.583 | n=6: 0.978/0.968/1.011 | n=6: 1.110/1.112/0.999 |
| T_scope/B_A8_matched | 12 | 0.957 | 0.777..1.139 | 0.083; 0.000..0.250 | n=6: 0.958/0.964/0.994 | n=6: 1.060/1.061/0.999 |

Per-canvas call histogram, call positions 0/1/2+, terminations, output-token distribution and per-host timing values are preserved in the CSV and byte-exact redacted JSON. The intervals cluster by question with both seeds kept together; they are exploratory, not a noninferiority claim.

## longbench_v2 (6 questions × 2 seeds)

| Arm | Task score mean | Correct | Official RULER task macro | Strict EOS | Task at cap | Parsed | EOS wrong | Output cap | Canvas cap | Native stop | Calls total | Calls/request | Calls/canvas pooled; per-request med/p90 | Tokens total; per-canvas pooled/med/p90 | A/D/H layer calls | First fail/missing; warm accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| B_A8_matched | 0.333 | 4/12 | N/A | 4 | 0 | 12/12 | 8 | 0 | 0 | 153 | 2779 | 231.58 | 18.16; 17.33/23.00 | 37357; 244.16/245.77/247.53 | 2060/0/11835 | 0/0; 12/12 |
| D_matched | 0.333 | 4/12 | N/A | 4 | 0 | 11/12 | 8 | 0 | 12 | 137 | 3009 | 250.75 | 20.33; 17.19/23.91 | 36793; 248.60/250.70/255.24 | 15045/0/0 | 0/0; 12/12 |
| D_native | 0.500 | 6/12 | N/A | 6 | 0 | 12/12 | 6 | 0 | 0 | 140 | 2360 | 196.67 | 16.86; 16.77/22.36 | 34130; 243.79/243.72/251.95 | N/A/N/A/N/A | 0/0; 12/12 |
| M1_R1_A8_current_output | 0.500 | 6/12 | N/A | 6 | 0 | 12/12 | 6 | 0 | 0 | 149 | 2623 | 218.58 | 17.60; 17.43/21.65 | 36703; 246.33/247.00/253.27 | 1945/11170/0 | 0/0; 12/12 |
| M3_R2_A8_current_output | 0.500 | 6/12 | N/A | 6 | 0 | 12/12 | 6 | 0 | 0 | 159 | 2842 | 236.83 | 17.87; 19.26/21.48 | 39267; 246.96/244.79/255.16 | 2135/5160/6915 | 0/0; 12/12 |
| M3_R3_A8_current_output | 0.417 | 5/12 | N/A | 5 | 0 | 12/12 | 7 | 0 | 0 | 147 | 2514 | 209.50 | 17.10; 17.38/21.69 | 36170; 246.05/242.69/253.26 | 1875/2975/7720 | 0/0; 12/12 |
| T_scope | 0.500 | 6/12 | N/A | 6 | 0 | 12/12 | 6 | 0 | 0 | 128 | 2223 | 185.25 | 17.37; 15.85/22.21 | 31468; 245.84/248.16/252.11 | N/A/N/A/N/A | 0/0; 12/12 |

Absolute accepted-warm timing, separately by host:

| Arm | Host | Accepted warm requests | Whole-request wall mean (s) | CUDA first-encoder-end to finish span mean (s) |
|---|---|---:|---:|---:|
| B_A8_matched | 149.165.151.254 | 6 | 36.225 | 33.877 |
| B_A8_matched | 149.165.159.64 | 6 | 43.459 | 40.879 |
| D_matched | 149.165.151.254 | 6 | 30.443 | 28.050 |
| D_matched | 149.165.159.64 | 6 | 56.034 | 53.344 |
| D_native | 149.165.151.254 | 6 | 30.099 | 27.832 |
| D_native | 149.165.159.64 | 6 | 43.369 | 40.751 |
| M1_R1_A8_current_output | 149.165.151.254 | 6 | 32.751 | 30.486 |
| M1_R1_A8_current_output | 149.165.159.64 | 6 | 47.409 | 44.827 |
| M3_R2_A8_current_output | 149.165.151.254 | 6 | 39.036 | 36.771 |
| M3_R2_A8_current_output | 149.165.159.64 | 6 | 44.512 | 41.929 |
| M3_R3_A8_current_output | 149.165.151.254 | 6 | 32.447 | 30.184 |
| M3_R3_A8_current_output | 149.165.159.64 | 6 | 41.606 | 38.913 |
| T_scope | 149.165.151.254 | 6 | 22.824 | 20.485 |
| T_scope | 149.165.159.64 | 6 | 45.453 | 42.795 |

Same-question/seed/GPU paired request ratios and question-cluster exploratory intervals:

| Method/reference | Pairs | Geometric ratio | 95% exploratory cluster interval | Paired score delta; interval | .254 total ratio / calls ratio / amortized wall-call | .64 total ratio / calls ratio / amortized wall-call |
|---|---:|---:|---|---|---|---|
| B_A8_matched/D_native | 12 | 1.001 | 0.844..1.213 | -0.167; -0.333..0.000 | n=6: 1.204/1.320/0.911 | n=6: 1.002/1.085/0.924 |
| B_A8_matched/T_scope | 12 | 1.102 | 0.879..1.503 | -0.167; -0.333..0.000 | n=6: 1.587/1.782/0.890 | n=6: 0.956/1.011/0.946 |
| D_matched/D_native | 12 | 1.072 | 0.936..1.229 | -0.167; -0.333..0.000 | n=6: 1.011/1.028/0.984 | n=6: 1.292/1.436/0.900 |
| D_matched/T_scope | 12 | 1.181 | 1.045..1.348 | -0.167; -0.333..0.000 | n=6: 1.334/1.388/0.961 | n=6: 1.233/1.338/0.921 |
| D_matched/B_A8_matched | 12 | 1.072 | 0.857..1.328 | 0.000; 0.000..0.000 | n=6: 0.840/0.779/1.080 | n=6: 1.289/1.324/0.974 |
| M1_R1_A8_current_output/D_native | 12 | 1.025 | 0.850..1.242 | 0.000; 0.000..0.000 | n=6: 1.088/1.118/0.973 | n=6: 1.093/1.107/0.987 |
| M1_R1_A8_current_output/T_scope | 12 | 1.129 | 0.917..1.448 | 0.000; 0.000..0.000 | n=6: 1.435/1.509/0.951 | n=6: 1.043/1.032/1.011 |
| M1_R1_A8_current_output/B_A8_matched | 12 | 1.024 | 0.930..1.125 | 0.167; 0.000..0.333 | n=6: 0.904/0.847/1.068 | n=6: 1.091/1.021/1.069 |
| M3_R2_A8_current_output/D_native | 12 | 1.111 | 0.916..1.326 | 0.000; 0.000..0.000 | n=6: 1.297/1.385/0.936 | n=6: 1.026/1.087/0.944 |
| M3_R2_A8_current_output/T_scope | 12 | 1.224 | 0.987..1.566 | 0.000; 0.000..0.000 | n=6: 1.710/1.869/0.915 | n=6: 0.979/1.013/0.967 |
| M3_R2_A8_current_output/B_A8_matched | 12 | 1.111 | 0.948..1.345 | 0.167; 0.000..0.333 | n=6: 1.078/1.049/1.027 | n=6: 1.024/1.002/1.022 |
| M3_R3_A8_current_output/D_native | 12 | 1.031 | 0.868..1.253 | -0.083; -0.250..0.000 | n=6: 1.078/1.146/0.940 | n=6: 0.959/1.013/0.947 |
| M3_R3_A8_current_output/T_scope | 12 | 1.136 | 0.961..1.363 | -0.083; -0.250..0.000 | n=6: 1.422/1.547/0.919 | n=6: 0.915/0.944/0.970 |
| M3_R3_A8_current_output/B_A8_matched | 12 | 1.031 | 0.797..1.335 | 0.083; 0.000..0.250 | n=6: 0.896/0.868/1.032 | n=6: 0.957/0.934/1.025 |
| T_scope/D_native | 12 | 0.908 | 0.764..1.046 | 0.000; 0.000..0.000 | n=6: 0.758/0.741/1.024 | n=6: 1.048/1.073/0.977 |
| T_scope/B_A8_matched | 12 | 0.908 | 0.654..1.131 | 0.167; 0.000..0.333 | n=6: 0.630/0.561/1.123 | n=6: 1.046/0.989/1.057 |

Per-canvas call histogram, call positions 0/1/2+, terminations, output-token distribution and per-host timing values are preserved in the CSV and byte-exact redacted JSON. The intervals cluster by question with both seeds kept together; they are exploratory, not a noninferiority claim.

The three datasets are reported separately. No cross-dataset accuracy or cross-host absolute time is pooled; no policy is selected from this report.
