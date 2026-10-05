Trajectory stability vs `dense_default_fix` (null: `dense_PIECEWISE_fix`). Per-cell log N ratio, item-clustered bootstrap.

### longbench_v2_32k

| arm | cells | sd [95% CI] | excess | median | p90 | P(>1.5x) |
|---|---:|---|---|---|---|---|
| dense_PIECEWISE_fix | 48 | 0.332 [0.224, 0.415] | 0.000 | 1.000 | 1.520 | 0.10 |
| mage1024_PIECEWISE_fix_fast2 | 48 | 0.681 [0.414, 0.944] | 0.594 | 1.217 | 3.281 | 0.40 |
| mage4096_PIECEWISE_fix_fast2 | 48 | 0.537 [0.401, 0.641] | 0.422 | 1.170 | 2.037 | 0.21 |
| method_base_PIECEWISE_fix | 48 | 0.456 [0.325, 0.584] | 0.313 | 1.183 | 1.870 | 0.31 |
| method_base_cgate_PIECEWISE_fix | 48 | 0.410 [0.311, 0.489] | 0.242 | 1.040 | 1.809 | 0.15 |
| method_base_cgate_PIECEWISE_fix_fast | 48 | 0.395 [0.288, 0.492] | 0.215 | 0.983 | 1.530 | 0.12 |
| method_m2c_PIECEWISE_fix_fast2 | 48 | 0.405 [0.305, 0.485] | 0.232 | 1.119 | 1.682 | 0.21 |
| method_m2c_cgate_PIECEWISE_fix_fast | 48 | 0.352 [0.265, 0.414] | 0.119 | 1.000 | 1.633 | 0.12 |
| method_m2c_cgate_r_PIECEWISE_fix_fast | 48 | 0.354 [0.267, 0.416] | 0.124 | 0.992 | 1.660 | 0.12 |
| method_m2c_k12_PIECEWISE_fix_fast | 48 | 0.573 [0.408, 0.712] | 0.467 | 1.286 | 2.436 | 0.38 |
| method_m2c_k12_cgate_PIECEWISE_fix_fast | 48 | 0.570 [0.356, 0.795] | 0.464 | 1.044 | 2.174 | 0.21 |
| method_m2c_k12_mass_PIECEWISE_fix_fast | 48 | 0.484 [0.357, 0.588] | 0.352 | 1.049 | 2.008 | 0.17 |
| method_m2c_k12_mass_cgate_PIECEWISE_fix_fast | 48 | 0.529 [0.332, 0.730] | 0.412 | 0.975 | 1.694 | 0.10 |
| method_m2c_k20_PIECEWISE_fix_fast | 40 | 0.376 [0.286, 0.461] | 0.178 | 0.948 | 1.471 | 0.05 |
| method_m2c_k30_PIECEWISE_fix_fast | 23 | 0.540 [0.330, 0.726] | 0.426 | 1.015 | 2.487 | 0.17 |
| method_m2c_k5_PIECEWISE_fix_fast | 48 | 0.529 [0.421, 0.617] | 0.412 | 1.440 | 2.595 | 0.48 |
| method_m2c_k5_cgate_PIECEWISE_fix_fast | 48 | 0.629 [0.425, 0.839] | 0.534 | 1.342 | 2.528 | 0.42 |
| method_m2c_k5_mass_PIECEWISE_fix_fast | 48 | 0.510 [0.396, 0.601] | 0.388 | 1.126 | 1.864 | 0.25 |
| method_m2c_k5_mass_cgate_PIECEWISE_fix_fast | 48 | 0.405 [0.310, 0.482] | 0.233 | 1.165 | 1.711 | 0.17 |
| method_m2c_r_PIECEWISE_fix_fast | 48 | 0.402 [0.303, 0.483] | 0.227 | 1.119 | 1.682 | 0.21 |
| method_main_PIECEWISE_fix | 48 | 0.501 [0.345, 0.641] | 0.376 | 1.002 | 1.803 | 0.17 |
| method_main_PIECEWISE_fix_fast2 | 48 | 0.428 [0.326, 0.516] | 0.271 | 0.999 | 1.629 | 0.17 |
| method_main_cgate_PIECEWISE_fix_fast2 | 48 | 0.384 [0.270, 0.481] | 0.193 | 0.963 | 1.519 | 0.10 |
| method_p1_PIECEWISE_fix | 48 | 0.635 [0.487, 0.759] | 0.541 | 1.722 | 3.207 | 0.58 |
| method_p1_cgate_PIECEWISE_fix | 48 | 0.533 [0.377, 0.671] | 0.418 | 1.400 | 2.240 | 0.48 |

### longbench_v2_64k

| arm | cells | sd [95% CI] | excess | median | p90 | P(>1.5x) |
|---|---:|---|---|---|---|---|
| dense_PIECEWISE_fix | 48 | 0.307 [0.212, 0.378] | 0.000 | 1.000 | 1.705 | 0.10 |
| mage1024_PIECEWISE_fix_fast2 | 48 | 0.657 [0.435, 0.836] | 0.581 | 1.015 | 3.164 | 0.21 |
| mage4096_PIECEWISE_fix_fast2 | 48 | 0.546 [0.287, 0.810] | 0.452 | 1.043 | 1.598 | 0.12 |
| method_base_PIECEWISE_fix | 48 | 0.568 [0.449, 0.671] | 0.478 | 1.253 | 2.618 | 0.38 |
| method_base_cgate_PIECEWISE_fix | 48 | 0.551 [0.404, 0.669] | 0.458 | 1.186 | 3.041 | 0.23 |
| method_base_cgate_PIECEWISE_fix_fast | 48 | 0.401 [0.279, 0.506] | 0.259 | 1.025 | 1.803 | 0.15 |
| method_m2c_PIECEWISE_fix_fast2 | 48 | 0.464 [0.347, 0.569] | 0.348 | 0.964 | 2.165 | 0.12 |
| method_m2c_cgate_PIECEWISE_fix_fast | 48 | 0.375 [0.255, 0.487] | 0.216 | 1.022 | 1.611 | 0.12 |
| method_m2c_cgate_r_PIECEWISE_fix_fast | 48 | 0.347 [0.257, 0.425] | 0.163 | 1.034 | 1.611 | 0.12 |
| method_m2c_k12_PIECEWISE_fix_fast | 48 | 0.412 [0.302, 0.508] | 0.275 | 1.087 | 1.577 | 0.17 |
| method_m2c_k12_cgate_PIECEWISE_fix_fast | 48 | 0.375 [0.296, 0.438] | 0.216 | 1.009 | 1.600 | 0.17 |
| method_m2c_k12_mass_PIECEWISE_fix_fast | 48 | 0.358 [0.277, 0.430] | 0.184 | 1.093 | 1.553 | 0.19 |
| method_m2c_k12_mass_cgate_PIECEWISE_fix_fast | 48 | 0.373 [0.270, 0.471] | 0.212 | 1.093 | 1.626 | 0.21 |
| method_m2c_k20_PIECEWISE_fix_fast | 32 | 0.421 [0.287, 0.530] | 0.288 | 0.995 | 1.939 | 0.25 |
| method_m2c_k30_PIECEWISE_fix_fast | 16 | 0.398 [0.178, 0.551] | 0.253 | 1.014 | 2.061 | 0.19 |
| method_m2c_k5_PIECEWISE_fix_fast | 48 | 0.515 [0.362, 0.661] | 0.414 | 1.079 | 2.336 | 0.29 |
| method_m2c_k5_cgate_PIECEWISE_fix_fast | 48 | 0.523 [0.406, 0.621] | 0.423 | 1.024 | 2.070 | 0.19 |
| method_m2c_k5_mass_PIECEWISE_fix_fast | 48 | 0.430 [0.296, 0.536] | 0.301 | 1.102 | 1.754 | 0.21 |
| method_m2c_k5_mass_cgate_PIECEWISE_fix_fast | 48 | 0.386 [0.265, 0.507] | 0.233 | 1.141 | 1.757 | 0.15 |
| method_m2c_r_PIECEWISE_fix_fast | 48 | 0.464 [0.346, 0.569] | 0.348 | 0.964 | 2.165 | 0.12 |
| method_main_PIECEWISE_fix | 48 | 0.410 [0.321, 0.490] | 0.273 | 1.039 | 1.820 | 0.19 |
| method_main_PIECEWISE_fix_fast2 | 48 | 0.399 [0.299, 0.496] | 0.255 | 1.036 | 1.732 | 0.17 |
| method_main_cgate_PIECEWISE_fix_fast2 | 48 | 0.394 [0.287, 0.487] | 0.248 | 1.089 | 1.582 | 0.15 |
| method_p1_PIECEWISE_fix | 48 | 0.604 [0.480, 0.707] | 0.521 | 1.609 | 3.600 | 0.56 |
| method_p1_cgate_PIECEWISE_fix | 48 | 0.462 [0.364, 0.551] | 0.346 | 1.433 | 2.355 | 0.42 |
