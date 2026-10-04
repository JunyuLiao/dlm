# Panel w: pre-registered held-out confirmation (RULER v32, seed 5353, 390 cells)

## vs dense FULL (reference)
common cells: 390

| arm | ruler32k_v32 | ruler64k_v32 | total | lost/gained | McNemar p | cwe | fwe | qa | mv | vt | sparse_kept | work |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense_default_fix | 164 | 159 | 323/390 | - | - | 19/30 | 30/30 | 41/60 | 24/30 | 1/30 | - | - |
| dense_PIECEWISE_fix | 164 | 159 | 323/390 | 0/0 | 1.000 | 19/30 | 30/30 | 41/60 | 24/30 | 1/30 | - | - |
| mage4096_PIECEWISE_fix | 154 | 153 | 307/390 | 20/4 | 0.002 | 3/30 | 29/30 | 42/60 | 25/30 | 1/30 | 0.084 | 0.407 |
| mage6144_PIECEWISE_fix | 157 | 151 | 308/390 | 18/3 | 0.001 | 5/30 | 30/30 | 42/60 | 24/30 | 1/30 | 0.126 | 0.433 |
| mage6144_PIECEWISE_fix_step1 | 154 | 150 | 304/390 | 22/3 | 0.000 | 4/30 | 30/30 | 40/60 | 22/30 | 0/30 | 0.126 | 0.598 |
| method_m2c_k12_mass_PIECEWISE_fix | 157 | 153 | 310/390 | 18/5 | 0.011 | 9/30 | 30/30 | 40/60 | 23/30 | 0/30 | 0.122 | 0.438 |
| method_m2c_r_PIECEWISE_fix | 150 | 149 | 299/390 | 26/2 | 0.000 | 1/30 | 30/30 | 40/60 | 20/30 | 0/30 | 0.053 | 0.389 |

## PRIMARY: vs MAGE k=6144 (arm 3); row method_m2c_k12_mass_PIECEWISE_fix is the registered test
common cells: 390

| arm | ruler32k_v32 | ruler64k_v32 | total | lost/gained | McNemar p | cwe | fwe | qa | mv | vt | sparse_kept | work |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mage6144_PIECEWISE_fix | 157 | 151 | 308/390 | - | - | 5/30 | 30/30 | 42/60 | 24/30 | 1/30 | 0.126 | 0.433 |
| dense_PIECEWISE_fix | 164 | 159 | 323/390 | 3/18 | 0.001 | 19/30 | 30/30 | 41/60 | 24/30 | 1/30 | - | - |
| dense_default_fix | 164 | 159 | 323/390 | 3/18 | 0.001 | 19/30 | 30/30 | 41/60 | 24/30 | 1/30 | - | - |
| mage4096_PIECEWISE_fix | 154 | 153 | 307/390 | 5/4 | 1.000 | 3/30 | 29/30 | 42/60 | 25/30 | 1/30 | 0.084 | 0.407 |
| mage6144_PIECEWISE_fix_step1 | 154 | 150 | 304/390 | 9/5 | 0.424 | 4/30 | 30/30 | 40/60 | 22/30 | 0/30 | 0.126 | 0.598 |
| method_m2c_k12_mass_PIECEWISE_fix | 157 | 153 | 310/390 | 10/12 | 0.832 | 9/30 | 30/30 | 40/60 | 23/30 | 0/30 | 0.122 | 0.438 |
| method_m2c_r_PIECEWISE_fix | 150 | 149 | 299/390 | 13/4 | 0.049 | 1/30 | 30/30 | 40/60 | 20/30 | 0/30 | 0.053 | 0.389 |

## secondary: vs MAGE k=4096 (row method_m2c_r_PIECEWISE_fix)
common cells: 390

| arm | ruler32k_v32 | ruler64k_v32 | total | lost/gained | McNemar p | cwe | fwe | qa | mv | vt | sparse_kept | work |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mage4096_PIECEWISE_fix | 154 | 153 | 307/390 | - | - | 3/30 | 29/30 | 42/60 | 25/30 | 1/30 | 0.084 | 0.407 |
| dense_PIECEWISE_fix | 164 | 159 | 323/390 | 4/20 | 0.002 | 19/30 | 30/30 | 41/60 | 24/30 | 1/30 | - | - |
| dense_default_fix | 164 | 159 | 323/390 | 4/20 | 0.002 | 19/30 | 30/30 | 41/60 | 24/30 | 1/30 | - | - |
| mage6144_PIECEWISE_fix | 157 | 151 | 308/390 | 4/5 | 1.000 | 5/30 | 30/30 | 42/60 | 24/30 | 1/30 | 0.126 | 0.433 |
| mage6144_PIECEWISE_fix_step1 | 154 | 150 | 304/390 | 11/8 | 0.648 | 4/30 | 30/30 | 40/60 | 22/30 | 0/30 | 0.126 | 0.598 |
| method_m2c_k12_mass_PIECEWISE_fix | 157 | 153 | 310/390 | 9/12 | 0.664 | 9/30 | 30/30 | 40/60 | 23/30 | 0/30 | 0.122 | 0.438 |
| method_m2c_r_PIECEWISE_fix | 150 | 149 | 299/390 | 13/5 | 0.096 | 1/30 | 30/30 | 40/60 | 20/30 | 0/30 | 0.053 | 0.389 |
