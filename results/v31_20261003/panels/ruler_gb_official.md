common cells: 174; reference dense_default_fix_ref

| arm | ruler32k official | ruler64k official | official | cwe | w/o cwe | strict correct | official diff vs ref [95% CI] | better/worse (p) | work |
|---|---|---|---|---|---|---|---|---|---|
| dense_default_fix_ref | 85.0 | 82.9 | 83.8 | 83.6 | 83.9 | 140/174 | - | - | - |
| mage0_PIECEWISE_fix_frac12_qblock_max_step1@gs | 85.8 | 83.7 | 84.6 | 88.6 | 84.3 | 136/174 | +0.83 [-0.55, +2.59] | 5/9 (0.424) | - |
| mage2048_PIECEWISE_fix_fast | 79.8 | 77.6 | 78.7 | 45.7 | 81.5 | 128/174 | -5.20 [-8.36, -2.33] | 4/17 (0.007) | - |
| mage2048_PIECEWISE_fix_qblock_max_step1@gb | 84.0 | 81.4 | 82.7 | 72.1 | 83.5 | 131/174 | -1.21 [-3.28, +0.80] | 4/13 (0.049) | - |
| mage4096_PIECEWISE_fix_kvhead_step1@gb | 84.3 | 82.1 | 83.1 | 78.6 | 83.5 | 133/174 | -0.69 [-2.76, +1.44] | 4/11 (0.118) | - |
| mage4096_PIECEWISE_fix_plain | 82.0 | 79.2 | 80.6 | 62.9 | 82.1 | 129/174 | -3.25 [-6.06, -0.75] | 4/15 (0.019) | - |
| mage4096_PIECEWISE_fix_qblock_max_step0@gb | 83.0 | 81.9 | 82.5 | 79.3 | 82.7 | 132/174 | -1.35 [-3.62, +0.75] | 4/13 (0.049) | - |
| mage4096_PIECEWISE_fix_qblock_max_step1@gb | 85.9 | 82.3 | 84.0 | 80.0 | 84.3 | 137/174 | +0.14 [-1.47, +1.98] | 5/8 (0.581) | - |
| mage6144_PIECEWISE_fix_fast | 82.6 | 80.4 | 81.6 | 75.0 | 82.1 | 130/174 | -2.27 [-4.97, +0.17] | 4/14 (0.031) | - |
| mage8192_PIECEWISE_fix_fast | 83.5 | 82.7 | 83.1 | 83.6 | 83.0 | 135/174 | -0.72 [-3.13, +1.61] | 5/9 (0.424) | - |
| method_m2c_k12_mass_PIECEWISE_fix_fast | 85.7 | 83.7 | 84.6 | 87.9 | 84.3 | 138/174 | +0.78 [-0.60, +2.61] | 5/7 (0.774) | - |
| method_m2c_r_PIECEWISE_fix_plain | 85.1 | 81.5 | 83.2 | 70.0 | 84.3 | 135/174 | -0.66 [-2.44, +1.12] | 5/10 (0.302) | - |
