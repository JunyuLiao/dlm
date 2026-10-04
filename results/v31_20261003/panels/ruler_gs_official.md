common cells: 174; reference dense_default_fix_ref

| arm | ruler32k official | ruler64k official | official | cwe | w/o cwe | strict correct | official diff vs ref [95% CI] | better/worse (p) | work |
|---|---|---|---|---|---|---|---|---|---|
| dense_default_fix_ref | 85.0 | 82.9 | 83.8 | 83.6 | 83.9 | 140/174 | - | - | - |
| mage0_PIECEWISE_fix_frac12_kvblock_max_step1@gs | 85.5 | 83.4 | 84.3 | 84.3 | 84.3 | 136/174 | +0.49 [-1.06, +2.36] | 5/9 (0.424) | - |
| mage0_PIECEWISE_fix_frac12_kvblock_max_step1_carry@gs | 85.5 | 83.4 | 84.3 | 84.3 | 84.3 | 136/174 | +0.49 [-1.03, +2.27] | 5/9 (0.424) | - |
| mage0_PIECEWISE_fix_frac12_kvhead_max_step1@gs | 85.3 | 83.5 | 84.3 | 83.6 | 84.3 | 136/174 | +0.43 [-0.80, +1.84] | 5/9 (0.424) | - |
| mage0_PIECEWISE_fix_frac12_kvhead_step0 | 82.9 | 80.4 | 81.7 | 78.6 | 81.9 | 131/174 | -2.13 [-4.89, +0.32] | 4/13 (0.049) | - |
| mage0_PIECEWISE_fix_frac12_kvhead_step1 | 84.1 | 83.1 | 83.6 | 82.1 | 83.7 | 133/174 | -0.26 [-2.18, +1.67] | 5/12 (0.143) | - |
| mage0_PIECEWISE_fix_frac12_kvhead_step1_carry@gs | 84.1 | 83.1 | 83.6 | 82.1 | 83.7 | 133/174 | -0.26 [-2.16, +1.75] | 5/12 (0.143) | - |
| mage0_PIECEWISE_fix_frac12_qblock_max_step1 | 85.8 | 83.7 | 84.6 | 88.6 | 84.3 | 136/174 | +0.83 [-0.52, +2.61] | 5/9 (0.424) | - |
| mage0_PIECEWISE_fix_frac12_qblock_max_step1@gs | 85.8 | 83.7 | 84.6 | 88.6 | 84.3 | 136/174 | +0.83 [-0.52, +2.64] | 5/9 (0.424) | - |
| mage0_PIECEWISE_fix_frac12_qblock_max_step1_carry@gs | 85.8 | 83.7 | 84.6 | 88.6 | 84.3 | 136/174 | +0.83 [-0.52, +2.64] | 5/9 (0.424) | - |
| mage0_PIECEWISE_fix_frac12_qblock_step1 | 85.7 | 83.4 | 84.5 | 87.9 | 84.2 | 136/174 | +0.63 [-0.63, +2.41] | 5/8 (0.581) | - |
| mage0_PIECEWISE_fix_frac12_qhead_step1 | 84.5 | 83.6 | 84.0 | 87.9 | 83.7 | 135/174 | +0.20 [-1.72, +2.13] | 6/9 (0.607) | - |
| mage4096_PIECEWISE_fix_plain | 82.0 | 79.2 | 80.6 | 62.9 | 82.1 | 129/174 | -3.25 [-6.24, -0.75] | 4/15 (0.019) | - |
| mage6144_PIECEWISE_fix_fast | 82.6 | 80.4 | 81.6 | 75.0 | 82.1 | 130/174 | -2.27 [-4.91, +0.26] | 4/14 (0.031) | - |
| method_m2c_k12_mass_PIECEWISE_fix_fast | 85.7 | 83.7 | 84.6 | 87.9 | 84.3 | 138/174 | +0.78 [-0.60, +2.50] | 5/7 (0.774) | - |
