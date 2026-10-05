common cells: 156 (13 tasks per length asserted); reference dense_default_fix; per length = unweighted mean over tasks of the task means; overall = mean over lengths; comparable settings checked on every cell
coverage: dense_default_fix: planned 156 present 156 dropped 0 scored 156
coverage: mage4096_PIECEWISE_fix_lean: planned 156 present 156 dropped 0 scored 156
coverage: mage4096_PIECEWISE_fix_lean_rs4: planned 156 present 156 dropped 0 scored 156
coverage: mage4096_PIECEWISE_fix_lean_rs4_cgate: planned 156 present 156 dropped 0 scored 156
coverage: mage4096_PIECEWISE_fix_lean_rs4_conf: planned 156 present 156 dropped 0 scored 156
coverage: mage4096_PIECEWISE_fix_lean_rs8_cgate: planned 156 present 156 dropped 0 scored 156

| arm | ruler32k_v34ofc official | ruler128k_v34ofc official | official | cwe | w/o cwe | official diff vs ref [95% CI] | better/worse (p) | work |
|---|---|---|---|---|---|---|---|---|
| dense_default_fix | 91.8 | 82.1 | 86.9 | 79.2 | 87.6 | - | - | - |
| mage4096_PIECEWISE_fix_lean | 90.4 | 83.9 | 87.1 | 70.8 | 88.5 | +0.22 [-1.89, +2.28] | 11/9 (0.824) | - |
| mage4096_PIECEWISE_fix_lean_rs4 | 90.4 | 84.7 | 87.5 | 74.2 | 88.6 | +0.61 [-1.57, +2.72] | 12/9 (0.664) | - |
| mage4096_PIECEWISE_fix_lean_rs4_cgate | 90.4 | 84.6 | 87.5 | 73.3 | 88.6 | +0.54 [-1.60, +2.63] | 12/10 (0.832) | - |
| mage4096_PIECEWISE_fix_lean_rs4_conf | 90.4 | 84.2 | 87.3 | 72.5 | 88.5 | +0.35 [-1.79, +2.47] | 11/10 (1.000) | - |
| mage4096_PIECEWISE_fix_lean_rs8_cgate | 90.4 | 83.7 | 87.0 | 70.8 | 88.4 | +0.10 [-2.02, +2.18] | 11/10 (1.000) | - |
