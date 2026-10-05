common cells: 585 (13 tasks per length asserted); reference dense_default_fix; per length = unweighted mean over tasks of the task means; overall = mean over lengths; comparable settings checked on every cell
coverage: dense_default_fix: planned 585 present 585 dropped 0 scored 585
coverage: mage2048_PIECEWISE_fix_lean: planned 585 present 585 dropped 0 scored 585
coverage: mage4096_PIECEWISE_fix_lean: planned 585 present 585 dropped 0 scored 585
coverage: mage4096_PIECEWISE_fix_plain: planned 585 present 585 dropped 0 scored 585

| arm | ruler32k_v33ofc official | ruler64k_v33ofc official | ruler128k_v33ofc official | official | cwe | w/o cwe | official diff vs ref [95% CI] | better/worse (p) | work |
|---|---|---|---|---|---|---|---|---|---|
| dense_default_fix | 94.0 | 88.7 | 84.4 | 89.0 | 84.0 | 89.5 | - | - | - |
| mage2048_PIECEWISE_fix_lean | 93.6 | 87.4 | 80.5 | 87.2 | 68.0 | 88.8 | -1.87 [-3.02, -0.73] | 15/50 (0.000) | - |
| mage4096_PIECEWISE_fix_lean | 93.7 | 87.8 | 82.5 | 88.0 | 77.6 | 88.9 | -1.06 [-2.12, -0.01] | 19/41 (0.006) | - |
| mage4096_PIECEWISE_fix_plain | 92.5 | 86.4 | 80.9 | 86.6 | 56.0 | 89.1 | -2.46 [-3.56, -1.38] | 16/53 (0.000) | - |
