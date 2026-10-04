mrcr: 72 common cells; reference dense_default_fix; comparable settings checked on every cell

coverage: dense_default_fix: planned 72 present 72 dropped 0 scored 72
coverage: mage4096_PIECEWISE_fix_lean: planned 72 present 72 dropped 0 scored 72
coverage: mage4096_PIECEWISE_fix_lean_rs4: planned 72 present 72 dropped 0 scored 72
coverage: mage4096_PIECEWISE_fix_lean_rs4_cgate: planned 72 present 72 dropped 0 scored 72
coverage: mage4096_PIECEWISE_fix_lean_rs4_conf: planned 72 present 72 dropped 0 scored 72
coverage: mage4096_PIECEWISE_fix_lean_rs8_cgate: planned 72 present 72 dropped 0 scored 72

| arm | subset | mean ratio | ref mean ratio | diff [95% CI] | sign test (better / worse cells) | cells | items |
|---|---|---|---|---|---|---|---|
| mage4096_PIECEWISE_fix_lean | mrcr2_32k_ofc | 0.3292 | 0.4537 | -0.1244 [-0.2716, +0.0182] | 2/11 (p=0.022) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean | mrcr2_64k_ofc | 0.1697 | 0.2385 | -0.0689 [-0.1525, -0.0016] | 7/10 (p=0.629) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean | mrcr2_128k_ofc | 0.1763 | 0.1789 | -0.0026 [-0.0837, +0.0580] | 11/6 (p=0.332) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs4 | mrcr2_32k_ofc | 0.4028 | 0.4537 | -0.0509 [-0.1734, +0.0614] | 2/10 (p=0.039) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs4 | mrcr2_64k_ofc | 0.2656 | 0.2385 | +0.0271 [-0.0295, +0.1019] | 9/6 (p=0.607) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs4 | mrcr2_128k_ofc | 0.1611 | 0.1789 | -0.0179 [-0.0887, +0.0373] | 9/7 (p=0.804) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs4_cgate | mrcr2_32k_ofc | 0.4038 | 0.4537 | -0.0498 [-0.1716, +0.0627] | 2/9 (p=0.065) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs4_cgate | mrcr2_64k_ofc | 0.2370 | 0.2385 | -0.0016 [-0.0744, +0.0801] | 6/8 (p=0.791) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs4_cgate | mrcr2_128k_ofc | 0.1480 | 0.1789 | -0.0309 [-0.1041, +0.0169] | 8/7 (p=1.000) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs4_conf | mrcr2_32k_ofc | 0.3937 | 0.4537 | -0.0600 [-0.1947, +0.0614] | 5/6 (p=1.000) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs4_conf | mrcr2_64k_ofc | 0.2543 | 0.2385 | +0.0158 [-0.0535, +0.0925] | 6/9 (p=0.607) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs4_conf | mrcr2_128k_ofc | 0.1463 | 0.1789 | -0.0326 [-0.1049, +0.0156] | 10/7 (p=0.629) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs8_cgate | mrcr2_32k_ofc | 0.3820 | 0.4537 | -0.0717 [-0.2067, +0.0582] | 2/8 (p=0.109) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs8_cgate | mrcr2_64k_ofc | 0.2459 | 0.2385 | +0.0074 [-0.0292, +0.0512] | 7/9 (p=0.804) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean_rs8_cgate | mrcr2_128k_ofc | 0.1556 | 0.1789 | -0.0233 [-0.1287, +0.0482] | 10/8 (p=0.815) | 24 | 24 |
