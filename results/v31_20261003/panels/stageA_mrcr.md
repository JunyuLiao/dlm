mrcr: 72 common cells; reference dense_default_fix; comparable settings checked on every cell

coverage: dense_default_fix: planned 72 present 72 dropped 0 scored 72
coverage: mage2048_PIECEWISE_fix_lean: planned 72 present 72 dropped 0 scored 72
coverage: mage4096_PIECEWISE_fix_lean: planned 72 present 72 dropped 0 scored 72
coverage: mage4096_PIECEWISE_fix_plain: planned 72 present 72 dropped 0 scored 72

| arm | subset | mean ratio | ref mean ratio | diff [95% CI] | sign test (better / worse cells) | cells | items |
|---|---|---|---|---|---|---|---|
| mage2048_PIECEWISE_fix_lean | mrcr2_32k_ofc | 0.2848 | 0.4537 | -0.1689 [-0.2987, -0.0569] | 2/12 (p=0.013) | 24 | 24 |
| mage2048_PIECEWISE_fix_lean | mrcr2_64k_ofc | 0.1897 | 0.2385 | -0.0489 [-0.1411, +0.0292] | 7/9 (p=0.804) | 24 | 24 |
| mage2048_PIECEWISE_fix_lean | mrcr2_128k_ofc | 0.1661 | 0.1789 | -0.0128 [-0.1004, +0.0411] | 11/7 (p=0.481) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean | mrcr2_32k_ofc | 0.3292 | 0.4537 | -0.1244 [-0.2716, +0.0182] | 2/11 (p=0.022) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean | mrcr2_64k_ofc | 0.1697 | 0.2385 | -0.0689 [-0.1525, -0.0016] | 7/10 (p=0.629) | 24 | 24 |
| mage4096_PIECEWISE_fix_lean | mrcr2_128k_ofc | 0.1763 | 0.1789 | -0.0026 [-0.0837, +0.0580] | 11/6 (p=0.332) | 24 | 24 |
| mage4096_PIECEWISE_fix_plain | mrcr2_32k_ofc | 0.3031 | 0.4537 | -0.1506 [-0.2603, -0.0549] | 5/13 (p=0.096) | 24 | 24 |
| mage4096_PIECEWISE_fix_plain | mrcr2_64k_ofc | 0.1455 | 0.2385 | -0.0930 [-0.2205, +0.0114] | 8/14 (p=0.286) | 24 | 24 |
| mage4096_PIECEWISE_fix_plain | mrcr2_128k_ofc | 0.1646 | 0.1789 | -0.0144 [-0.0640, +0.0339] | 8/15 (p=0.210) | 24 | 24 |
