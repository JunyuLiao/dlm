longbench: 503 common cells; reference dense_default_fix; comparable settings checked on every cell

coverage: dense_default_fix: planned 503 present 503 dropped 0 scored 503
coverage: mage2048_PIECEWISE_fix_lean: planned 503 present 503 dropped 0 scored 503
coverage: mage4096_PIECEWISE_fix_lean: planned 503 present 503 dropped 0 scored 503
coverage: mage4096_PIECEWISE_fix_plain: planned 503 present 503 dropped 0 scored 503

| arm | subset | accuracy | ref accuracy | diff [95% CI] | McNemar (arm only / ref only correct) | cells | items |
|---|---|---|---|---|---|---|---|
| mage2048_PIECEWISE_fix_lean | longbench_v2_0shot | 11.9 | 12.1 | -0.20 [-1.99, +1.59] | 10/11 (p=1.000) | 503 | 503 |
| mage4096_PIECEWISE_fix_lean | longbench_v2_0shot | 12.9 | 12.1 | +0.80 [-0.99, +2.58] | 12/8 (p=0.503) | 503 | 503 |
| mage4096_PIECEWISE_fix_plain | longbench_v2_0shot | 13.1 | 12.1 | +0.99 [-0.80, +2.78] | 13/8 (p=0.383) | 503 | 503 |

result.py columns (mean over runs of unrounded values; one decimal); diff = arm - ref

| arm | dataset | Overall | Easy | Hard | Short | Medium | Long |
|---|---|---|---|---|---|---|---|
| dense_default_fix | longbench_v2_0shot | 12.1 | 12.5 | 11.9 | 14.4 | 13.0 | 6.5 |
| mage2048_PIECEWISE_fix_lean | longbench_v2_0shot | 11.9 (-0.2) | 13.0 (+0.5) | 11.3 (-0.6) | 17.2 (+2.8) | 11.2 (-1.9) | 4.6 (-1.9) |
| mage4096_PIECEWISE_fix_lean | longbench_v2_0shot | 12.9 (+0.8) | 14.6 (+2.1) | 11.9 (+0.0) | 17.8 (+3.3) | 12.6 (-0.5) | 5.6 (-0.9) |
| mage4096_PIECEWISE_fix_plain | longbench_v2_0shot | 13.1 (+1.0) | 14.6 (+2.1) | 12.2 (+0.3) | 16.7 (+2.2) | 14.9 (+1.9) | 3.7 (-2.8) |
