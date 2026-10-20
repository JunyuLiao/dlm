All164 HumanEval problems, seed42. Accuracy is correct/164; N is denoising forwards excluding commit calls.
W is clean mean request time including prefill and all selector/correction work.
Sparsity is measured in the trajectory-matched audit pass; LOCAL is dense.

**K256, thinking on**

| Method | Correct/164 | N | C | N/C | Output tokens | S ms | W ms | S/N ms | GLOBAL % | Overall % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense FULL | 161 | 48.079 | 5.195 | 9.255 | 1202.51 | 929.363 | 952.487 | 19.330 | 0.000 | 0.000 |
| Dense PIECEWISE | 161 | 48.079 | 5.195 | 9.255 | 1202.51 | 940.269 | 959.672 | 19.557 | 0.000 | 0.000 |
| All-kept FA4 | 164 | 51.945 | 5.482 | 9.476 | 1266.24 | 1091.848 | 1111.270 | 21.019 | 0.000 | 0.000 |
| V31 | 159 | 49.256 | 5.122 | 9.617 | 1189.19 | 1104.579 | 1123.988 | 22.425 | 48.325 | 10.922 |
| V2 | 162 | 50.043 | 5.268 | 9.499 | 1223.60 | 1128.215 | 1147.622 | 22.545 | 25.618 | 5.824 |
| V3a | 159 | 51.854 | 5.323 | 9.741 | 1233.70 | 1186.225 | 1205.628 | 22.876 | 50.587 | 11.857 |
| Full-value swaps | 155 | 49.085 | 5.183 | 9.471 | 1198.24 | 1805.949 | 1825.361 | 36.792 | 48.880 | 11.202 |
| Direction-guarded residual | 158 | 52.543 | 5.293 | 9.927 | 1228.07 | 1301.915 | 1321.328 | 24.778 | 50.979 | 11.921 |

V2 tau=0.0944851926739; first-canvas matching: {'absolute_kept_ratio': 0.9939731591937474, 'kept_fraction_ratio': 1.0017258234074917, 'matched': True, 'sparsity_delta_pp': -0.17089518140988025}.

**K256, thinking off**

| Method | Correct/164 | N | C | N/C | Output tokens | S ms | W ms | S/N ms | GLOBAL % | Overall % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense FULL | 156 | 9.470 | 1.341 | 7.059 | 228.49 | 183.129 | 205.666 | 19.339 | 0.000 | 0.000 |
| Dense PIECEWISE | 156 | 9.470 | 1.341 | 7.059 | 228.49 | 184.888 | 203.842 | 19.525 | 0.000 | 0.000 |
| All-kept FA4 | 156 | 9.957 | 1.341 | 7.423 | 236.62 | 209.045 | 228.095 | 20.994 | 0.000 | 0.000 |
| V31 | 157 | 10.774 | 1.372 | 7.853 | 244.56 | 242.932 | 262.178 | 22.547 | 20.031 | 3.579 |
| V2 | 158 | 10.854 | 1.409 | 7.706 | 247.77 | 244.821 | 264.080 | 22.557 | 5.168 | 0.889 |
| V3a | 157 | 10.305 | 1.335 | 7.717 | 234.32 | 240.810 | 260.063 | 23.369 | 9.504 | 1.587 |
| Full-value swaps | 156 | 10.445 | 1.341 | 7.786 | 235.07 | 294.975 | 314.227 | 28.240 | 14.663 | 2.515 |
| Direction-guarded residual | 156 | 10.280 | 1.341 | 7.664 | 235.20 | 245.081 | 264.348 | 23.839 | 10.074 | 1.685 |

V2 tau=0.0981299396515; first-canvas matching: {'absolute_kept_ratio': 0.9813049853372434, 'kept_fraction_ratio': 1.0009674971028133, 'matched': True, 'sparsity_delta_pp': -0.09601062555383288}.

**K128, thinking on**

| Method | Correct/164 | N | C | N/C | Output tokens | S ms | W ms | S/N ms | GLOBAL % | Overall % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense FULL | 161 | 48.079 | 5.195 | 9.255 | 1202.51 | 929.363 | 952.487 | 19.330 | 0.000 | 0.000 |
| Dense PIECEWISE | 161 | 48.079 | 5.195 | 9.255 | 1202.51 | 940.269 | 959.672 | 19.557 | 0.000 | 0.000 |
| All-kept FA4 | 164 | 51.945 | 5.482 | 9.476 | 1266.24 | 1091.848 | 1111.270 | 21.019 | 0.000 | 0.000 |
| V31 | 157 | 57.415 | 5.360 | 10.712 | 1243.13 | 1278.469 | 1297.955 | 22.267 | 59.870 | 14.630 |
| V2 | 158 | 53.323 | 5.476 | 9.738 | 1277.45 | 1204.595 | 1224.068 | 22.590 | 56.641 | 14.744 |
| V3a | 154 | 58.445 | 5.335 | 10.954 | 1245.00 | 1324.915 | 1344.402 | 22.669 | 61.818 | 15.724 |
| Full-value swaps | 155 | 55.982 | 5.341 | 10.481 | 1233.15 | 1999.532 | 2019.004 | 35.718 | 59.735 | 14.802 |
| Direction-guarded residual | 155 | 56.518 | 5.341 | 10.581 | 1237.51 | 1402.708 | 1422.193 | 24.819 | 60.891 | 15.400 |

V2 tau=0.560071620607; first-canvas matching: {'absolute_kept_ratio': 1.0027716967227327, 'kept_fraction_ratio': 1.0002544141412766, 'matched': True, 'sparsity_delta_pp': -0.023822152179597822}.

**K128, thinking off**

| Method | Correct/164 | N | C | N/C | Output tokens | S ms | W ms | S/N ms | GLOBAL % | Overall % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense FULL | 156 | 9.470 | 1.341 | 7.059 | 228.49 | 183.129 | 205.666 | 19.339 | 0.000 | 0.000 |
| Dense PIECEWISE | 156 | 9.470 | 1.341 | 7.059 | 228.49 | 184.888 | 203.842 | 19.525 | 0.000 | 0.000 |
| All-kept FA4 | 156 | 9.957 | 1.341 | 7.423 | 236.62 | 209.045 | 228.095 | 20.994 | 0.000 | 0.000 |
| V31 | 156 | 11.488 | 1.384 | 8.300 | 243.12 | 255.146 | 274.204 | 22.210 | 29.656 | 5.477 |
| V2 | 160 | 10.439 | 1.354 | 7.712 | 233.58 | 233.637 | 252.697 | 22.381 | 18.355 | 3.096 |
| V3a | 158 | 10.165 | 1.329 | 7.647 | 231.34 | 234.295 | 253.350 | 23.050 | 14.152 | 2.362 |
| Full-value swaps | 157 | 10.280 | 1.317 | 7.806 | 229.80 | 320.709 | 339.769 | 31.196 | 15.792 | 2.650 |
| Direction-guarded residual | 158 | 11.829 | 1.402 | 8.435 | 249.40 | 283.234 | 302.297 | 23.943 | 32.730 | 6.100 |

V2 tau=0.715741013753; first-canvas matching: {'absolute_kept_ratio': 0.9368037430300155, 'kept_fraction_ratio': 0.9943726881324186, 'matched': False, 'sparsity_delta_pp': 0.531160265754127}.

**Reproducibility:** all22 full audit/clean comparisons per budget pass; independent engine prefixes and official rescores pass.
**Work limitation:** K128 thinking-disabled V2 misses the first-canvas0.25pp target: +0.53116pp sparsity and0.93680 absolute kept-tile ratio. The predeclared nearest-work rule retained its threshold. Whole-trajectory work is not matched in any panel by first-canvas calibration alone.
Dense FULL/PIECEWISE/all-kept references are shared between budgets and are not additional independent trials.
V2/V3a use Gaussian32 on all five GLOBAL layers; full-value swaps/residual affect only layer29. All LOCAL attention stays dense.
Overall sparsity counts denoising GLOBAL plus native-window LOCAL Q128/KV64 rectangles, excluding prefill/commits and additional selector/PV work. W includes the complete method cost; S/N is amortized.
Main accuracy uses seed42. Seed43 covers six examined questions and remains descriptive. Question-bootstrap intervals condition on the observed engines and do not establish seed robustness.
