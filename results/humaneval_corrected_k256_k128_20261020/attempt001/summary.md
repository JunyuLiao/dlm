All164 HumanEval problems, seed42. Accuracy is correct/164; N is denoising forwards excluding commit calls.
W is clean mean request time including prefill and all selector/correction work.
Sparsity is measured in the trajectory-matched audit pass; LOCAL is dense.

**K256, thinking on**

| Method | Accuracy | N/problem | N/C | W ms | GLOBAL / overall sparsity |
|---|---:|---:|---:|---:|---:|
| Dense FULL | 98.17% (161/164) | 48.08 | 9.25 | 952.5 | 0.00% / 0.00% |
| Dense PIECEWISE | 98.17% (161/164) | 48.08 | 9.25 | 959.7 | 0.00% / 0.00% |
| All-kept FA4 | 100.00% (164/164) | 51.95 | 9.48 | 1111.3 | 0.00% / 0.00% |
| V31 | 96.95% (159/164) | 49.26 | 9.62 | 1124.0 | 48.32% / 10.92% |
| V2 | 98.78% (162/164) | 50.04 | 9.50 | 1147.6 | 25.62% / 5.82% |
| V3a | 96.95% (159/164) | 51.85 | 9.74 | 1205.6 | 50.59% / 11.86% |
| Full-value swaps | 94.51% (155/164) | 49.09 | 9.47 | 1825.4 | 48.88% / 11.20% |
| Direction-guarded residual | 96.34% (158/164) | 52.54 | 9.93 | 1321.3 | 50.98% / 11.92% |

V2 tau=0.0944851926739; first-canvas matching: {'absolute_kept_ratio': 0.9939731591937474, 'kept_fraction_ratio': 1.0017258234074917, 'matched': True, 'sparsity_delta_pp': -0.17089518140988025}.

**K256, thinking off**

| Method | Accuracy | N/problem | N/C | W ms | GLOBAL / overall sparsity |
|---|---:|---:|---:|---:|---:|
| Dense FULL | 95.12% (156/164) | 9.47 | 7.06 | 205.7 | 0.00% / 0.00% |
| Dense PIECEWISE | 95.12% (156/164) | 9.47 | 7.06 | 203.8 | 0.00% / 0.00% |
| All-kept FA4 | 95.12% (156/164) | 9.96 | 7.42 | 228.1 | 0.00% / 0.00% |
| V31 | 95.73% (157/164) | 10.77 | 7.85 | 262.2 | 20.03% / 3.58% |
| V2 | 96.34% (158/164) | 10.85 | 7.71 | 264.1 | 5.17% / 0.89% |
| V3a | 95.73% (157/164) | 10.30 | 7.72 | 260.1 | 9.50% / 1.59% |
| Full-value swaps | 95.12% (156/164) | 10.45 | 7.79 | 314.2 | 14.66% / 2.52% |
| Direction-guarded residual | 95.12% (156/164) | 10.28 | 7.66 | 264.3 | 10.07% / 1.68% |

V2 tau=0.0981299396515; first-canvas matching: {'absolute_kept_ratio': 0.9813049853372434, 'kept_fraction_ratio': 1.0009674971028133, 'matched': True, 'sparsity_delta_pp': -0.09601062555383288}.

**K128, thinking on**

| Method | Accuracy | N/problem | N/C | W ms | GLOBAL / overall sparsity |
|---|---:|---:|---:|---:|---:|
| Dense FULL | 98.17% (161/164) | 48.08 | 9.25 | 952.5 | 0.00% / 0.00% |
| Dense PIECEWISE | 98.17% (161/164) | 48.08 | 9.25 | 959.7 | 0.00% / 0.00% |
| All-kept FA4 | 100.00% (164/164) | 51.95 | 9.48 | 1111.3 | 0.00% / 0.00% |
| V31 | 95.73% (157/164) | 57.41 | 10.71 | 1298.0 | 59.87% / 14.63% |
| V2 | 96.34% (158/164) | 53.32 | 9.74 | 1224.1 | 56.64% / 14.74% |
| V3a | 93.90% (154/164) | 58.45 | 10.95 | 1344.4 | 61.82% / 15.72% |
| Full-value swaps | 94.51% (155/164) | 55.98 | 10.48 | 2019.0 | 59.73% / 14.80% |
| Direction-guarded residual | 94.51% (155/164) | 56.52 | 10.58 | 1422.2 | 60.89% / 15.40% |

V2 tau=0.560071620607; first-canvas matching: {'absolute_kept_ratio': 1.0027716967227327, 'kept_fraction_ratio': 1.0002544141412766, 'matched': True, 'sparsity_delta_pp': -0.023822152179597822}.

**K128, thinking off**

| Method | Accuracy | N/problem | N/C | W ms | GLOBAL / overall sparsity |
|---|---:|---:|---:|---:|---:|
| Dense FULL | 95.12% (156/164) | 9.47 | 7.06 | 205.7 | 0.00% / 0.00% |
| Dense PIECEWISE | 95.12% (156/164) | 9.47 | 7.06 | 203.8 | 0.00% / 0.00% |
| All-kept FA4 | 95.12% (156/164) | 9.96 | 7.42 | 228.1 | 0.00% / 0.00% |
| V31 | 95.12% (156/164) | 11.49 | 8.30 | 274.2 | 29.66% / 5.48% |
| V2 | 97.56% (160/164) | 10.44 | 7.71 | 252.7 | 18.35% / 3.10% |
| V3a | 96.34% (158/164) | 10.16 | 7.65 | 253.3 | 14.15% / 2.36% |
| Full-value swaps | 95.73% (157/164) | 10.28 | 7.81 | 339.8 | 15.79% / 2.65% |
| Direction-guarded residual | 96.34% (158/164) | 11.83 | 8.43 | 302.3 | 32.73% / 6.10% |

V2 tau=0.715741013753; first-canvas matching: {'absolute_kept_ratio': 0.9368037430300155, 'kept_fraction_ratio': 0.9943726881324186, 'matched': False, 'sparsity_delta_pp': 0.531160265754127}.

**Reproducibility:** all22 full audit/clean comparisons per budget pass; independent engine prefixes and official rescores pass.
**Work limitation:** K128 thinking-disabled V2 misses the first-canvas0.25pp target: +0.53116pp sparsity and0.93680 absolute kept-tile ratio. The predeclared nearest-work rule retained its threshold. Whole-trajectory work is not matched in any panel by first-canvas calibration alone.
Dense FULL/PIECEWISE/all-kept references are shared between budgets and are not additional independent trials.
V2/V3a use Gaussian32 on all five GLOBAL layers; full-value swaps/residual affect only layer29. All LOCAL attention stays dense.
Overall sparsity counts denoising GLOBAL plus native-window LOCAL Q128/KV64 rectangles, excluding prefill/commits and additional selector/PV work. W includes the complete method cost; S/N is amortized.
Main accuracy uses seed42. Seed43 covers six examined questions and remains descriptive. Question-bootstrap intervals condition on the observed engines and do not establish seed robustness.
