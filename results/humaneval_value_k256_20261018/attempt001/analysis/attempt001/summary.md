# HumanEval164, seed42, K256: verified results

All164 tasks, one measured generation per task and arm, thinking on/off, native output limit8192. K256 is four selectable complete prefix KV64 blocks per query-head/Q128 tile; canvas/boundary tiles stay dense. V2 uses a calibrated variable-budget threshold. LOCAL remains dense.

**Interpretation:** clean accuracy and request time come from the same generation records. Audit accuracy and physical work are reported separately. Audit and clean output hashes differ substantially, including dense controls; the cause has not been localized. First-canvas threshold matching was established on the audit population and is not a verified clean-population work match.

V2/V3a use existing Gaussian32 statistics on all five GLOBAL layers. Full-value swaps and direction-guarded residual use full V only at GLOBAL layer29, with V31 selection elsewhere. Residual explicitly modifies held attention output; the other value methods change support only. The inherited V31 lineage is Yuhan's.

## thinking_on: clean generation

| Method | Pass@1 | N mean / median / P90 / max | C | N/C | T | S (s) | S/N (ms) | W (s) | Output caps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_before | 159/164 (96.95%) | 51.787 / 36.5 / 101.9 / 295 | 5.232 | 9.899 | 1213.32 | 1.15039 | 22.214 | 1.16967 | 0 |
| value_v2_online_preserve_mass | 155/164 (94.51%) | 56.537 / 37.5 / 91.0 / 640 | 5.689 | 9.938 | 1325.75 | 1.30440 | 23.072 | 1.32367 | 3 |
| value_v3a_singleton_delete | 160/164 (97.56%) | 49.774 / 36.0 / 97.1 / 386 | 5.152 | 9.660 | 1198.29 | 1.13074 | 22.717 | 1.15001 | 0 |
| full_swap8 | 160/164 (97.56%) | 50.232 / 37.5 / 81.4 / 490 | 5.189 | 9.680 | 1203.18 | 1.81751 | 36.182 | 1.83676 | 1 |
| residual_direction_guard | 159/164 (96.95%) | 52.793 / 36.0 / 93.6 / 518 | 5.323 | 9.918 | 1227.57 | 1.29765 | 24.580 | 1.31692 | 1 |
| control_after | 159/164 (96.95%) | 52.030 / 36.5 / 101.9 / 295 | 5.226 | 9.957 | 1211.48 | 1.15439 | 22.187 | 1.17366 | 0 |
| dense_full_fix51994 | 158/164 (96.34%) | 48.104 / 38.0 / 81.1 / 511 | 5.195 | 9.259 | 1205.90 | 0.93033 | 19.340 | 0.95358 | 1 |
| dense_piecewise | 157/164 (95.73%) | 52.994 / 36.0 / 88.4 / 858 | 5.384 | 9.843 | 1239.13 | 1.02051 | 19.257 | 1.03978 | 1 |
| allkept_fa4 | 158/164 (96.34%) | 48.037 / 35.0 / 81.7 / 568 | 5.195 | 9.246 | 1205.16 | 1.01331 | 21.094 | 1.03270 | 0 |

## thinking_on: separate audit generation

| Method | Pass@1 | N | C | N/C | T | GLOBAL sparsity | LOCAL sparsity | Overall sparsity | Output caps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_before | 162/164 (98.78%) | 51.061 | 5.238 | 9.749 | 1206.68 | 48.168% | 0.0% | 10.818% | 1 |
| value_v2_online_preserve_mass | 158/164 (96.34%) | 50.061 | 5.372 | 9.319 | 1248.31 | 29.977% | 0.0% | 6.910% | 1 |
| value_v3a_singleton_delete | 161/164 (98.17%) | 56.835 | 5.561 | 10.220 | 1297.15 | 54.401% | 0.0% | 13.593% | 1 |
| full_swap8 | 158/164 (96.34%) | 54.409 | 5.372 | 10.128 | 1239.68 | 53.977% | 0.0% | 13.294% | 1 |
| residual_direction_guard | 159/164 (96.95%) | 51.451 | 5.293 | 9.721 | 1213.43 | 48.881% | 0.0% | 11.194% | 1 |
| control_after | 162/164 (98.78%) | 51.073 | 5.238 | 9.751 | 1206.47 | 48.164% | 0.0% | 10.819% | 1 |
| dense_full_fix51994 | 156/164 (95.12%) | 47.896 | 5.189 | 9.230 | 1204.26 | 0.000% | 0.0% | 0.000% | 0 |
| dense_piecewise | 158/164 (96.34%) | 49.421 | 5.335 | 9.263 | 1237.20 | 0.000% | 0.0% | 0.000% | 1 |
| allkept_fa4 | 160/164 (97.56%) | 49.244 | 5.256 | 9.369 | 1219.92 | 0.000% | 0.0% | 0.000% | 0 |

V2 threshold 0.119043883153; audit first-canvas matching: {'absolute_kept_ratio': 0.9496292205743136, 'kept_fraction_ratio': 0.9991418698119103, 'matched': True, 'sparsity_delta_pp': 0.08488885800082002}.

Clean versus audit hash/score disagreements:

| Method | Output hash differences | Correctness differences |
|---|---:|---:|
| control_before | 163/164 | 7/164 |
| value_v2_online_preserve_mass | 164/164 | 11/164 |
| value_v3a_singleton_delete | 163/164 | 7/164 |
| full_swap8 | 163/164 | 6/164 |
| residual_direction_guard | 163/164 | 8/164 |
| control_after | 163/164 | 7/164 |
| dense_full_fix51994 | 164/164 | 8/164 |
| dense_piecewise | 164/164 | 11/164 |
| allkept_fa4 | 164/164 | 8/164 |

V31 technical bookends: {'clean': {'hash_differences': 1, 'N_differences': 1, 'max_absolute_N_difference': 40, 'score_differences': 0}, 'audit': {'hash_differences': 1, 'N_differences': 1, 'max_absolute_N_difference': 2, 'score_differences': 0}}.

Paired20,000-draw question-bootstrap95% intervals (single seed, conditional on observed engines):

| Method | Accuracy delta vs V31 (pp) [CI] | Request speedup vs V31 [CI] |
|---|---:|---:|
| value_v2_online_preserve_mass | -2.439 [-6.097560975609756, 1.2195121951219512] | 0.8837 [0.7521947747049498, 1.037140248717948] |
| value_v3a_singleton_delete | +0.610 [-2.4390243902439024, 3.658536585365854] | 1.0171 [0.9427717006111759, 1.1029339927614839] |
| full_swap8 | +0.610 [-1.2195121951219512, 3.048780487804878] | 0.6368 [0.5764428835382727, 0.7067602643036488] |
| residual_direction_guard | +0.000 [-3.048780487804878, 3.048780487804878] | 0.8882 [0.8072882700771941, 0.9740992510494385] |

Audit selector/correction envelopes:

| Method | Recorded ms/problem | Extra full-value anchors/problem |
|---|---:|---:|
| value_v2_online_preserve_mass | {'discovery_ms': 15.959378159082517, 'selection_ms': 101.09018056349056} | not applicable |
| value_v3a_singleton_delete | {'discovery_ms': 19.709352226882444, 'selection_ms': 36.70617015459916} | not applicable |
| full_swap8 | {'anchor_module_cuda_elapsed_ms': 745.225678001962} | 8.5 |
| residual_direction_guard | {'anchor_module_cuda_elapsed_ms': 142.77461506680746} | 8.341463414634147 |

## thinking_off: clean generation

| Method | Pass@1 | N mean / median / P90 / max | C | N/C | T | S (s) | S/N (ms) | W (s) | Output caps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_before | 155/164 (94.51%) | 10.280 / 7.0 / 16.0 / 120 | 1.409 | 7.299 | 241.88 | 0.23086 | 22.457 | 0.24978 | 0 |
| value_v2_online_preserve_mass | 154/164 (93.90%) | 10.677 / 7.0 / 16.0 / 156 | 1.421 | 7.515 | 243.84 | 0.23725 | 22.221 | 0.25614 | 0 |
| value_v3a_singleton_delete | 154/164 (93.90%) | 9.335 / 7.0 / 14.0 / 73 | 1.372 | 6.804 | 230.85 | 0.21656 | 23.197 | 0.23545 | 0 |
| full_swap8 | 154/164 (93.90%) | 10.073 / 7.0 / 14.7 / 129 | 1.402 | 7.183 | 240.43 | 0.28982 | 28.771 | 0.30871 | 0 |
| residual_direction_guard | 154/164 (93.90%) | 10.055 / 7.0 / 15.0 / 118 | 1.409 | 7.139 | 239.53 | 0.23866 | 23.736 | 0.25756 | 0 |
| control_after | 155/164 (94.51%) | 10.280 / 7.0 / 16.0 / 120 | 1.409 | 7.299 | 241.88 | 0.23083 | 22.454 | 0.24973 | 0 |
| dense_full_fix51994 | 157/164 (95.73%) | 10.335 / 8.0 / 16.0 / 146 | 1.372 | 7.533 | 233.99 | 0.19880 | 19.235 | 0.22114 | 0 |
| dense_piecewise | 157/164 (95.73%) | 9.445 / 8.0 / 15.7 / 62 | 1.323 | 7.138 | 232.00 | 0.18365 | 19.444 | 0.20258 | 0 |
| allkept_fa4 | 154/164 (93.90%) | 9.890 / 8.0 / 15.7 / 62 | 1.366 | 7.241 | 233.15 | 0.20647 | 20.876 | 0.22536 | 0 |

## thinking_off: separate audit generation

| Method | Pass@1 | N | C | N/C | T | GLOBAL sparsity | LOCAL sparsity | Overall sparsity | Output caps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_before | 156/164 (95.12%) | 10.494 | 1.354 | 7.752 | 235.16 | 14.243% | 0.0% | 2.421% | 0 |
| value_v2_online_preserve_mass | 159/164 (96.95%) | 10.232 | 1.354 | 7.559 | 237.12 | 8.374% | 0.0% | 1.448% | 0 |
| value_v3a_singleton_delete | 158/164 (96.34%) | 12.683 | 1.427 | 8.889 | 257.05 | 38.823% | 0.0% | 7.867% | 0 |
| full_swap8 | 156/164 (95.12%) | 12.140 | 1.421 | 8.545 | 252.99 | 35.044% | 0.0% | 6.909% | 0 |
| residual_direction_guard | 157/164 (95.73%) | 9.683 | 1.335 | 7.251 | 230.52 | 8.261% | 0.0% | 1.379% | 0 |
| control_after | 156/164 (95.12%) | 10.494 | 1.354 | 7.752 | 235.16 | 14.243% | 0.0% | 2.421% | 0 |
| dense_full_fix51994 | 153/164 (93.29%) | 9.884 | 1.354 | 7.302 | 233.30 | 0.000% | 0.0% | 0.000% | 0 |
| dense_piecewise | 155/164 (94.51%) | 9.848 | 1.402 | 7.022 | 235.51 | 0.000% | 0.0% | 0.000% | 0 |
| allkept_fa4 | 157/164 (95.73%) | 10.189 | 1.366 | 7.460 | 238.65 | 0.000% | 0.0% | 0.000% | 0 |

V2 threshold 0.144224957031; audit first-canvas matching: {'absolute_kept_ratio': 0.9774664336097018, 'kept_fraction_ratio': 0.9988324873046428, 'matched': True, 'sparsity_delta_pp': 0.11532416686446112}.

Clean versus audit hash/score disagreements:

| Method | Output hash differences | Correctness differences |
|---|---:|---:|
| control_before | 109/164 | 9/164 |
| value_v2_online_preserve_mass | 108/164 | 5/164 |
| value_v3a_singleton_delete | 109/164 | 6/164 |
| full_swap8 | 109/164 | 8/164 |
| residual_direction_guard | 109/164 | 7/164 |
| control_after | 109/164 | 9/164 |
| dense_full_fix51994 | 112/164 | 10/164 |
| dense_piecewise | 102/164 | 6/164 |
| allkept_fa4 | 107/164 | 11/164 |

V31 technical bookends: {'clean': {'hash_differences': 0, 'N_differences': 0, 'max_absolute_N_difference': 0, 'score_differences': 0}, 'audit': {'hash_differences': 0, 'N_differences': 0, 'max_absolute_N_difference': 0, 'score_differences': 0}}.

Paired20,000-draw question-bootstrap95% intervals (single seed, conditional on observed engines):

| Method | Accuracy delta vs V31 (pp) [CI] | Request speedup vs V31 [CI] |
|---|---:|---:|
| value_v2_online_preserve_mass | -0.610 [-3.658536585365854, 1.829268292682927] | 0.9751 [0.882272942158519, 1.067334541259568] |
| value_v3a_singleton_delete | -0.610 [-1.829268292682927, 0.0] | 1.0608 [0.9939580248652226, 1.1548069203470162] |
| full_swap8 | -0.610 [-1.829268292682927, 0.0] | 0.8091 [0.7470002744008851, 0.873837442895495] |
| residual_direction_guard | -0.610 [-1.829268292682927, 0.0] | 0.9698 [0.8936123204690253, 1.0644579027060097] |

Audit selector/correction envelopes:

| Method | Recorded ms/problem | Extra full-value anchors/problem |
|---|---:|---:|
| value_v2_online_preserve_mass | {'discovery_ms': 2.3908480023283785, 'selection_ms': 1.1635685830218037} | not applicable |
| value_v3a_singleton_delete | {'discovery_ms': 2.727024392565576, 'selection_ms': 9.69022946822934} | not applicable |
| full_swap8 | {'anchor_module_cuda_elapsed_ms': 71.51510136301924} | 0.8719512195121951 |
| residual_direction_guard | {'anchor_module_cuda_elapsed_ms': 11.805847452908028} | 0.6890243902439024 |

## Findings and boundaries

- V3a is the best clean point candidate among the value methods: thinking on +1 correct and 1.7% lower W versus V31; thinking off -1 correct and 5.7% lower W. Both request-speedup intervals include1. No accuracy improvement is established.
- All requested sparse methods are slower than dense FULL in the clean runs. PIECEWISE dense is faster still without thinking.
- Full-value swaps spend about745ms/problem (thinking) and72ms/problem (non-thinking) in the audit anchor envelope. Directional residual spends about143ms/12ms. These are diagnostic envelopes, not clean subtraction estimates.
- First-canvas calibration does not equalize the full trajectory. Audit V2 GLOBAL sparsity is29.98% versus V31 48.17% with thinking, and8.37% versus14.24% without thinking.
- Audit/clean trajectory differences also occur in dense controls. Quality rankings differ across passes. Do not attach audit masks/work to clean outputs as if they were identical, infer a clean work match, or claim robust superiority from these single-seed point estimates.
- Overall sparsity uses actual denoising GLOBAL full-support plus LOCAL window-intersecting Q128/KV64 rectangles. Prefill and commits remain dense and are outside this sparsity denominator. This is a common geometry accounting unit, not CUDA CTA count.
- C follows the inherited bench definition ceil(output_tokens/256). N is sampler_calls minus C and is independently checked against GLOBAL-call scope and available denoising-interval trace sums. S/N is amortized over native generation, not a controlled forward price.
- All planned records, source/input bindings and sampler geometry checks pass. There are58 completed engine jobs, zero failed jobs, zero measured CUDA captures, zero LOCAL skips, and zero fixed-K budget errors. Older LongBench campaigns remain paused.

