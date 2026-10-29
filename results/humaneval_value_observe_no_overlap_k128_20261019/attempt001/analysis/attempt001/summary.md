# HumanEval164 K128 seed42

Clean native accuracy and request latency; separate audit sparsity. Full inputs, generated text, gold and token IDs remain private.

## thinking_on

V2 threshold: 0.560071620607; full164 first-canvas matching: {'absolute_kept_ratio': 1.0027716967227327, 'kept_fraction_ratio': 1.0002544141412766, 'matched': True, 'sparsity_delta_pp': -0.023822152179597822}.

| Method | Correct/164 | N/problem | C/problem | N/C | T | S (s) | W (s) | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_before | 157/164 | 57.415 | 5.360 | 10.712 | 1243.13 | 1.27847 | 1.29796 | 59.870% | 14.630% |
| value_v2_online_preserve_mass | 158/164 | 53.323 | 5.476 | 9.738 | 1277.45 | 1.20459 | 1.22407 | 56.641% | 14.744% |
| value_v3a_singleton_delete | 154/164 | 58.445 | 5.335 | 10.954 | 1245.00 | 1.32492 | 1.34440 | 61.818% | 15.724% |
| full_swap8 | 155/164 | 55.982 | 5.341 | 10.481 | 1233.15 | 1.99953 | 2.01900 | 59.735% | 14.802% |
| residual_direction_guard | 155/164 | 56.518 | 5.341 | 10.581 | 1237.51 | 1.40271 | 1.42219 | 60.891% | 15.400% |
| control_after | 157/164 | 57.415 | 5.360 | 10.712 | 1243.13 | 1.27847 | 1.29794 | 59.870% | 14.630% |
| dense_full_fix51994 | 161/164 | 48.079 | 5.195 | 9.255 | 1202.51 | 0.92936 | 0.95249 | 0.000% | 0.000% |
| dense_piecewise | 161/164 | 48.079 | 5.195 | 9.255 | 1202.51 | 0.94027 | 0.95967 | 0.000% | 0.000% |
| allkept_fa4 | 164/164 | 51.945 | 5.482 | 9.476 | 1266.24 | 1.09185 | 1.11127 | 0.000% | 0.000% |

Paired question-bootstrap intervals, technical bookend results, audit/clean drift, caps, physical totals and selector overhead are in summary.json.

## thinking_off

V2 threshold: 0.715741013753; full164 first-canvas matching: {'absolute_kept_ratio': 0.9368037430300155, 'kept_fraction_ratio': 0.9943726881324186, 'matched': False, 'sparsity_delta_pp': 0.531160265754127}.

| Method | Correct/164 | N/problem | C/problem | N/C | T | S (s) | W (s) | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_before | 156/164 | 11.488 | 1.384 | 8.300 | 243.12 | 0.25515 | 0.27420 | 29.656% | 5.477% |
| value_v2_online_preserve_mass | 160/164 | 10.439 | 1.354 | 7.712 | 233.58 | 0.23364 | 0.25270 | 18.355% | 3.096% |
| value_v3a_singleton_delete | 158/164 | 10.165 | 1.329 | 7.647 | 231.34 | 0.23430 | 0.25335 | 14.152% | 2.362% |
| full_swap8 | 157/164 | 10.280 | 1.317 | 7.806 | 229.80 | 0.32071 | 0.33977 | 15.792% | 2.650% |
| residual_direction_guard | 158/164 | 11.829 | 1.402 | 8.435 | 249.40 | 0.28323 | 0.30230 | 32.730% | 6.100% |
| control_after | 156/164 | 11.488 | 1.384 | 8.300 | 243.12 | 0.25454 | 0.27360 | 29.656% | 5.477% |
| dense_full_fix51994 | 156/164 | 9.470 | 1.341 | 7.059 | 228.49 | 0.18313 | 0.20567 | 0.000% | 0.000% |
| dense_piecewise | 156/164 | 9.470 | 1.341 | 7.059 | 228.49 | 0.18489 | 0.20384 | 0.000% | 0.000% |
| allkept_fa4 | 156/164 | 9.957 | 1.341 | 7.423 | 236.62 | 0.20905 | 0.22810 | 0.000% | 0.000% |

Paired question-bootstrap intervals, technical bookend results, audit/clean drift, caps, physical totals and selector overhead are in summary.json.

## Limits

- primary seed42; six-question seed43 confirmation is descriptive; calibration overlaps prompts
- deterministic model/helper compiler policy disables horizontal combo fusion; native graph modes remain
- V2/V3a Gaussian32 all GLOBAL; swaps/residual modify GLOBAL layer29 only
- attention-consumer sparsity from matched audit; extra selector/PV work is separate
- Q128/KV64 geometry rectangles, not a count of CUDA CTAs
- S/N is amortized across native trajectory, not a controlled per-forward price
- mass-observing FA4 disables intra-warp-group overlap; plain paths stay native; its cost is included in measured latency
- dense FULL/PIECEWISE/all-kept controls are the same completed common-source runs used for K256; they are not independent additional trials or additional GPU charges

Charged GPU reservation: 9212.142 seconds. Older LongBench campaigns remain paused.

Audit/clean output, interval and private-completion comparisons pass for all164 questions per method. Independent engine prefix replication passes. Independent official rescores agree. Seed43 covers six examined questions per method/mode and remains descriptive.
