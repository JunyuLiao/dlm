# HumanEval164 K256 seed42

Clean native accuracy and request latency; separate audit sparsity. Full inputs, generated text, gold and token IDs remain private.

## thinking_on

V2 threshold: 0.119043883153; full164 first-canvas matching: {'absolute_kept_ratio': 0.9496292205743136, 'kept_fraction_ratio': 0.9991418698119103, 'matched': True, 'sparsity_delta_pp': 0.08488885800082002}.

| Method | Correct/164 | N/problem | C/problem | N/C | T | S (s) | W (s) | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_before | 159/164 | 51.787 | 5.232 | 9.899 | 1213.32 | 1.15039 | 1.16967 | 48.168% | 10.818% |
| value_v2_online_preserve_mass | 155/164 | 56.537 | 5.689 | 9.938 | 1325.75 | 1.30440 | 1.32367 | 29.977% | 6.910% |
| value_v3a_singleton_delete | 160/164 | 49.774 | 5.152 | 9.660 | 1198.29 | 1.13074 | 1.15001 | 54.401% | 13.593% |
| full_swap8 | 160/164 | 50.232 | 5.189 | 9.680 | 1203.18 | 1.81751 | 1.83676 | 53.977% | 13.294% |
| residual_direction_guard | 159/164 | 52.793 | 5.323 | 9.918 | 1227.57 | 1.29765 | 1.31692 | 48.881% | 11.194% |
| control_after | 159/164 | 52.030 | 5.226 | 9.957 | 1211.48 | 1.15439 | 1.17366 | 48.164% | 10.819% |
| dense_full_fix51994 | 158/164 | 48.104 | 5.195 | 9.259 | 1205.90 | 0.93033 | 0.95358 | 0.000% | 0.000% |
| dense_piecewise | 157/164 | 52.994 | 5.384 | 9.843 | 1239.13 | 1.02051 | 1.03978 | 0.000% | 0.000% |
| allkept_fa4 | 158/164 | 48.037 | 5.195 | 9.246 | 1205.16 | 1.01331 | 1.03270 | 0.000% | 0.000% |

Paired question-bootstrap intervals, technical bookend results, audit/clean drift, caps, physical totals and selector overhead are in summary.json.

## thinking_off

V2 threshold: 0.144224957031; full164 first-canvas matching: {'absolute_kept_ratio': 0.9774664336097018, 'kept_fraction_ratio': 0.9988324873046428, 'matched': True, 'sparsity_delta_pp': 0.11532416686446112}.

| Method | Correct/164 | N/problem | C/problem | N/C | T | S (s) | W (s) | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_before | 155/164 | 10.280 | 1.409 | 7.299 | 241.88 | 0.23086 | 0.24978 | 14.243% | 2.421% |
| value_v2_online_preserve_mass | 154/164 | 10.677 | 1.421 | 7.515 | 243.84 | 0.23725 | 0.25614 | 8.374% | 1.448% |
| value_v3a_singleton_delete | 154/164 | 9.335 | 1.372 | 6.804 | 230.85 | 0.21656 | 0.23545 | 38.823% | 7.867% |
| full_swap8 | 154/164 | 10.073 | 1.402 | 7.183 | 240.43 | 0.28982 | 0.30871 | 35.044% | 6.909% |
| residual_direction_guard | 154/164 | 10.055 | 1.409 | 7.139 | 239.53 | 0.23866 | 0.25756 | 8.261% | 1.379% |
| control_after | 155/164 | 10.280 | 1.409 | 7.299 | 241.88 | 0.23083 | 0.24973 | 14.243% | 2.421% |
| dense_full_fix51994 | 157/164 | 10.335 | 1.372 | 7.533 | 233.99 | 0.19880 | 0.22114 | 0.000% | 0.000% |
| dense_piecewise | 157/164 | 9.445 | 1.323 | 7.138 | 232.00 | 0.18365 | 0.20258 | 0.000% | 0.000% |
| allkept_fa4 | 154/164 | 9.890 | 1.366 | 7.241 | 233.15 | 0.20647 | 0.22536 | 0.000% | 0.000% |

Paired question-bootstrap intervals, technical bookend results, audit/clean drift, caps, physical totals and selector overhead are in summary.json.

## Limits

- single seed42; work calibration overlaps evaluated prompts
- V2/V3a Gaussian32 all GLOBAL; swaps/residual modify GLOBAL layer29 only
- effective physical sparsity from separate audit; clean/audit drift explicitly reported
- Q128/KV64 geometry rectangles, not a count of CUDA CTAs
- S/N is amortized across native trajectory, not a controlled per-forward price

Charged GPU reservation: 15032.143 seconds. Older LongBench campaigns remain paused.
