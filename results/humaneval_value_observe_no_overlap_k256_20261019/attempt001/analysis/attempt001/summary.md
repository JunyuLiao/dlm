# HumanEval164 K256 seed42

Clean native accuracy and request latency; separate audit sparsity. Full inputs, generated text, gold and token IDs remain private.

## thinking_on

V2 threshold: 0.0944851926739; full164 first-canvas matching: {'absolute_kept_ratio': 0.9939731591937474, 'kept_fraction_ratio': 1.0017258234074917, 'matched': True, 'sparsity_delta_pp': -0.17089518140988025}.

| Method | Correct/164 | N/problem | C/problem | N/C | T | S (s) | W (s) | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_before | 159/164 | 49.256 | 5.122 | 9.617 | 1189.19 | 1.10458 | 1.12399 | 48.325% | 10.922% |
| value_v2_online_preserve_mass | 162/164 | 50.043 | 5.268 | 9.499 | 1223.60 | 1.12822 | 1.14762 | 25.618% | 5.824% |
| value_v3a_singleton_delete | 159/164 | 51.854 | 5.323 | 9.741 | 1233.70 | 1.18622 | 1.20563 | 50.587% | 11.857% |
| full_swap8 | 155/164 | 49.085 | 5.183 | 9.471 | 1198.24 | 1.80595 | 1.82536 | 48.880% | 11.202% |
| residual_direction_guard | 158/164 | 52.543 | 5.293 | 9.927 | 1228.07 | 1.30191 | 1.32133 | 50.979% | 11.921% |
| control_after | 159/164 | 49.256 | 5.122 | 9.617 | 1189.19 | 1.10463 | 1.12404 | 48.325% | 10.922% |
| dense_full_fix51994 | 161/164 | 48.079 | 5.195 | 9.255 | 1202.51 | 0.92936 | 0.95249 | 0.000% | 0.000% |
| dense_piecewise | 161/164 | 48.079 | 5.195 | 9.255 | 1202.51 | 0.94027 | 0.95967 | 0.000% | 0.000% |
| allkept_fa4 | 164/164 | 51.945 | 5.482 | 9.476 | 1266.24 | 1.09185 | 1.11127 | 0.000% | 0.000% |

Paired question-bootstrap intervals, technical bookend results, audit/clean drift, caps, physical totals and selector overhead are in summary.json.

## thinking_off

V2 threshold: 0.0981299396515; full164 first-canvas matching: {'absolute_kept_ratio': 0.9813049853372434, 'kept_fraction_ratio': 1.0009674971028133, 'matched': True, 'sparsity_delta_pp': -0.09601062555383288}.

| Method | Correct/164 | N/problem | C/problem | N/C | T | S (s) | W (s) | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control_before | 157/164 | 10.774 | 1.372 | 7.853 | 244.56 | 0.24293 | 0.26218 | 20.031% | 3.579% |
| value_v2_online_preserve_mass | 158/164 | 10.854 | 1.409 | 7.706 | 247.77 | 0.24482 | 0.26408 | 5.168% | 0.889% |
| value_v3a_singleton_delete | 157/164 | 10.305 | 1.335 | 7.717 | 234.32 | 0.24081 | 0.26006 | 9.504% | 1.587% |
| full_swap8 | 156/164 | 10.445 | 1.341 | 7.786 | 235.07 | 0.29498 | 0.31423 | 14.663% | 2.515% |
| residual_direction_guard | 156/164 | 10.280 | 1.341 | 7.664 | 235.20 | 0.24508 | 0.26435 | 10.074% | 1.685% |
| control_after | 157/164 | 10.774 | 1.372 | 7.853 | 244.56 | 0.24312 | 0.26238 | 20.031% | 3.579% |
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

Charged GPU reservation: 12954.977 seconds. Older LongBench campaigns remain paused.

Audit/clean output, interval and private-completion comparisons pass for all164 questions per method. Independent engine prefix replication passes. Independent official rescores agree. Seed43 covers six examined questions per method/mode and remains descriptive.
