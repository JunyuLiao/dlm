# Replicated ruler128k development evaluation

exploratory repeated evaluation, not pass@2; separate item/task and engine-block intervals are not a joint confidence region. Clean timing remains paused.

| Method | Score | N/trial | N/canvas | C/trial | T/trial | GLOBAL kept/trial |
|---|---:|---:|---:|---:|---:|---:|
| current_v31_control | 86.250 | 5.240 | 5.240 | 1.000 | 26.0 | 923455.4 |
| dense_full_fix51994 | 86.538 | 5.750 | 5.750 | 1.000 | 25.8 | 1876236.9 |
| dense_piecewise | 86.538 | 5.654 | 5.654 | 1.000 | 25.7 | 1844836.9 |
| allkept_fa4 | 84.856 | 5.558 | 5.558 | 1.000 | 25.7 | 1813915.4 |
| full_swap8 | 87.308 | 5.231 | 5.231 | 1.000 | 26.0 | 923250.8 |
| residual_direction_guard | 86.058 | 5.317 | 5.317 | 1.000 | 25.9 | 930996.9 |
| residual | 86.346 | 5.375 | 5.375 | 1.000 | 25.9 | 926320.0 |
| full_swap8_direction_guard | 85.962 | 5.481 | 5.481 | 1.000 | 26.0 | 934475.4 |
