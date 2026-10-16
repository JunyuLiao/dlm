# Replicated longbench_v2 development evaluation

exploratory repeated evaluation, not pass@2; separate item/task and engine-block intervals are not a joint confidence region. Clean timing remains paused.

| Method | Score | N/trial | N/canvas | C/trial | T/trial | GLOBAL kept/trial |
|---|---:|---:|---:|---:|---:|---:|
| current_v31_control | 45.703 | 251.039 | 21.646 | 11.598 | 2835.6 | 10428170.6 |
| dense_full_fix51994 | 51.562 | 234.625 | 20.183 | 11.625 | 2848.2 | 56678802.5 |
| dense_piecewise | 46.875 | 258.781 | 19.371 | 13.359 | 3293.0 | 62154952.5 |
| allkept_fa4 | 51.562 | 305.953 | 23.592 | 12.969 | 3199.4 | 81529147.5 |
| full_swap8 | 46.875 | 192.023 | 18.578 | 10.336 | 2500.9 | 8460611.2 |
| residual_direction_guard | 43.750 | 249.516 | 21.207 | 11.766 | 2884.5 | 10366455.0 |
| residual | 43.750 | 327.094 | 25.781 | 12.688 | 3132.5 | 12310997.5 |
| full_swap8_direction_guard | 47.656 | 299.438 | 24.320 | 12.312 | 3020.2 | 11410612.5 |
