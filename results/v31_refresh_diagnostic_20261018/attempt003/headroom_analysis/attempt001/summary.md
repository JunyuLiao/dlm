## Separate clean runtime headroom

selected batch-one headroom diagnostic; clean timing and audited work remain separate

| Arm | Requests | Mean score | Mean calls | Mean request s |
|---|---:|---:|---:|---:|
| allkept_fa4 | 4 | 0 | 338.5 | 13.2431 |
| dense_full_fix51994 | 4 | 0.5 | 241.5 | 10.2313 |
| dense_piecewise | 4 | 0.5 | 289.5 | 11.7634 |
| dynamic | 4 | 0.5 | 281.5 | 9.75161 |
| fixed16 | 4 | 0 | 212.5 | 8.01623 |

Execution sources: 6aab86325990edf6e5ab6bb72bcd909781234e97, db755c43be763a8af9edfe8a147ec333d8b8542b. Referenced clean leaves retain their original protocol and source; successor profiling is a separate execution source. Compare native trajectories before cost transfer.

JSON reports paired dynamic/fixed16 ratios and substrate-control differences. Audit spans, physical rectangles, and profiler activity are separate. The removable request-critical-path bound remains **unresolved**: inclusive spans and disjoint activity labels do not establish overlap or latency savings.

