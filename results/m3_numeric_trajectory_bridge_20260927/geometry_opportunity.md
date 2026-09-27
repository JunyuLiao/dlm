# Same-state geometry opportunity (development diagnostic)

Inputs: the private mpk/dllm `geometry_geometry_b85fa23` receipts, pinned by SHA-256 in [JSON](geometry_opportunity.json). The [CSV](geometry_opportunity.csv) has one row per state × geometry plus explicit failed rows. No prompts, tokens, Q/K/V or support tensors are copied here.

Six inputs (one per task per host) yielded **34 qualified layer-states**: 17 GLOBAL layer 5 and 17 LOCAL layer 0. Twelve bootstrap call-0 layer-states failed because live T was absent (`RuntimeError`, source line 222); they remain diagnostic failures, not invented neutral-T measurements. The production method itself uses neutral sensitivity when T is `None` at bootstrap, so these rows do not assert a production-call failure. The dllm RULER canvas reached calls 0–2 and the predeclared call 3 is missing. All six final captured calls had same-history native stop verified; these are the only near-stop labels. All 34 coarse route parity checks matched exactly. Each state uses the inherited frozen threshold; achieved work is measured, not inferred from the G75 policy name.

| Host | Task | Layer kind | Qualified | Coarse Q128/K64 kept legal % | Q16/K64 kept legal % | old GQA/K32 kept legal % | Coarse / Q16 / old median output rel-L2 | Native attention event ms | Fixed coarse event ms |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| dllm | aime26 | GLOBAL | 3 | 98.1 | 89.1 | 95.9 | 0.020 / 0.122 / 0.022 | 0.325 | 0.269 |
| dllm | aime26 | LOCAL | 3 | 80.5 | 61.2 | 48.1 | 0.054 / 0.107 / 0.123 | 0.057 | — |
| dllm | longbench_v2 | GLOBAL | 3 | 34.4 | 18.7 | 16.2 | 0.289 / 0.394 / 0.365 | 5.365 | 1.683 |
| dllm | longbench_v2 | LOCAL | 3 | 63.5 | 41.2 | 33.4 | 0.099 / 0.163 / 0.166 | 0.064 | — |
| dllm | ruler4k | GLOBAL | 2 | 26.5 | 16.5 | 16.7 | 0.121 / 0.199 / 0.152 | 1.282 | 0.424 |
| dllm | ruler4k | LOCAL | 2 | 41.3 | 26.1 | 22.3 | 0.114 / 0.150 / 0.160 | 0.066 | — |
| mpk | aime26 | GLOBAL | 3 | 97.4 | 85.0 | 93.9 | 0.019 / 0.093 / 0.043 | 0.286 | 0.228 |
| mpk | aime26 | LOCAL | 3 | 73.0 | 57.7 | 46.5 | 0.089 / 0.142 / 0.148 | 0.090 | — |
| mpk | longbench_v2 | GLOBAL | 3 | 34.0 | 17.4 | 15.2 | 0.231 / 0.398 / 0.382 | 4.299 | 1.487 |
| mpk | longbench_v2 | LOCAL | 3 | 53.3 | 34.0 | 28.4 | 0.106 / 0.152 / 0.163 | 0.070 | — |
| mpk | ruler4k | GLOBAL | 3 | 36.8 | 23.6 | 24.7 | 0.209 / 0.315 / 0.183 | 1.329 | 0.494 |
| mpk | ruler4k | LOCAL | 3 | 45.3 | 28.5 | 24.2 | 0.088 / 0.142 / 0.144 | 0.071 | — |

Kept fractions weight actual legal pairs across captured states **within each host/task/layer**; error figures are medians of per-state full-FP32-reference relative L2. Error tails are material: see CSV `output_row_p99`, `output_row_max`, `removed_mass_row_p99` and JSON maxima. Lower kept work at the inherited threshold is not a matched-error or task-quality win. The LOCAL row is explicitly a **current-score oracle**, since production M3 routes only GLOBAL and has no LOCAL historical cache. `g0_extra_pairs` quantifies closure excess from a *common independent rowwise K32 logical support*; it does not predict the selector result, which reruns its retained-state recurrence separately for each geometry. `mass_argmax_matched_*` is an equal-tile-count historical-mass control, distinct from the value-aware rule and the old production G75/L30 selector.

| Host | Task | Layer | G0 closure extra legal pairs, coarse / Q16 / old (median state) | Matched historical-mass versus value-aware symmetric legal pairs, coarse / Q16 / old (median state) |
|---|---|---|---|---|
| dllm | aime26 | GLOBAL | 1165440 / 839936 / 1179904 | 0 / 68608 / 48128 |
| dllm | aime26 | LOCAL | 1607328 / 1054880 / 840352 | 397312 / 651776 / 731648 |
| dllm | longbench_v2 | GLOBAL | 35350836 / 15828532 / 15923404 | 17811456 / 12039296 / 11875648 |
| dllm | longbench_v2 | LOCAL | 3868428 / 2629796 / 1857588 | 1481856 / 1658640 / 1670048 |
| dllm | ruler4k | GLOBAL | 4877516 / 2364300 / 2746476 | 1581568 / 1210144 / 1252768 |
| dllm | ruler4k | LOCAL | 3397682 / 1693242 / 1240770 | 1529472 / 1426736 / 1379528 |
| mpk | aime26 | GLOBAL | 689977 / 430556 / 693788 | 13952 / 18656 / 33248 |
| mpk | aime26 | LOCAL | 755440 / 448768 / 302015 | 304896 / 472144 / 397488 |
| mpk | longbench_v2 | GLOBAL | 30481564 / 12976268 / 9025366 | 12824960 / 8011152 / 6661392 |
| mpk | longbench_v2 | LOCAL | 3663754 / 2254826 / 1765242 | 1587712 / 1664464 / 1649136 |
| mpk | ruler4k | GLOBAL | 6250504 / 3117704 / 3727112 | 2321408 / 1997824 / 2418688 |
| mpk | ruler4k | LOCAL | 3167466 / 1695170 / 1289810 | 1456128 / 1291680 / 1301696 |

G0 counts close one common rowwise K32 support into each layout; the common logical kept pairs are recorded per state in JSON. The historical-mass comparison fixes each geometry’s value-aware tile count before applying the argmax-mass rule. These are support-set differences, not task scores or latency.

Attention timings are 3 warmed CUDA-event repetitions per state, with native/fixed order rotated, both on the **M3 method-path QKV** and with fixed support already prepared. They exclude QKV projection, route/selection, metadata, full decoder, sampler and stopping. Synchronised whole-call wall repetitions are in CSV/JSON. Only GLOBAL has a fixed coarse pre-QK comparison; LOCAL production is native. The full-current-V projection event prices in JSON deliberately project all tokens and do not represent the existing leased prefix producer. Cached production norm versus full recomputation differences are reported rather than silently treated as exact identity.

The selected geometry should be judged next at matched physical work or matched output-error band, then with a real executable consumer and complete forward/request timings. These diagnostics alone do not choose a speed winner.
