# Small frozen P0/P1 execution tradeoff

Screen002 measured N4 complete model calls on predeclared native states, before any scored answer. All ratios method/native within the same GPU. No achieved50/60 label is attached to transferred thresholds. A8 and the original native sampler/stop remain fixed.

| Host | Dataset | P0 R2/native | P1 R2/native | P0 routed QK/PV skip | P1 routed QK/PV skip |
|---|---|---:|---:|---:|---:|
| mpk | ruler4k | 1.018 | 1.018 | 31.1% / 42.7% | 38.7% / 52.6% |
| mpk | aime26 | 1.035 | 1.035 | 14.7% / 20.3% | 20.5% / 28.1% |
| mpk | longbench_v2 | 0.969 | 0.961 | 50.4% / 67.9% | 55.4% / 74.5% |
| dllm | ruler4k | 1.024 | 1.019 | 43.1% / 58.1% | 48.0% / 64.5% |
| dllm | aime26 | 1.027 | 1.026 | 17.8% / 25.0% | 23.5% / 32.8% |
| dllm | longbench_v2 | 0.939 | 0.928 | 46.6% / 65.0% | 52.2% / 72.0% |

Physical fractions here cover the five routed GLOBAL layers, using exact legal-pair denominators including partial tiles. The 25 native LOCAL layers are not counter-instrumented and are excluded, so these are not full-model skip fractions. Anchors/fresh T still compute full current QK even when PV support is dropped.

P1 creates more physical sparsity, but its complete-forward gains are small and not consistent across all families. Keep P0 according to the pre-answer rule. Do not search intermediate thresholds or choose policy from accuracy. The screen does not establish whether extra adaptive calls will erase a forward saving; whole requests remain to be measured.

N4 pays one A per four calls and must not be priced as an asymptotic A8 average. R2/R3 include actual M1 decisions, while B performs fewer decisions; the longer native-reached replay will quantify anchor/decision amortization. Forward-cost ratios cannot be multiplied by unrelated historical generation gains.

Selected scope GLOBAL is restricted. ALL-layer execution is measured and currently negative. The chosen GLOBAL Triton consumer had only ~0.03% lower equal-family/equal-host mean than Hopper, inconsistent by host; this determines a common implementation by the predeclared rule, not a demonstrated consumer speedup. No R winner has been chosen from task outcomes.
