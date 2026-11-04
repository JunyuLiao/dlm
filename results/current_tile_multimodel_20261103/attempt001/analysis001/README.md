# Current-tile V2: 6 completed model/benchmark panels

Only the new V2 variants were benchmarked. Prior methods remain unchanged. All selected complete panels achieve40–60% physical eligible denoising-consumer sparsity after small count-only calibration; any rejected completed attempt is retained in attempts.json.

| Model | Benchmark | Score | Requests | Mean output | Steps/request | Canvases/request | Steps/canvas | Sparsity | Threshold | Attempt |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llada22 | humaneval | 72.56% | 164 | 219.3 | 26.09 | 7.77 | 3.355 | 55.67% | 0.213270967 | 1 |
| trado | humaneval | 74.39% | 164 | 122.2 | 59.99 | 31.32 | 1.915 | 53.31% | 0.127350308 | 2 |
| trado | ruler32k | 64.72% | 130 | 31.0 | 21.28 | 8.56 | 2.486 | 49.45% | 0.000139002585 | 1 |
| llada22 | ruler128k | 85.21% | 130 | 28.0 | 8.01 | 1.82 | 4.411 | 49.63% | 2.21165362e-05 | 2 |
| trado | aime26 | 8.89% | 90 | 1018.1 | 527.96 | 255.24 | 2.068 | 50.51% | 0.0133616799 | 1 |
| llada22 | aime26 | 30.00% | 90 | 5964.2 | 1445.84 | 187.31 | 7.719 | 51.95% | 0.00302488967 | 1 |

AIME:30 problems x seeds42/43/44. HumanEval:164 problems,seed42. RULER:10 samples/task across13 official tasks,seed42; fractional retrieval score. RULER128K on LLaDA/Gemma,32K on TraDo.

LLaDA and TraDo preserve their native32/4-token samplers, all-layer sparse scope, Q128/KV64, Gaussian32 and four-call reuse. Gemma preserves its256-token native sampler, GLOBAL-only layers5/11/17/23/29, dense25 LOCAL layers, dense prefill/commits, and original carry/progress/sticky lifecycle. Only initialization/order changes. Per-model output caps and thinking settings match the original setups.

Sparsity includes mandatory/current tiles and all native denoising consumer phases, excludes discovery/prefill/commits. Gemma also reports overall GLOBAL+LOCAL sparsity in summary.json. Short-benchmark calibration uses six fixed prompt-length strata of examined benchmark prompts, count-only and transductive; retrieval uses13 separately generated development prompts. No gold or score chooses a threshold. A completed outside40–60% panel triggers a new calibration/attempt; inside the range it is retained without retuning.

Historical references are in historical_results_unchanged.json. The old Gemma RULER cohort contains104 items (8/task); its score is a separate cohort reference, not a paired baseline for the new130. TraDo32K has no historical matched panel. Paired intervals are emitted only after prompt/seed/cap identity checks; repeated examined benchmarks are exploratory. No clean latency or universal accuracy-preservation claim.
