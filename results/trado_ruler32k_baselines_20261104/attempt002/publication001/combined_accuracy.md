# Completed two-model accuracy table, including TraDo RULER32K baselines

| Model | Benchmark | Native dense | All-kept FA4 | V31 | Original V2 | V3a | Current-tile V2 |
|---|---|---:|---:|---:|---:|---:|---:|
| llada22 | aime26 | 40.00% | 32.22% | 23.33% | 31.11% | 27.78% | 30.00% |
| llada22 | humaneval | 85.98% | 84.15% | 50.61% | 6.10% | 62.80% | 72.56% |
| llada22 | ruler128k | 85.44% | 84.82% | 86.01% | 86.23% | 85.03% | 85.21% |
| trado | aime26 | 8.89% | 4.44% | 3.33% | 6.67% | 2.22% | 8.89% |
| trado | humaneval | 79.27% | 78.05% | 42.68% | 76.22% | 47.56% | 74.39% |
| trado | ruler32k | 64.64% | 64.85% | 63.49% | — | 64.96% | 64.72% |

Existing results remain tied to their frozen historical artifacts. No original-V2 TraDo RULER32K evaluation exists. See README.md for work, scope, scoring and uncertainty.
