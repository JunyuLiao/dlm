# Equal-work V2 versus V31 official audit

The current V31 control and both V2 arms use the same 503-cell manifest, seed, runtime, model, native sampler, scorer, and source/protocol fingerprints. Audit instrumentation is present; wall time is descriptive and is not clean latency. Physical FA4 consumer work, V2 selector discovery/evaluation, and native trajectory work are separate quantities.

## Accuracy and trajectory

| arm | threshold | correct/503 | overall | Easy | Hard | Short | Medium | Long | N | C | N/problem | N/canvas | T | output-hash mismatches | token mismatches | denoising-call mismatches | capped | unparsed | no-final-response | truncated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_v31_control | control | 256/503 | 50.9 | 53.1 | 49.5 | 61.7 | 46.5 | 41.7 | 134521 | 6712 | 267.44 | 20.04 | 1652559 | — | — | — | 17 | 35 | 27 | 237 |
| v2_work_match_total | 0.015 | 263/503 | 52.3 | 54.2 | 51.1 | 63.3 | 48.4 | 41.7 | 124671 | 6441 | 247.85 | 19.36 | 1585338 | 503 | 494 | 500 | 14 | 34 | 28 | 237 |
| v2_work_match_early | 0.021 | 274/503 | 54.5 | 56.2 | 53.4 | 68.9 | 46.5 | 46.3 | 131786 | 6508 | 262.00 | 20.25 | 1600642 | 503 | 498 | 498 | 14 | 37 | 31 | 237 |

## Physical work and selector cost

| arm | threshold | GLOBAL eligible | GLOBAL kept | GLOBAL skipped | GLOBAL sparsity | overall eligible rectangles | overall sparsity | kept ratio vs V31 | C=20 early delta | selector QK/eval rectangles | discovery ms | selection ms | FA4 kept work |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2_work_match_total | 0.015 | 30,391,721,600 | 5,879,477,757 | 24,512,243,843 | 0.8065 | 32,385,461,600 | 0.7569 | 1.0329 | +217,382,567 (20 canvases) | 2,849,464,000 / 2,849,464,000 | 594820.7 | 426473.1 | 5,879,477,757 |
| v2_work_match_early | 0.021 | 32,408,240,000 | 5,248,339,635 | 27,159,900,365 | 0.8381 | 34,515,519,200 | 0.7869 | 0.9221 | +17,950,508 (20 canvases) | 2,847,457,760 / 2,847,457,760 | 594394.6 | 426229.8 | 5,248,339,635 |

Phase-level eligible/kept work is in [phase_accounting.csv](phase_accounting.csv); cumulative work and per-canvas item coverage are in [cumulative_canvas_work.csv](cumulative_canvas_work.csv). The scorer coverage and strata are in [official_scorer_summary.json](official_scorer_summary.json), and paired bootstrap/McNemar results are in each arm’s `paired_accuracy.json`.

Interpretation: `tau=0.015` matches total work on development but over-allocates C=20 work and retains 1.0329x official total tiles. `tau=0.021` matches C=20 work within 1.19% on the official panel but retains 0.9221x total tiles. Neither is an equal native trajectory or throughput result.
