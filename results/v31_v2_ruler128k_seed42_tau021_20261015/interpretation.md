# RULER128K: completed seed-42 rerun

Both methods generated fresh answers for the same 104 prompts, eight per each
of 13 tasks. V31 is Yuhan's inherited control. V2 is exactly
`value_v2_online_preserve_mass`, with uniform GLOBAL threshold 0.021 and no
quota. Both use the same frozen source, runtime, model, native sampler and
official RULER scorer. Projection rank 32/seed 1729 and seed base 31 stay fixed.

| Metric | V31 control | V2 tau=0.021 |
|---|---:|---:|
| Official RULER score, /100 | 86.250 | 85.769 |
| Full-credit problems | 81/104 | 81/104 |
| Denoising calls, total | 545 | 586 |
| Calls/problem | 5.240 | 5.635 |
| Median calls/problem | 4 | 4 |
| P90 calls/problem | 8.7 | 9.7 |
| Maximum calls/problem | 14 | 35 |
| Committed canvases, total | 104 | 104 |
| Committed canvases/problem | 1.000 | 1.000 |
| Calls/committed canvas | 5.240 | 5.635 |
| Output tokens, total | 2,700 | 2,694 |
| Output tokens/problem | 25.962 | 25.904 |
| GLOBAL eligible tiles | 177,836,960 | 191,311,840 |
| GLOBAL kept tiles | 96,039,360 | 95,030,781 |
| GLOBAL skipped tiles | 81,797,600 | 96,281,059 |
| GLOBAL physical sparsity | 46.00% | 50.33% |
| Overall eligible tiles | 186,548,160 | 200,679,040 |
| Overall kept tiles | 104,750,560 | 104,397,981 |
| Overall physical sparsity | 43.85% | 47.98% |
| LOCAL sparsity | 0% | 0% |
| Length-capped / empty predictions | 1 / 0 | 2 / 0 |

RULER gives fractional credit. Its primary score is not binary accuracy; both
methods have 77.88% full-credit problems, and their full-credit sets coincide.
V2 minus V31 is -0.481 score points, with a 100,000-draw 13-task-cluster
bootstrap 95% interval [-1.442, 0.000]. The secondary full-credit McNemar p is
1.0. This does not establish noninferiority. One problem's fractional score
improves, five worsen, and 98 tie. At task level, the net score loss is in
common-word extraction: 70.00 versus 63.75; the other 12 task averages tie.

V2 uses **7.52% more denoising calls**. This is concentrated in variable
tracking: 61 to 106 total calls, while the other tasks combined fall from
484 to 480. The two largest increases are 7 to 35 calls and 6 to 16 calls;
the former reaches its output-length cap. Across problems, calls decrease on
eight, increase on 11, and tie on 85. The median paired change is zero.
The shorter-trajectory observation from LongBench does not carry over to this
seed-42 RULER panel.

V2 retains **1.05% fewer GLOBAL tiles**, or only 0.34% fewer overall tiles.
Its higher percentage sparsity partly reflects a larger eligible denominator
from its longer trajectory. Dense/initial/refresh phases contribute 94.0% of
V31's retained GLOBAL work and 95.7% of V2's. Held-phase kept fractions are
6.51% and 4.06%, respectively; carried fractions are 6.71% and 4.15%.
Smaller sparse supports therefore produce little reduction in total consumer
work on this short-output workload.

The work unit is an audited Q128 x KV64 attention rectangle, not a literal
CUDA kernel launch. Counts cover denoising attention and keep dense LOCAL
work in the overall denominator. `C` is the frozen worker's committed-canvas
count, `ceil(output_tokens/256)`; adapter lifecycle IDs are separate.

Selector costs are additional to the consumer tiles. V2 performs 57,019,520
discovery QK rectangles (plus rank-32 PV) and 57,019,520 candidate evaluations.
Instrumented discovery/selection times are 12.30/27.67 seconds. Recorded audit
request times are 686.83/724.84 seconds for V31/V2; recorded decode times are
31.38/69.53 seconds. These are instrumented audits, not clean latency or
throughput measurements. The logical split-FA4 call counts are 1,860/2,055;
they are not CUDA launch counts either.

All eight shards and 208 official score cells are complete. Independent
verification checks 886 frozen source files, 11 inputs, all source/seed/scorer
bindings, physical and phase sums, unchanged FA4 output, zero invalid rows,
zero request CUDA captures and zero LOCAL skips. Qualification repeats match
the full runs on output hash, token count, calls and canvases for both methods.
All 16 model/tokenizer inventory files (51.68 GB) also match their frozen hashes.
The inherited `value_budget_policy` receipt label is static; the verified
`triton_log_mass_online` branch uses the threshold without a quota.

This is a paired rerun on a previously examined subset, not a new held-out
confirmation. It is not a controlled threshold ablation against the older
seed-1/tau=0.027 archive, which also used a different source snapshot. No
additional threshold, dense arm or paused campaign was launched. H100 is idle;
clean timing stays paused. Per-task scores, phase accounting, distributions,
selector costs and verification receipts accompany this report.
