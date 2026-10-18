# Headroom trajectory and scope review

Headroom is partial: all clean controls and the shared/FULL profiles completed; the all-kept profile was budget-skipped. All six profiled requests differ from clean in output hash, denoising calls, canvases and output length. Direct cost transfer and the removable critical-path bound remain unresolved.

This is a selected two-question, batch-one diagnostic with one seed and two clean repetitions; it does not establish population accuracy or noninferiority.

## Clean results by question

| Tokens | Arm | Score | N | C | T | W, s | S/N, ms | Prefill/W |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 111616 | allkept_fa4 | 0 | 425 | 19 | 4826 | 19.0062 | 33.230 | 25.69% |
| 111616 | dense_full_fix51994 | 0 | 249 | 13 | 3161 | 13.5612 | 34.742 | 36.21% |
| 111616 | dense_piecewise | 0 | 335 | 15 | 3725 | 16.4189 | 34.385 | 29.84% |
| 111616 | dynamic | 0 | 238 | 11 | 2622 | 10.8750 | 25.178 | 44.90% |
| 111616 | fixed16 | 0 | 279 | 13 | 3170 | 11.5996 | 24.081 | 42.08% |
| 41189 | allkept_fa4 | 0 | 252 | 11 | 2723 | 7.4800 | 25.367 | 14.54% |
| 41189 | dense_full_fix51994 | 1 | 234 | 14 | 3452 | 6.9014 | 24.817 | 15.86% |
| 41189 | dense_piecewise | 1 | 244 | 13 | 3239 | 7.1079 | 24.655 | 15.37% |
| 41189 | dynamic | 1 | 325 | 16 | 3965 | 8.6282 | 23.203 | 12.60% |
| 41189 | fixed16 | 0 | 146 | 8 | 2024 | 4.4328 | 22.910 | 24.54% |

S/N is an amortized trajectory cost. All clean repeats match hash/N/C/T/score. Request-time spreads are at most 0.213%, and measured captures are zero. The systems observer found no measured JIT warnings; this is not an exhaustive JIT counter.

## Audit versus clean

| Index | Arm | Clean N → audit N | Clean T → audit T | Clean score → audit score | Same output hash | Complete module inventory |
|---:|---|---:|---:|---:|---|---|
| 339 | dynamic | 325 → 276 | 3965 → 3700 | 1 → 0 | False | True |
| 339 | fixed16 | 146 → 263 | 2024 → 3139 | 0 → 0 | False | True |
| 254 | dynamic | 238 → 190 | 2622 → 2125 | 0 → 0 | False | True |
| 254 | fixed16 | 279 → 170 | 3170 → 1821 | 0 → 0 | False | True |
| 339 | dense_full_fix51994 | 234 → 354 | 3452 → 4309 | 1 → 0 | False | False |
| 254 | dense_full_fix51994 | 249 → 993 | 3161 → 7641 | 0 → 0 | False | False |

FULL profiles observe only prefill modules: 90/11220 and 210/30900 expected attention calls. Missing denoising/commit scopes are unknown, not zero cost. Shared module coverage is complete, but its trajectory also changes. This does not isolate which instrumentation or execution difference caused the changes.

## Cost interpretation

For the dynamic audit, progress refresh is 123.52 ms/7.966 s at 41K and 158.72 ms/10.834 s at 111K: about 1.5% of audited request time. The distinct bootstrap, initial-selection, progress-refresh and commit parent spans sum to about 5.17% and 5.45%. Discovery is nested inside those parents and must not be added again. These include dispatch gaps and necessary computation; they are not bounds on removable clean critical-path time.

The dynamic 111K Chrome trace contains 168348 GPU activities, of which 35286 link to module scopes. GPU busy union is 9087.20 ms within 10826.58 ms, with 1739.38 ms of no-activity intervals. These gaps do not establish host causality. MoE classification based on kernel names is a candidate category, not full module attribution.

Physical fractions cover decoder GLOBAL+LOCAL rectangles only, excluding prefill/commit. Dynamic 111K skips 83.27% of GLOBAL or 79.02% of overall rectangles. Its dense bootstrap/initial/refresh phases occupy 59.71% of executed GLOBAL rectangles, while progress refresh occupies about 1.47% of audited request time. These denominators answer different questions. Equal K does not establish equal executed work across native trajectories.

A 10% incremental request gain from cheaper refresh alone is not established and looks unpromising for these batch-one cases. The clean V31 requests would need about 0.86 s and 1.09 s saved, or 11.4% and 18.1% of decode time. Causal trajectory diagnosis remains a separate question.

Clean leaves retain source 6aab8632; profiling uses source db755c43be763a8af9edfe8a147ec333d8b8542b. The failed earlier profile remains preserved and charged. The JSON and separate review receipt bind all source protocols and scored inputs. The observed headroom phase stop charged 1693.745 seconds; mutable coordinator state may subsequently advance.
