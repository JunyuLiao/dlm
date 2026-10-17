# Completed bounded native cross

This diagnostic uses the predeclared overlap questions0 and49, panel seeds1
and42, reference/device implementations within each engine, V31 before/after,
and two independent engines per pass. Engine order is audit/clean/clean/audit;
candidate order rotates/reverses. All96 measured requests and20 warmups finish.
No valid outcome is discarded. Source is frozen at dcae0ca0. All structural
receipts pass; there are no measured JIT warnings or allocator OOM warnings.
GPU process reservation is2018.38s (33.64 minutes), including startup/warmup.

This is two previously examined questions, not a new benchmark estimate or
held-out confirmation. The questions cover a truncated120K prompt and an
untruncated16K prompt. Data:
[cross_analysis/attempt001/summary.json](cross_analysis/attempt001/summary.json).
The unchanged reference selector, residual cache, and V31 adapter match the
previous study byte-for-byte; see
[reference_source_comparison.json](reference_source_comparison.json).

## Swaps: implementation matches, seed-dependent behavior remains

All16 reference/device pairs match in full output hash, N and official score.
For each question/seed, candidate output also matches across all four engines
and both instrumentation passes. The optimized selector therefore does not
explain the trajectory reversal on these particular cases.

| Pass | Panel seed | V31 mean N | Swap mean N | Difference | V31 correct | Swap correct |
|---|---:|---:|---:|---:|---:|---:|
| Clean, two engines | 1 | 280.000 | 176.000 | −104.000 | 1/2 | 1/2 |
| Clean, two engines | 42 | 224.125 | 233.000 | +8.875 | 0/2 | 1/2 |
| Audit, two engines | 1 | 280.000 | 176.000 | −104.000 | 1/2 | 1/2 |
| Audit, two engines | 42 | 206.500 | 233.000 | +26.500 | 0/2 | 1/2 |

Correct counts refer to the two distinct inputs; repeated trials do not add
independent questions. On question0, swaps change N218→209 at seed1, both
correct; at seed42 they change an ordinary V31 N100→192, turning an incorrect
answer into a correct one. On question49, N342→143 at seed1 and313→274 at
seed42, with both methods incorrect. This supports an interaction between
seed and method, including a quality/call tradeoff, rather than a consistent
convergence improvement. It cannot explain the remaining501 questions.

Execution variation is also material. One clean question0/seed42 V31 pair
changes N100→241, both wrong, while the other seven bookends for that cell are
N100. The control mean is retained unchanged, which explains the clean/audit
control difference in the table. No output-parity filtering is applied.

Even the favorable clean seed1 pair loses request latency: optimized swaps
use37.14% fewer calls but W10.312s versus9.366s V31 (+10.10%). The reference is
slower again at11.175s. These are diagnostic two-question timings; they do not
replace the64-question clean performance table. They directly illustrate that
fewer calls alone are insufficient at the observed observation cost.

## Residual: no stable benefit and two unresolved execution discrepancies

Fourteen of16 reference/device pairs match. Both discrepancies occur on
question0/seed1 in audit engines: reference/device N321/310 in the first,
and274/321 in the second. All four candidate answers are incorrect. Both
clean engines instead match at321. This is an execution-associated trajectory
discrepancy with no consistent directional bias, not evidence that one backend
is more accurate. It is still insufficient to claim full trajectory equivalence.

On the two clean seed1 cases, residual averages N336.5 versus V31280, with
equal1/2 accuracy and W11.864 versus9.366s. On seed42 it averages N187.5 versus
224.125, but both methods are wrong on both inputs. W8.102 versus8.119s is
essentially flat and influenced by the long V31 bookend. Audit seed42 residual
is slower (8.411 versus7.958s). None is a reproducible joint quality/latency gain.

The one authorized trace engine is redirected to the observed residual
question0/seed1 discrepancy, as frozen in TRACE_GATE_ADDENDUM.md. It preserves
V31 bookends, uses the same four explicit warmups, and compares reference/device
guarded residual. It is an exploratory localization, not another accuracy
screen. No full503 job is launched.
