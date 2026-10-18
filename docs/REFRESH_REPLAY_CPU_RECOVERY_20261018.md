# Bounded CPU-comparison recovery

The first correctly bootstrapped replay produced two complete snapshot reports
with no variation, but CPU error reductions dominated execution. Stop that owned
process, preserve its partial reports and charge all elapsed reservation time.
Do not resume or overwrite it. The second engine was not launched.

In replay_corrected/attempt002, keep the same native GPU code, captured values,
layouts, auto splitting, warmups and repetitions. Replace only CPU analysis:
return zero error metrics immediately after equal shape/dtype/exact byte hashes;
otherwise call the original metric implementation. When every partial slot is
active, flatten the partial output instead of boolean-indexing all its elements.
Retain the identical active-mask representation. Tests require exact agreement
with old metrics, including NaNs, signed zero and actual mismatches. Set CPU
diagnostic torch threads to one and record this explicitly. These runs do not
measure operator speed or model latency.

Replacement engine1 uses at most300 seconds from the remaining original replay
allocation. Replacement engine2 uses the preallocated300-second recovery phase.
This is an explicit repair allocation, not automatic phase borrowing; the total
campaign limit remains2700 seconds. If either replacement remains incomplete,
report the numerical gate unresolved and stop GPU work. Runtime attribution and
its completed clean controls remain independently usable.
