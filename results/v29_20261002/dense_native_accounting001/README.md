# Dense/native diagnostic accounting audit

Recomputed from the unchanged public raw002/003 artifacts. These are profiled singleton observations,
**not formal timing, a speed comparison, a causal overhead estimate or accuracy evidence.**
W includes prefill; S/N includes commit/sampler/scheduling amortization. It is not pure model-forward time.
Both campaigns used engine seed28001 and one preceding unprofiled warm request. Warm N/length and
the actual sampling RNG states are absent from these public records; they are unknown, not equal.

| Diagnostic | Arm | Requested graphs | W s | Prefill s | S s | N | C | Output tokens | S/N ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| cost32k_002 | dense | default | 5.052 | 0.900 | 4.152 | 137 | 10 | 2497 | 30.309 |
| cost32k_002 | native | PIECEWISE | 12.818 | 0.906 | 11.912 | 374 | 19 | 4761 | 31.849 |
| cost32k_002 | allkept | PIECEWISE | 5.785 | 0.909 | 4.875 | 150 | 11 | 2734 | 32.501 |
| cost32k_002 | method | PIECEWISE | 4.670 | 0.913 | 3.757 | 115 | 9 | 2205 | 32.672 |
| cost32k_events003 | dense | PIECEWISE | 12.114 | 0.883 | 11.231 | 374 | 19 | 4761 | 30.029 |
| cost32k_events003 | native | PIECEWISE | 4.147 | 0.914 | 3.233 | 99 | 9 | 2107 | 32.657 |
| cost32k_events003 | allkept | PIECEWISE | 5.823 | 0.910 | 4.913 | 150 | 11 | 2734 | 32.752 |
| cost32k_events003 | method | PIECEWISE | 4.737 | 0.913 | 3.823 | 115 | 8 | 2006 | 33.248 |

003 uses PIECEWISE for every arm. Its no-hook dense and native-hook differ in N and output length;
therefore graph configuration alone does not explain that discrepancy. Native also differs between002/003,
which have different source/profiler versions. Same seed and model math do not identify the first divergence.
No claim is made that hooks are correct merely because they delegate the same attention function.

## Observed attention scopes in003

| Arm | GLOBAL inclusive ms/N | LOCAL inclusive ms/N |
|---|---:|---:|
| dense | unknown phase | unknown phase |
| native | 4.494699 | 6.460856 |
| allkept | 4.894036 | 6.618996 |
| method | 4.329062 | 6.611120 |

Each denoising forward contains5GLOBAL and25LOCAL layers. These event sums are inclusive,
include dispatch gaps and instrumentation effects, and are from different generated trajectories.
Dense no-hook has only mixed prefill/commit/denoise attention ranges: dividing them by N would be wrong.
The complete event_scopes.csv preserves all recorded leaves; overlapping/nested/side-stream spans must
not be added to manufacture a wall-time breakdown.002 has no qualified event-span decomposition.

Next causal check: same-state native attention versus sparse/observer costs plus a fresh-engine
default/no-hook PIECEWISE/hooked PIECEWISE diagnostic with actual graph-mode receipts and sampling
state checks. Per-request RNG-reset/replay, if used, is separately labelled diagnostic and cannot
retroactively change the existing official-sampler formal campaign. No published run is overwritten.
