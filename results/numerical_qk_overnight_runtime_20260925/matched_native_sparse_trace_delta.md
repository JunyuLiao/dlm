# v10 CP2: matched native vs sparse trace (with explicit attribution limits)

Deployed `947cf49`. Arms: D = unbound native SDPA, no T. P = M1 preqk with LOCAL prefix summary, generic kernels,
full telemetry, separate guards (the qualified parent). aime26/2, torch.profiler windows of 2 steps
(1 anchor + 1 ordinary). Early = canvas 1 (absolute 365); late = canvas 8 (absolute 2157). Four windows total.
Machine-readable: `matched_native_sparse_trace_delta.json`; raw Chrome traces are on the large disk.

**Boundary:** D and P each follow their own natural trajectory. Absolute positions match, but the canvases
differ, so MoE expert routes and counts differ (D has more MoE kernels in one window, P in the other) and
cannot be credited to the selector.

## Per 2-step window (profiled; device times are device-measured)
| | D early | P early | D late | P late |
|---|---:|---:|---:|---:|
| window span ms (profiled, ~2x unprofiled) | 476 | 621 | 476 | 601 |
| device-active union ms | 105.2 | 136.2 | 115.6 | 163.0 |
| kernels / memcpy / memset | 18006/329/39 | 20845/449/109 | 18520/329/39 | 20570/449/109 |
| launch API calls (host ms, profiled) | 18006 (81.2) | 20845 (102.1) | 18520 (81.0) | 20570 (97.2) |
| explicit sync API calls | 131 | 131 | 131 | 131 |
| cudaMalloc/cudaFree | 0 | 0 | 0 | 0 |
| attention-layer device union LOCAL / GLOBAL ms | 10.2 / 4.5 | 32.0 / 8.1 | 11.5 / 11.1 | 47.3 / 20.9 |

## Where P's additions are (late window, 2 steps)
| label | events | device ms |
|---|---:|---:|
| `_route` in anchor route+PV (`sel_route_fused`) | 90 | 25.6 |
| `_route` ordinary (`sel_route`) | 30 | 15.1 |
| anchor score materialization | 415 (+30 memcpy) | 5.0 |
| selector glue (`sel_call`: guards, telemetry reductions, copies) | 1200 (+60 memset) | 4.0 |
| preqk current-output consumer | 60 | 3.6 |
| V-sketch lease | 900 (+90 memcpy) | 2.9 |
| T bookkeeping (begin + observe_logits) | 95 (+10 memset) | 3.2 |
Native attention modules become cheaper in P (GLOBAL 11.1 -> 2.3 ms; the attention core is replaced).

## Unprofiled same-state step replay (`v9_step_replay_profile`, median ms)
| | canvas 1 | canvas 6 |
|---|---:|---:|
| native step (step-1 inputs) | 122.5 | 133.3 |
| P ordinary / anchor | 153.6 / 162.4 | 166.9 / 168.9 |
| O ordinary / anchor (CP2 repair) | 150.6 / 160.2 | 163.7 / 166.1 |
Phase weights come from actual per-canvas calls. Anchor fraction is 18.6% on /2 (19/102) and 17.1% on /8
(31/181), not an assumed 1/8.

## Reading
- The **largest measured added device cost is the `_route` selector kernel.** It runs 32 CTAs (Q-tile x
  head) with a sequential scan over KV tiles, and its time grows with key length. LOCAL layers dominate the
  added device time.
- Submissions: about +1.0-1.4k device events per step, of which ~20 per layer call is glue. In unprofiled replay
  the step is ~30 ms slower than native at both states. The profiled windows cannot give an exact split
  between host submission and device time (the profiler roughly doubles the span).

## Unassigned / not claimed
- An exact unprofiled utilization split, and MoE differences between arms.
- The earlier v9 label "1.4k selector launches" counted consumer/anchor work that replaces dense work. The
  net added kernel count here is +1.0-1.4k per step.
