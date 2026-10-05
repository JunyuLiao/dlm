# v11 decision: preselected-support Hopper consumer -> H1 / H3

Scope: aime26/2, /8, /14, /20 (4 questions, seed 42), native adaptive, thinking ON, 8192, EOS, frozen
final-channel scorer, H100. 39 complete-request executions (2 wiring, 1 wiring receipt lost to a serialization
bug, 36 panel) of the 64 cap. 0.56 of 4 GPU-hours.

## 1. Is the consumer correct and actually preselected? Yes.
- `support_consumer.cu` (ABI 1, `vd_support_v1`) tests the read-only support BEFORE any K load, QK, V load or PV
  of a tile. Evidence: executed-path counters; NaN poisoning of dropped K/V gives a bit-identical output; time is
  linear in kept tiles (GLOBAL K=2669: 0.52 ms all kept -> 0.05 ms none).
- **Bit-identical to fresh T** when fed fresh T's own support. memcheck/racecheck/synccheck are clean after
  fixing an aligned `bar.sync` shared by two roles.
- Numerics vs the frozen Triton (O) consumer on 660 real calls: relative Frobenius <= 0.0025, but the max-abs
  error vs FP32 exceeded the pre-fixed envelope (1.5x Triton) on 60 calls (median ratio 1.32, max 2.08). The
  cause is fresh T's P convention. The tolerance was not loosened: **H1 is numerically different from O**.

## 2. Same-input execution (`same_support_execution.*`)
Direct complete calls, median ms (canvas 1 / canvas 6):
| | LOCAL c1 | GLOBAL c1 | LOCAL c6 | GLOBAL c6 |
|---|---:|---:|---:|---:|
| native SDPA | 0.070 | 0.335 | 0.075 | 0.738 |
| fresh T complete | 0.381 | 0.444 | 0.517 | 0.792 |
| O complete (selector + Triton consumer) | 0.773 | 0.938 | 0.934 | 1.836 |
| H1 complete (selector + Hopper consumer) | 0.801 | 0.922 | 0.989 | 1.827 |
| consumer only: Triton / Hopper | 0.146 / 0.128 | 0.219 / 0.164 | 0.173 / 0.173 | 0.433 / 0.385 |

The consumer is 0-25% faster, but it is only ~15-25% of the M1 call. The existing selector path (route_only,
V sketch, glue) costs 0.6-1.4 ms, 2-3x fresh T's ENTIRE call. On LOCAL layers even the consumer alone costs
about twice native SDPA's full attention.

Full step (per-rep restored replay, both arm orders): native 124/131 ms, genuine fresh-T step 141/149 ms,
O = H1 within +-2 ms (ordinary ~152-165, held-same-support ~138-149). **There is no material step-level gain
from the new consumer.**

## 3. Complete answers (`complete_request_results.*`; attempt 0 = quality; one warm repeat, all token-identical)
| | Dn native | T fresh | H1 | H3 (R=2) |
|---|---|---|---|---|
| correct | 3/4 (/14 cap) | 3/4 (/20 wrong) | **2/4** (/2 wrong; /14 EOS, wrong) | 3/4 (/14 cap) |
| warm s /2,/8,/14,/20 | 22.6, 20.1, 64.6, 55.1 | 34.6, 32.4, 63.7, 21.5 | 17.8, 31.4, 64.1, 22.5 | 30.9, 28.2, 110.0, 36.1 |
| decoder calls | 146,132,438,370 | 208,197,401,133 | 97,177,361,125 | 181,167,637,212 |
| amortized ms/call | 147-155 | 159-167 | 177-183 | 169-173 |

Named ratios:
- H1/T: geometric 0.851, summed 0.892 = 0.809 (call count) x 1.102 (amortized per-call cost).
- H1/Dn: geometric 0.840, summed 0.836, with per-call +19.5%.
- H3/T: geometric 1.225, summed 1.348 = 1.275 x 1.057.
- H3/Dn: geometric 1.210.
- T/Dn: geometric 0.988.

**Interpretation (section 8 of the spec):**
- H1 is "faster" than T and native only through shorter trajectories, and it lost two answers. That is a failed
  quality-speed tradeoff, not an efficiency result.
- H3 keeps 3/4 but is slower than T (longer trajectories; per call only ~6% above T).
- Neither method works on the qualified faster consumer.

D_mask controls (dense with the legacy legal mask, diagnostic only): 3/4 correct, calls 135/99/587/141. The LOCAL
mask convention alone strongly changes trajectory length vs native (146/132/438/370), so the Dn comparison
remains confounded.

## 4. What is still missing
Same-mask speed baseline quality at scale; more questions and seeds; a history vs simple prior-bitmap ablation;
per-layer selector attribution in natural requests; a TMA variant of the consumer. Haowei's grouping work is NOT
integrated. M2 was not attempted.

## 5. One next decision (measured requirement for Junyu/Haowei)
The missing margin is the SELECTOR, not the consumer: 0.6-1.4 ms per call against fresh T's 0.38-0.79 ms whole
call. Requirement:
- **Move the historical decision into the Hopper producer warpgroup.** It reads the anchor's per-tile summaries
  (block log-mass, mu[32], active/bad; already produced exactly by the LOCAL summary path) plus live T/ref, and
  decides tile by tile in the same pass that computes retained current QK/PV. No route_only, sketch or glue
  launches.
- **Target:** a complete ordinary call <= fresh T's call on the same inputs.
- **Evaluate on GLOBAL layers first.** There native attention grows with context (0.34 -> 0.74 ms per call
  measured) and the Hopper consumer is already ~2x cheaper. LOCAL layers at window 1024 offer no measured margin:
  native 0.07 ms vs consumer-only 0.13-0.17 ms.
