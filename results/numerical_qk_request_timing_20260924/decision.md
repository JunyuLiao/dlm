# v9 decision

## What is correct
- The v8 prefix-summary selector is decision-preserving at request level: on /2 and /8, S and L produce
  identical completion tokens, per-canvas decoder calls and termination in attempt 0 and in all 4
  warm repeats (v9), and match the v7/v8 receipts (`v8_smoke_receipt_verification.json`).
- Measurement path qualified (`measurement_contract_and_dispatch.md`). The native baseline is unbound
  (spy: 30/30 native SDPA). v8's "native_dense" row was dense_eager and has been relabeled. Replays restore
  RNG, sampler, stopping criterion and T state before every rep, and inputs and outputs are bit-identical
  across reps (`replay_state_tests.json`).

## What accuracy was actually tested
Attempt 0 on aime26/2 and /8 only: D 2/2, L 2/2, S 2/2 (frozen final-channel scorer, EOS, none capped).
This is a 2-question subset chosen in v7. It is not a noninferiority or broad quality claim. v7's 3/4
four-question result is historical.

## Did clean E2E improve? No.
Warm, same process, alternating order, 2 repeats per cell (`clean_request_timing.json`):

| | D native | L legacy | S summary | L/S (>1 = S faster) |
|---|---:|---:|---:|---:|
| /2 wall (s) | 22.91 | 19.53 | 19.59 | 0.997 |
| /8 wall (s) | 20.36 | 33.72 | 33.98 | 0.992 |
| ms per decoder call | 154-157 | 186-192 | 188-192 | |

- **S vs L (the algorithm-preserving comparison): no gain; S is 0.3-0.8% slower.** The −29% per-call
  LOCAL selector gain does not reach the request. The step is launch-bound, and total `_route` time grows
  with key length (8.5 -> 18.1 ms/step, canvas 1 -> 8). It was not split by layer, so which layers
  dominate is not established.
- **L/S vs D: about 20-23% slower per decoder call.** Request totals differ because the trajectories differ.
  On /2, L/S use 102 calls vs D's 146, so the L/S request is 15% shorter. On /8 they use 181 vs 132, so it
  is 1.66x longer. The call-count differences come from different support (legacy_junyu_mask crop +
  sparse support vs native mask) and are not an execution speedup. No net E2E improvement over a strong
  compatible dense baseline is demonstrated.
- Cold (first request in a process, or a new prompt length): L/S pay +4 to +48 s of in-request Triton
  compilation. D pays none.

## Largest verified cost
In-request Triton re-specialization: `K`/`KT`/`PREFIX_TILES` are `tl.constexpr`, and the global key length grows
every canvas. This cost +44 s on L /2 and +48 s on S /8 (attempt 0 vs warm), with 39-61 new compiled
kernels. It explains the old "430 ms/forward vs 206 ms replay" discrepancy. In warm steady state, the
largest measured item is the launch-bound decoder step: the MoE expert loop accounts for ~5k of ~10.7k
launches per step and is shared with native. The selector adds ~14-27 ms of kernel time per step
(mostly `_route`) and ~1.4k launches. The native-vs-sparse split is not traced (see
`request_trace_accounting.md`, unassigned).

## No remedy implemented this round (deliberately)
The dominant verified cost (constexpr re-specialization) is removable only by making `K`/`KT`/`PREFIX_TILES`
runtime arguments in three Triton kernels. That keeps the arithmetic order, but it is a kernel-signature change.
It must pass the bit-exact CUDA suites and a request-level token identity check, and the 18-execution request
budget for this round is spent. Implementing it without that paired check would repeat v8's mistake.

## Single next decision
**Remove shape re-specialization from `_route`/`_pv`/`_preqk_pv` (runtime K/KT/PREFIX_TILES, or pow2-bucketed
KT with masking), then rerun the same bounded D/L/S protocol on /2 and /8 with a FRESH disk cache for both arms.**
Acceptance: bit-identical selector tests, L==S==v9 tokens, and cold attempt-0 wall within ~1 s of warm.
If warm per-call remains ~20% above native after that, the method has no efficiency path on this backend
without reducing launches (e.g. fusing the per-layer selector glue). That should be settled before any
accuracy or multimodel expansion.
