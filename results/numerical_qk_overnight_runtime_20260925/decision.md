# v10 decision

Scope: aime26/2, /8, /14, /20 (4 independent questions), seed 42, native adaptive, thinking ON, 8192, EOS,
frozen final-channel scorer, H100. Arms:
- **D**: unbound native SDPA, no T. It uses the NATIVE mask, while T/P/O use the legacy Junyu local mask.
- **T**: fresh Junyu T through the verified extension (`value_direction_db080045f7a5fbce.so`).
- **P**: M1 preqk current output, LOCAL prefix summary, CP1 generic kernels.
- **O**: P plus the CP2 repair.

48 panel executions (16 attempt 0 + 32 timing repeats) plus 8 CP1 executions: 56 of 72. 1.14 of 5 GPU-hours.

## 1. Compilation fixed? Yes, for our kernels, and the fix is bit-exact.
Generic kernels are bit-identical to the static ones on every production entry point (35 tests). Tokens are
identical to static and v9. Variants are bounded (65 total, vs 206 and growing). A cold new process with an
empty cache: /2 94.7 -> 32.5 s, /8 first-seen 85.2 -> 38.6 s. With the bounded startup warmup (0.5 s from a
populated cache, 56.6 s from empty), every panel request had **0 Triton misses on attempt 0**.
This is engineering, not a sparse speedup.

## 2. Warm execution faster? Barely, versus the parent. No, versus native or fresh T.
| warm median s (amortized ms/call) | D | T | P | O |
|---|---:|---:|---:|---:|
| /2 | 22.8 (156) | 35.0 (168) | 19.5 (191) | 19.3 (190) |
| /8 | 20.3 (153) | 32.7 (166) | 33.5 (185) | 33.3 (184) |
| /14 (capped D/P/O) | 64.9 (148) | 64.1 (160) | 119.5 (189) | 118.5 (187) |
| /20 | 55.6 (150) | 21.7 (163) | 29.7 (185) | 29.5 (183) |

| paired warm ratios | geometric mean of per-question ratios | ratio of summed times |
|---|---:|---:|
| O/P | **0.992** (0.991-0.993 on all 4) | 0.992 |
| O/D | 1.078 | 1.226 |
| O/T | 1.091 | 1.307 |
| T/D | 0.988 | 0.938 |

Same-state step replay: O ordinary -3.0 ms/step vs P (~2%). Amortized per decoder call: O +20-26% vs native,
+11-17% vs fresh T. Request totals are dominated by trajectory length (calls: D 146/132/438/370, T 208/197/401/133,
P=O 102/181/633/161), which is a support/mask effect, not execution speed.

## 3. Complete answers preserved? Between P and O, exactly. Across arms, 3/4 each.
P and O are token-, call- and termination-identical on all 4 questions. All 32 repeats match their attempt 0.
Attempt 0: D 3/4, T 3/4, P 3/4, O 3/4.
- D, P and O fail /14 by hitting the 8192 cap.
- T solves /14 but is wrong on /20.
Four questions do not establish noninferiority either way.

## 4. E2E vs parent / fresh T / native
- vs parent: -0.8% (consistent, small).
- vs fresh T: slower (geometric 1.09, summed 1.31).
- vs native: slower (geometric 1.08, summed 1.23), with the mask discrepancy disclosed.
**No net E2E improvement over a strong compatible dense baseline, and none over fresh Junyu T.**

## 5. Largest verified costs, and the counterfactual
- Matched trace (D vs P, early/late). The largest added device cost is the `_route` selector kernel: late,
  15.1 ms for one ordinary step and 25.6 ms for anchor route+PV. It runs on 32 CTAs with a sequential tile scan,
  and LOCAL layers dominate. There are about +1.0-1.4k device events per step (the glue is ~20 kernels per layer
  call). Explicit syncs are equal and there are no allocator calls.
- Same-support consumer-only counterfactual (step 1 re-timed with its OWN decision held; T and the consumer
  kept; routing removed):
  - canvas 1: native 119.8 / O ordinary 147.5 / held 134.0 ms;
  - canvas 6: native 131.1 / 161.0 / 144.6 ms.
  **Even free routing leaves this execution path ~11% slower per step than native.** Fresh T's whole path
  (fresh QK + value-aware routing + PV, fused Hopper CUDA) is only +7-9% per call above native (request-amortized).
  So our Triton consumer path alone costs about as much as, or more than, Junyu's complete fused path.
  Caveat: the counterfactual is step-level on /2; the T numbers are request-amortized.

## 6. Missing evidence (not claimed)
- A same-mask dense control (native-mask vs legacy-mask quality/trajectory).
- Step-level T replay.
- Per-layer `_route` split beyond LOCAL/GLOBAL unions.
- More questions and seeds.
- The numerical-history vs simple-bitmap ablation.
- Haowei's grouping integration, M2, and other models.
Per-tile skip statistics for O are N/A (minimal telemetry); P, token-identical, carries them.

## 7. One next decision
**Stop optimizing the Triton M1 pre-QK execution path as the efficiency vehicle. Instead, test historical
selection inside Junyu's fused Hopper kernel.** Feed an M1/M3 bitmap in place of its per-step fresh routing,
with the same mask and the same kernel, and compare head-to-head with fresh T. The evidence behind this:
- on this backend, the consumer path alone (routing removed) is already slower per step than native;
- fresh T is cheaper per call than our full path.

If historical selection inside the fused kernel does not beat fresh T per call, the temporal-information
contribution must rest on quality/selection (vs a simple previous-bitmap baseline), not on runtime.
