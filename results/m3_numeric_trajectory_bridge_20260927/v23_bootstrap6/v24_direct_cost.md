# v24 direct cost of the bootstrap arms, observation breakdown, and RULER coverage

All GPU work here ran inside the user-renewed window (granted ~07:12Z, GPU stop 09:30Z). Sources:
- Profile: deploy `3ae5e1d` (`scripts/v24_profile.py` over the v20 replay profiler).
- RULER: deploy `5cedc12`.
- Observation probes: deploys `d12b0be` and `526580d`.

Every stage's completeness and receipts were checked, and GPUs were confirmed at 0 MiB after each. Private raw outputs stay on the hosts; the scalar extracts are [direct_cost_extract.json](direct_cost_extract.json) and [observation_probe_extract.json](observation_probe_extract.json).

## 1. Direct complete-forward and native-step cost (fixed native states, N16 from canvas start)

Setup:
- Same captured native incoming states for all six arms.
- 1 warmup, then 3 blocks, each opened and closed by a native bracket (bracket drift ≤1.3%), with arm order rotated.
- No JIT during accepted timing; input/output/phase digests are identical across repetitions.
- `model_forward` = full `model.forward(...).logits`; `denoising_step` = native step including T/sampler/stop.
- Ratios are medians of the N16 sum of timed calls, relative to native.

This is a fixed-state timing diagnostic, not a scored or fixed-step method. The dllm RULER canvas reached only 3 calls, so it has no N16 cell (unavailable, not extended).

| Host | Task/state | Boundary | Calls | Native N16 ms | T | Incumbent M3 | M1 boot | M3 boot | B boot |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mpk | aime26 canvas0 | denoising_step | 11 | 1532 | 1.015 | 1.025 | 1.037 | 1.026 | 1.023 |
| mpk | aime26 canvas0 | model_forward | 11 | 1461 | 1.016 | 1.026 | 1.034 | 1.024 | 1.022 |
| mpk | longbench_v2 canvas0 | denoising_step | 14 | 2372 | 0.993 | 0.979 | 1.043 | 0.993 | 0.971 |
| mpk | longbench_v2 canvas0 | model_forward | 14 | 2289 | 0.991 | 0.976 | 1.043 | 0.988 | 0.965 |
| mpk | ruler4k canvas0 | denoising_step | 7 | 999 | 1.017 | 1.016 | 1.027 | 1.016 | 1.012 |
| mpk | ruler4k canvas0 | model_forward | 7 | 957 | 1.015 | 1.014 | 1.028 | 1.015 | 1.009 |
| dllm | aime26 canvas1 | denoising_step | 12 | 1566 | 1.009 | 1.017 | 1.026 | 1.016 | 1.012 |
| dllm | aime26 canvas1 | model_forward | 12 | 1475 | 1.019 | 1.015 | 1.025 | 1.024 | 1.022 |
| dllm | longbench_v2 canvas1 | denoising_step | 15 | 2471 | 0.983 | 0.934 | 0.980 | 0.946 | 0.927 |
| dllm | longbench_v2 canvas1 | model_forward | 15 | 2396 | 0.977 | 0.930 | 0.975 | 0.941 | 0.919 |

**Per-call cost vs native at the same call index (model_forward).** Phase from counter deltas: B0 = call 0 native, BO = call 1 native + observation, A = later score anchor (call 9), D = redecision, H = held.

| Host | Task/state | Arm | Per-call ratio to native at the same call, by phase (model_forward) |
|---|---|---|---|
| mpk | aime26 canvas0 | M3_R3_A8_incumbent | A 1.060 (n=2), H 1.013 (n=7), D 1.034 (n=2) |
| mpk | aime26 canvas0 | M1_native_bootstrap2_observe1 | B0 1.009 (n=1), BO 1.070 (n=1), D 1.031 (n=8), A 1.039 (n=1) |
| mpk | aime26 canvas0 | M3_native_bootstrap2_observe1 | B0 1.009 (n=1), BO 1.073 (n=1), H 1.012 (n=6), D 1.033 (n=2), A 1.044 (n=1) |
| mpk | aime26 canvas0 | B_native_bootstrap2_observe1 | B0 1.013 (n=1), BO 1.072 (n=1), H 1.015 (n=8), A 1.042 (n=1) |
| mpk | longbench_v2 canvas0 | M3_R3_A8_incumbent | A 1.188 (n=2), H 0.922 (n=9), D 1.014 (n=3) |
| mpk | longbench_v2 canvas0 | M1_native_bootstrap2_observe1 | B0 1.002 (n=1), BO 1.307 (n=1), D 1.019 (n=11), A 1.158 (n=1) |
| mpk | longbench_v2 canvas0 | M3_native_bootstrap2_observe1 | B0 1.003 (n=1), BO 1.307 (n=1), H 0.918 (n=8), D 1.014 (n=3), A 1.159 (n=1) |
| mpk | longbench_v2 canvas0 | B_native_bootstrap2_observe1 | B0 0.999 (n=1), BO 1.313 (n=1), H 0.916 (n=11), A 1.165 (n=1) |
| mpk | ruler4k canvas0 | M3_R3_A8_incumbent | A 1.073 (n=1), H 0.996 (n=4), D 1.023 (n=2) |
| mpk | ruler4k canvas0 | M1_native_bootstrap2_observe1 | B0 1.009 (n=1), BO 1.079 (n=1), D 1.020 (n=5) |
| mpk | ruler4k canvas0 | M3_native_bootstrap2_observe1 | B0 1.007 (n=1), BO 1.081 (n=1), H 0.999 (n=4), D 1.018 (n=1) |
| mpk | ruler4k canvas0 | B_native_bootstrap2_observe1 | B0 1.006 (n=1), BO 1.075 (n=1), H 0.999 (n=5) |
| dllm | aime26 canvas1 | M3_R3_A8_incumbent | A 1.051 (n=2), H 1.003 (n=7), D 1.023 (n=3) |
| dllm | aime26 canvas1 | M1_native_bootstrap2_observe1 | B0 1.002 (n=1), BO 1.060 (n=1), D 1.023 (n=9), A 1.031 (n=1) |
| dllm | aime26 canvas1 | M3_native_bootstrap2_observe1 | B0 1.012 (n=1), BO 1.068 (n=1), H 1.014 (n=7), D 1.034 (n=2), A 1.044 (n=1) |
| dllm | aime26 canvas1 | B_native_bootstrap2_observe1 | B0 1.016 (n=1), BO 1.071 (n=1), H 1.015 (n=9), A 1.045 (n=1) |
| dllm | longbench_v2 canvas1 | M3_R3_A8_incumbent | A 1.074 (n=2), H 0.893 (n=9), D 0.945 (n=4) |
| dllm | longbench_v2 canvas1 | M1_native_bootstrap2_observe1 | B0 0.999 (n=1), BO 1.216 (n=1), D 0.947 (n=12), A 1.071 (n=1) |
| dllm | longbench_v2 canvas1 | M3_native_bootstrap2_observe1 | B0 1.003 (n=1), BO 1.217 (n=1), H 0.892 (n=9), D 0.947 (n=3), A 1.072 (n=1) |
| dllm | longbench_v2 canvas1 | B_native_bootstrap2_observe1 | B0 0.996 (n=1), BO 1.194 (n=1), H 0.876 (n=12), A 1.050 (n=1) |

Reading:
- B0 is priced as native (0.996–1.016).
- **BO is the most expensive call: +19–31% of a native forward on LB, +6–8% on AIME/RULER.**
- The later A costs +5–16% on LB.
- H saves 8–12% per call on LB and nothing on short contexts; D is roughly neutral.
- Over whole N16 sequences on LB, bootstrap M3 is 0.988 / 0.941 of native (two states), matched B 0.965 / 0.919, incumbent 0.976 / 0.930, fresh T 0.991 / 0.977. On AIME and RULER every method is 1.5–3.7% slower than native.
- **Bootstrap M3 is not cheaper per forward than matched B or the incumbent.** Its N16 advantage over native on LB comes from H calls, net of BO and A.

## 2. Where observation time goes (isolated pieces, one GLOBAL layer, same Q/K/V)

| State | Keys (alignment) | Native attention | Score producer | Valid map | Full V projection proxy | Route + summary store | Retained FP32 consumer |
|---|---|---:|---:|---:|---:|---:|---:|
| mpk LB | 13,703 (odd) | 4.19–4.25 | 1.29–1.31 | 0.46 | 0.60–0.65 | **6.58–6.67** | 0.94–1.76 |
| dllm LB | 17,540 (÷4) | 5.23–5.27 | 1.17–1.18 | 0.56 | 0.52–0.57 | 3.89–3.91 | 0.98–2.10 |
| RULER | 4,084–4,216 | 1.27–1.31 | 0.31–0.33 | 0.16–0.17 | 0.47–0.56 | 1.00–1.04 | 0.38–0.71 |
| AIME | 365–608 | 0.24–0.30 | 0.14–0.18 | 0.07–0.08 | 0.47–0.50 | 0.22–0.28 | 0.18–0.23 |

All times are median ms. These are isolated CUDA-event pieces; they are not additive to a forward price.

**The route kernel is the dominant observation cost on LB.** On odd key counts it is ~2.6× slower per key than on ÷4 counts (0.49 vs 0.22 µs/key). The pieces reproduce the measured BO excess: ≈9 ms per layer on mpk, ≈6 ms per layer on dllm.

## 3. The one candidate execution change: 16-aligned route key extent — not exact, not applied

The probe pads cached scores (−inf) and projected V (0) to a multiple of 16. This leaves the tile count unchanged, and padded lanes are already loaded as −inf/0 today.

| Key alignment | Route + store ms, K → K16 | Bitmaps / eligible / invalid | Stored summary z, mu | Zero-padded producer |
|---|---|---|---|---|
| odd (mpk LB 13,703; AIME 365) | 6.6 → 3.1 (LB); 0.28 → 0.19 | identical | **differ (last-ulp; layout changes reduction order)** | **differs**, 1.29 → 0.81 ms |
| ÷4 / ÷8 / ÷16 (dllm LB, RULER, AIME 608) | unchanged | identical | identical | identical |

The candidate therefore changes stored numerics at odd K. Per the v24 contract it is a **named numerical variant (`aligned_route_k16`)**, not an execution-only improvement, and it was **not** folded into production.

The predicted request effect, if accepted later, is small. It removes ≈3.5 ms per GLOBAL layer on each BO/A call at odd K: about 2.4 observation calls per LB canvas × 5 layers ≈ 40 ms per ~2.8 s canvas, i.e. ~1–2% of odd-K LB canvases and less on average. It would help every arm that observes (incumbent, M1, M3 and B), not only M3.

Going ahead needs a decision: accept a new numerical variant and requalify it with token-level paired outputs, or leave it.

## 4. RULER coverage (frozen `v21_bootstrap6_ruler_4dbecd0c10841a11`, 90/90 executions, 0 failures)

13 task-balanced RULER-4K inputs × seed 101 × six arms of first runs, plus warm repeats on the first two tasks. Every arm scored **12/13** (official 13-task macro 0.923) with 46–55 decoder calls over 13 canvases. RULER requests are about 4 calls, so they carry no timing signal. Redacted scored files: `ruler_scored.*`.

## 5. Decision

- No exact execution improvement was found. The evidence-backed candidate changes numerics, so it is reported, not applied.
- The measured direct costs do not make an M3-specific repeatable E2E gain plausible: matched B is cheaper per forward, and bootstrap M3's request ratio remains geoN 1.04 × geoW/N 0.95. So the optional LB expansion was **not** run.
- Fresh T remains the fastest arm at the request point estimate.
