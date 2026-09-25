# v10 morning brief (2026-09-25)
- Final SHA: see `git log -1` on `research/numerical-qk-reuse-native-20260924` (this brief is in the final commit).
  No active jobs: GPU idle, and all owned processes have exited.
- Used: 56/72 complete requests, 1.14/5 GPU-h, ~1.6/8 h elapsed. Root has 56 GB free; the large disk has 967 GB free.
- Data: aime26/2, /8, /14, /20, seed 42, H100. Arms: D native, T fresh Junyu, P = generic M1 + summary, O = P + repair.
- CP1 (done): length-generic Triton kernels, BIT-IDENTICAL to static; the first mismatch (K%4 alignment) was fixed
  with a K/KDIV constexpr. Variants are bounded at 65 (static: 206 and growing). Cold /2 94.7 -> 32.5 s; with the
  bounded warmup, panel requests had 0 compiles.
- CP2 (done): one repair = minimal telemetry + fused guard (same coverage). -3 ms/step (~2%) same-state;
  -0.8% request (O/P geometric 0.992, consistent on all 4). Bit-identical outputs and tokens.
- CP3 (done, 48 runs, 0 failures; all 32 repeats token-identical to attempt 0): quality 3/4 in every arm.
  D/P/O cap on /14; T is wrong on /20.
- NEGATIVE: O is slower than native (geometric 1.08, summed 1.23; +20-26% per call) and slower than fresh T
  (geometric 1.09, summed 1.31). No E2E gain.
- Counterfactual: even with routing removed (same support), the step is ~11% slower than native. Fresh T's full
  fused path is only +7-9% per call.
- Largest added device cost: the `_route` kernel (32 CTAs, sequential tiles; LOCAL layers dominate).
- The native-mask vs legacy-mask discrepancy is still uncontrolled.
- Next decision: evaluate historical selection INSIDE Junyu's fused Hopper kernel (bitmap in, same mask) vs fresh
  T; stop tuning the Triton M1 path.
- Files: `results/numerical_qk_overnight_runtime_20260925/` (decision.md first).
