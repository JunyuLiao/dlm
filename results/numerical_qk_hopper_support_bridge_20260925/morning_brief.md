# v11 brief (2026-09-25)
- Final SHA: `git log -1` on `research/numerical-qk-reuse-native-20260924`. No active jobs; GPU idle.
- Used: 39/64 complete requests (incl. 1 lost receipt), 0.56/4 GPU-h. Root has 54 GB free.
- CP1 (done): new preselected-support Hopper consumer (`vd_support_v1`, build e3c283b8cbb9b79c):
  - support is read before any K/QK/V/PV of a tile (NaN poisoning, counters, linear scaling);
  - bit-identical to fresh T on fresh T's own support; sanitizer clean after an aligned-barrier fix;
  - vs the O Triton consumer: max-abs FP32 envelope failed on 60/660 real calls (fresh-T P convention), so H1 is
    numerically different from O.
- CP2 (done): the consumer is 0-25% faster than Triton, but the complete M1 call H1 ~= O (the selector costs
  0.6-1.4 ms/call = 2-3x fresh T's whole call). No step gain. Genuine fresh-T full step is +14% vs native.
- CP3 (36 runs, 0 failures, repeats token-identical):
  - quality: Dn 3/4, T 3/4, H1 **2/4**, H3 3/4;
  - H1/T geometric 0.85 (summed 0.89) only via fewer calls (x0.81) with per-call x1.10 and two lost answers;
  - H3/T geometric 1.22 (summed 1.35).
  - No acceptable quality-speed result.
- D_mask controls: the mask convention alone changes trajectory lengths a lot (Dn vs Dm calls on /20: 370 vs 141).
- Also: fixed a v10 driver bug (panel O had separate guards) with correction notes appended to v10; fixed a
  support-parameter shadowing bug.
- Next: fuse the historical decision into the Hopper producer (anchor summaries + live T), GLOBAL layers first;
  target complete call <= fresh T.
- Files: `results/numerical_qk_hopper_support_bridge_20260925/` (decision.md first).
