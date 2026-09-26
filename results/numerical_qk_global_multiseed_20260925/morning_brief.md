# v13 morning brief (2026-09-26)
- Final SHA: `git log -1` on `research/numerical-qk-reuse-native-20260924`. No owned GPU jobs; GPU idle.
- Scope: frozen GLOBAL-only D_native / T_G / G1 / G3 / B8_G; all 30 AIME26 x seeds 17 and 29 (user-authorized);
  attempt 0 + one warm repeat.
- Result: 600/600 executions, 0 failures, all 300 warm repeats accepted. 602 executions total incl. checks; 7.71 of 12 GPU-h.
- Seed-safe driver: seed in the cell ID/request/receipts; resume validation; 9 tests; the seed-42 check reproduced the v12 hash.
- Quality (of 60): T_G 34, G3 34, B8 34, D 32, G1 30. Caps: 23 / 25 / 20 / 24 / 26. All paired quality CIs include 0.
- **The G3 speed signal reversed:**
  - G3/T_G time ratio 1.160 [1.071, 1.259]; robust across seeds, strata and leave-one-out;
  - G3/D_native 1.091 [1.011, 1.184];
  - the cause is 12% more denoising calls (more canvases and more steps per canvas); per call ~1.003x.
- B8/D_native summed 0.927 [0.856, 0.997], geometric 0.949 [0.851, 1.052]. T_G/D_native geometric 0.941 [0.868, 1.019].
- M1 re-selection shows no benefit over the frozen anchor bitmap (G1/B8 time 1.109; 30 vs 34 correct).
- Files: `results/numerical_qk_global_multiseed_20260925/` (fan_update.md, paired_summary.md).
- Next: no further kernel work on this path. Decide between a different model/workload regime and a
  follow-up on B8's small fewer-calls effect.
