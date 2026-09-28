# v26 (M1 / M2 / M3 frontier) — seven-arm panel reported; screen/expansion next

Authority: user v26 doc. GPUs: standing authorization. Independent v26 budget 12 GPU-h / 1800 executions
(used: see STATE.v26_budget); historical ledgers closed, not reset.

## Done
- CPU: current_facts, old_vs_new_contract, missing_measurements, clock_opportunity (validated 139/139).
- M2 (`mu_mode=pooled`) + A16 + `aligned16_odd` implemented and tested (`6c77cc47`, 111 tests on H100).
- Seven-arm panel `v26_seven_99cf56d6ef8412d7` 84/84: `results/m1_m2_m3_frontier_v26_20260928/seven_panel_report.md`,
  paired CSV, redacted generations; Chinese Fan draft `fan_update_zh.md`.

## Next
Policy screen on captured states (A{8,16} x R{1,3,6} x threshold P0+{-ln2,0,+ln2}, B, M2), then a frozen
expansion with more distinct questions x 2 seeds x 6 core arms. Attribution trace separately.
