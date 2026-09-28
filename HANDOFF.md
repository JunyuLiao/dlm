# v25 continuation (aligned route storage) — complete

Authority: user v25 handoff; started at `9ded5f80`. GPUs: user standing authorization (no approval needed),
recorded in `E:/dlm/gpu_authorization.json`. Budget 17,655.5 / 21,600 GPU-process s, 510 / 512 executions.
No active jobs; both GPUs 0 MiB.

## Results (R = results/m3_numeric_trajectory_bridge_20260927/v23_bootstrap6)
- `R/v25_corrections.md`, `R/v25_qualification_contract.md` (frozen before measurement).
- `aligned16` qualified: 18/18 sequences logits bitwise equal and decisions identical; odd-K summary deltas
  z<=9.5e-7, mu<=5.7e-6 abs (`R/v25_aligned16_evidence.json`).
- Direct cost odd-K LB: BO 0.917, A 0.909, D ~0.96; sequence 0.977 (M3)/0.980 (B); aligned K +0.1%.
- Pilot `v25_aligned6_pilot_56d3b98fe1261756` 90/90: aligned vs logical M3 tokens identical on all 8
  first cells; request 1.001 (LB); quality equal across arms; T 0.869, M1 0.877, B 0.911, M3 ~0.99 vs native.
- Report `R/v25_route_storage_report.md`. Builder probe NOT run (ceiling 2-4%, 2 executions left).

## Next
User decision: write up the quality-time frontier, or a new frozen direction with a raised execution cap.
