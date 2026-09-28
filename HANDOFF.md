# v24 continuation (close bootstrap measurement) — complete for this window

Authority: user v24 handoff `dllm_claude_v24_close_bootstrap_measurement_20260928.md`; started at `ca297e05`.
GPU window renewed by the user at ~07:12Z until 09:30Z (STATE.gpu_authorization). No active jobs; GPUs 0 MiB.
Budget: 13,278.8 / 21,600 GPU-process s, 408 / 512 executions (v24: 1,246.5 s, 108 executions).

## Results (R = results/m3_numeric_trajectory_bridge_20260927/v23_bootstrap6)
- CPU: strict union audit PASS (`R/v24_audit.*`), five-phase conservation, errata (`R/v24_errata_and_audit.md`).
- Direct cost (`R/v24_direct_cost.md`, `R/direct_cost_extract.json`): LB N16 model_forward vs native
  M3-boot 0.988/0.941, B-boot 0.965/0.919, incumbent 0.976/0.930, T 0.991/0.977; AIME/RULER 1.009-1.037.
  Per-call: BO +19-31% (LB), A +5-16%, H -8-12%, B0 ~1.00.
- Observation breakdown (`R/observation_probe_extract.json`): route kernel dominates; odd K ~2.6x slower/key.
- Candidate `aligned_route_k16`: route 6.6->3.1 ms at odd K, bitmaps identical but summary z/mu and
  padded producer differ in last bits -> numerical variant, NOT applied.
- RULER stage `v21_bootstrap6_ruler_4dbecd0c10841a11`: 90/90, every arm 12/13 (`R/ruler_scored.*`).
- LB expansion not run (no plausible M3-specific E2E gain).

## Next (user decision)
(a) qualify `aligned_route_k16` as a new numerical variant with token-level paired outputs, or (b) stop/write up.
Helpers: E:/dlm/v23_transport.py, v23_score_transport.py, v24_launch_profile.py, v24_launch_components.py,
v24_launch_alignment.py (private coordinator).
