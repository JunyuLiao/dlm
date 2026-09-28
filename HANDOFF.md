# v22 continuation (Claude Code takeover of the v21b campaign)

Authority: user v22 handoff `dllm_claude_v22_resume_4ab56f6d_20260927.md` (sole continuation spec).
Worktree `E:/dlm/m3_output_numerics_20260927`, branch `research/m3-output-numerics-20260927`.
Takeover verified remote `4ab56f6d27253baeae210724ea58d1bd6cb0f626` at 2026-09-28T00:10Z; peer refs
unchanged (Junyu `053441c6`, `ljy/value_aware@47c47d9d`, Haowei `b23f969a`). No Codex/Sol/Astra agents.

## Fixed campaign limits (unchanged, shared with v21/v21b)
GPU stop 2026-09-28T05:52Z; final report 06:22Z. 21,600 aggregate GPU-process seconds, 512
executions, at most one worker per host (mpk 149.165.151.254, dllm 149.165.159.64).
Current charge and reservations live only in `STATE.json`; outer receipts count a stage once.

## Completed (do not repeat)
v20 800 records audit; 1,047 tensor payloads per host; operator 19/19 per host; 17/18 numerical
states; exact `model_major` layout; four-mode factorial profile (22 cells); 12 natural zero-pruning
first-only diagnostics; 34 inherited-threshold geometry layer-states (all coarse parity matched).

## Current stage: CP-A (bootstrap fill + frozen two-state Q16 calibration)
v22 adds to `scripts/v21b_geometry_capture.py` a saved `completeness` verdict and exit code 3 when
any selected layer failed or a calibration screen is missing (exit 0 no longer implies complete),
plus tests for bootstrap `T=None` versus explicit ones in reference and capture paths.
Run from a fresh immutable deploy with a freshly bound config:
`python -m scripts.v21b_geometry_capture --config <bound diagnostic.json> --out <fresh> --bootstrap-calibration --qualify-tests`.
Combine the two LB call-3 receipts on CPU under `R/matched_work_calibration_contract.md`.

## Not done
No production fine-geometry kernel, no fine M1/M3 full-forward result, no new scored panel.
Old precision-only v21 panel is cancelled; at most one v22 geometry panel (CP-D) may run.

## Publication
Only explicit allowlisted source/tests/scalar reports; no prompts, gold, credentials or tensors.
