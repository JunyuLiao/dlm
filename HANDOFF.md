# v22 continuation (Claude Code takeover of the v21b campaign)

Authority: user v22 handoff `dllm_claude_v22_resume_4ab56f6d_20260927.md` (sole continuation spec).
Worktree `E:/dlm/m3_output_numerics_20260927`, branch `research/m3-output-numerics-20260927`.
Takeover verified remote `4ab56f6d` at 2026-09-28T00:10Z; peer refs unchanged. No Codex/Sol/Astra agents.

## Status: CP-A complete; fine-geometry promotion STOPPED (v22 section 6.3)
- Bootstrap: 6 call-0 states x 2 layers qualified (T=None -> ones), coarse parity 12/12.
- Frozen calibration: both rules select Q16 offset -1.0 = 1.158x coarse work, 0.918x pooled error;
  no Q16 point saves work inside the error band.
- Validation (20 other states, no reselection): LB equal-work error ~0.91x; equal-error saving ~4-5%
  of retained GLOBAL pairs (<0.5% of a forward).
- Attention-share ceiling (native forward): deleting all GLOBAL attention = 0.84x on LB (14-18K keys),
  0.97x RULER-4K, 0.98x AIME. Coarse M3 already realises 5-9% on LB.
- CP-B/C/D NOT run (conditional on a useful CP-A candidate). No new panel launched.
Reports: `results/m3_numeric_trajectory_bridge_20260927/q16_calibration.md`, `q16_validation.{json,csv}`, `q16_calibration.{json,csv}`,
`morning_brief_zh.md`, `fan_update_zh.md`, novelty addendum in `novelty_gap.md` (SparseD, PulseCol).

## Next single justified action (needs user decision; not started)
Either (a) a frozen, answer-free price experiment at longer contexts (16K/32K/64K keys) where GLOBAL
attention dominates, after a full-text novelty check against SparseD; or (b) reposition the work as a
measurement study of approximation error vs adaptive call count. Do not revive the 440 panel.

## Fixed campaign limits
GPU stop 2026-09-28T05:52Z; final 06:22Z; 21,600 GPU-process s; 512 executions; one worker per host.
Charge in `STATE.json`. Private receipts: E:/dlm/v22_private and host `diagnostics/*v22_*`.

## Publication
Only explicit allowlisted source/tests/scalar reports; no prompts, gold, credentials or tensors.
