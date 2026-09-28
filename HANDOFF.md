# v24 continuation (close bootstrap measurement)

Authority: user v24 handoff `dllm_claude_v24_close_bootstrap_measurement_20260928.md`; started at `ca297e05`.
Worktree `E:/dlm/m3_output_numerics_20260927`, branch `research/m3-output-numerics-20260927`.
GPU window ended 2026-09-28T05:52Z; **no GPU activity until the user renews a deadline** (STATE.gpu_authorization).
Budget recorded: 12,032.3 / 21,600 GPU-process s, 300 / 512 executions (no v24 GPU use).

## Done (CPU)
- Strict union audit of the two v23 segments: PASS, 120/120 cells, lineage for both bindings, method
  sources byte-identical (`scripts/v24_bootstrap6_audit.py`, tests 8/8, `R/v23_bootstrap6/v24_audit.*`).
- Five-phase B0/BO/A/D/H conservation from raw receipts holds for every first run.
- Errata `R/v23_bootstrap6/v24_errata_and_audit.md`: W = N x W/N on the same cells (LB M3-boot/native
  0.990 = 1.040 x 0.952); W/N is amortized, not direct forward; preview 0.77 does not carry to full panel
  (0.9997); host = seed confounded.
- Direct-cost harness `scripts/v24_profile.py` (six frozen arms, N16 from canvas start, model_forward +
  denoising_step, per-call B0/BO/A/D/H prices); derived configs CPU-validated on both hosts.

## Next (needs authorization)
1. v24 direct-cost profile, one worker per host (~700 GPU-s + 3 captures each). Command in STATE.next.
2. Only if a phase is materially dominant: at most one exact execution change (none chosen yet).
3. RULER 13-task stage (90 executions) needs a small frozen protocol extension (1 seed, warm on 2 tasks).
