# v23 continuation (cheap anchor + native bootstrap)

Authority: user v23 handoff `dllm_claude_v23_cheap_anchor_native_bootstrap_20260928.md`; starts from `6b795ca6`.
Worktree `E:/dlm/m3_output_numerics_20260927`, branch `research/m3-output-numerics-20260927`.
Limits unchanged: GPU stop 2026-09-28T05:52Z, final 06:22Z, 21,600 GPU-process s, 512 executions.

## Done in v23
- `R/v23_corrections.md`: narrows six v22 claims (zero-attention oracle is not a ceiling, AIME share scope,
  interpolated Q16 saving, all-kept attribution, no sample-size guarantee, residual is not all MoE).
- Track E (`R/anchor_memory_cost.md`): grouped-Q observation producer is bitwise identical to
  repeat_interleave on 23 real GLOBAL states (scores + bitmaps); LB producer 0.83x, -210..270 MiB peak.
  Accepted and applied to every method arm of the v23 panel; effect on requests is ~0.1% (A-frequency).
- Track P: `native_bootstrap2_observe1` (call0 native; call1 native + charged observation/selection;
  score origin 1 -> A at 9, R3 D at 4,7,12). Qualified on 6 real states x M1/M3/B on both hosts:
  native logits equal at calls 0 and 1, exact per-call A/D/H schedule (stage `v23_boot_6af4d42`).
- Frozen panel `v21_bootstrap6_b8b2c59449fd0ebf` (private `E:/dlm/v23_private/bootstrap6_001`):
  D_native, T_scope, M3_R3_A8_incumbent, M1/M3/B_native_bootstrap2_observe1; LB 6 q x 2 seeds and
  AIME first 4 x 2 seeds, first+warm (240); stages lb_preview (48), aime (96), lb_rest (96). RULER deferred.

## Running / next
See `STATE.json` gpu_jobs. Collect+score: `python E:/dlm/v23_score_transport.py --tag v23_boot_6af4d42 --label <label>`.
Next stage: `python E:/dlm/v23_transport.py --tag v23_boot_6af4d42 --action launch --stage aime --max-seconds 3000 --remaining-requests 48`.
