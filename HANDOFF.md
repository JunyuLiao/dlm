# v23 continuation (cheap anchor + native bootstrap) — COMPLETE for this window

Authority: user v23 handoff `dllm_claude_v23_cheap_anchor_native_bootstrap_20260928.md`; started at `6b795ca6`.
Worktree `E:/dlm/m3_output_numerics_20260927`, branch `research/m3-output-numerics-20260927`.
GPU window closed 2026-09-28T05:52Z. Charged 12,032.3 / 21,600 GPU-process s, 300 / 512 executions
(see STATE.json). No active jobs; both GPUs 0 MiB.

## Results (R = results/m3_numeric_trajectory_bridge_20260927)
- `R/v23_corrections.md`: six v22 claims narrowed.
- Track E `R/anchor_memory_cost.md`: grouped-Q observation producer bitwise identical on 23 states; applied to all
  method arms; ~0.1% of request time.
- Track P `native_bootstrap2_observe1`: qualified (native logits at calls 0/1, exact A/D/H) on both hosts.
- Panel `v21_bootstrap6_b8b2c59449fd0ebf` 240/240, report `R/v23_bootstrap6/bootstrap6_report.md`:
  LB M3-bootstrap 6/12 correct, 2415 calls vs native 2360, paired warm 0.990 [0.94,1.05]; incumbent M3
  2604 calls, 5/12; B-bootstrap 4/12; fresh T 0.904. AIME: no speed gain.
- LB blocks 4-11 ran under a second binding (deploy dc52522, run dir bootstrap6_001_lbrest) because the
  prefix-resume rule refused stage order preview->aime->lb_rest; method sources byte-identical.

## Not done
RULER stage; direct forward/step pricing of bootstrap arms. Next action needs a user decision and a GPU
window extension (morning_brief_zh.md).

## Resume helpers (private, coordinator)
E:/dlm/v23_transport.py (bind/launch, --run-dir), E:/dlm/v23_score_transport.py (closed-ledger scoring),
scripts/v23_bootstrap6_reduce.py (report).
