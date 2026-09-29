# v27 (M1/M2/M3 frontier, BRISK-guided) — Tier 1 done, Tier 2 dev panel running

Authority: user v27 doc + request to also cover the v26 A64 document. Standing GPU authorization.
Independent v26/v27 budget: 12 GPU-h, 1,800 executions (see STATE). Old ledgers are closed, not reset.

## Done (see results/m1_m2_m3_frontier_v27_20260929/)
- **CP0:** effective A/R identity, hold-only B (B16/B64 without D9), R6, A64, named threshold shifts; clock and config tests (`v27_contract_audit.md`).
- **Compact M2** (`pooled_compact`): a shared per-KV-head tile pool with a row-legality guard (`m2_compact_contract.md`).
- **Tier 1 direct cost** (`direct_cost_report.md`, CSVs):
  - H ≈ 0.91 of native per forward on LB; the ceiling is set by the ~10% GLOBAL-attention share;
  - D costs +8–14 ms, A costs +25–40 ms;
  - D_matched (all-kept consumer) is 0.982, so about 2% of any gain is kernel efficiency;
  - AIME and RULER show no gain;
  - the threshold saving saturates beyond P0.
- **Composed 45-point screen:** B A64 < M3 R6/A64 < M3 R3/A64 < M3 R3/A8 < M2c < M1.
- **Parallel prefix-summary builder:** not built. With A64 its ceiling is ≤ 0.5%, which is below its own stop rule.

## Running
Tier 2 dev panels, frozen specs in `results/.../specs/`, runs `v23_private/v27_dev_{lb,aime}_001`, deploy `v27_dev_f66bfba`:
- LB: the six exposed ids × seeds 101/202, first + warm, 216 executions;
- AIME: four new ids × two seeds, first only, 72 executions; launched automatically after LB by `E:/dlm/v27_chain_dev.sh`.
- Arms: native, D_matched, fresh T, M1, compact M2, M3 R3/A8, M3 R3/A64, M3 R6/A64, B A64.

## Next
1. Score Tier 2 on mpk via `E:/dlm/v23_score_transport.py` (`--tag v27_dev_f66bfba --run-dir v27_dev_lb_001 --label dev_lb --worker-ends 1`).
2. Freeze Tier 3 (one primary M3 plus pre-declared secondary). Ids are reserved in `specs/tier3_reserved_ids.json`: only six new LB (short of 12; reported, not fabricated), eight new AIME, 13 RULER.
