# HANDOFF — v14 CVM-T (current only; v10–v13 archived verbatim in docs/handoff_archive/)

## Identity
- Repo `coconight01/dlm_test`, branch `research/numerical-qk-reuse-native-20260924`, based on v13 `73b6a668`.
- Junyu reference `query-sensitivity-aware-v3@053441c6` was verified unchanged and never written.
- Host mpk (149.165.151.254), 1× H100 80GB. Env: conda `ljy_dlm` with the user site ON; torch 2.6.0+cu124 and triton 3.2.0 from `~/.local`, transformers 5.11.0.
- Model DiffusionGemma-26B-A4B `f7f5b7f5`. AIME26 dev IDs /2,/8,/14,/20,/23,/30 × seeds 17/29. Native adaptive schedule, thinking ON, 8192 output budget.
- Every GPU job ran from a read-only `../deploy/<sha12>/`: tests/diagnostics `82b77afb907c`, pilot `8c0c44f907f3`, post-pilot profiles `d19acb240fd3`.
- Private/bulky root: `/media/volume/dllm-1/dyh/numerical_qk_cvm_t_20260926/`.
  - Subdirs: `cp2/`, `cp3/` (ledgers, receipts), `post/`, `build/v5_be53e4c706032aef/`.
  - `resource_ledger.jsonl`, `job_ledger.jsonl`.

## Method (see results/numerical_qk_cvm_t_20260926/method_contract.md)
- **Anchor (new canvas / A8 / new epoch).** Private v5 of Junyu's fresh value-direction kernel (ABI 5, `csrc/value_direction_v5.*`, `torch_bridge_v5.cpp`).
  - It exports the unweighted `log rho[B,H,J,Q]` and enforces the mandatory tiles `j >= prefix//64` inside `decide()`.
  - Export/protect off is bit-identical to v4.
- **Ordinary step.** The Triton planner `experiments/value_direction_hopper/cvm.py:_plan` restores a tile iff `max_rows(log rho + log s_t) >= log tau` (add-only within an epoch). Then the v11 support consumer runs, then the fused guard.
- **Arms** (`experiments/numerical_qk_reuse/global_scope.py`), all GLOBAL-only with LOCAL native:
  - `global_TP` (T_P), `global_B8P` (B8_P), `global_CVM` (CVM_T);
  - `global_T` + `fast_t` (T_G_original).
- **Exact T-only fast path:** `query_adaptive.State(fast_t=True)` with a `t_history` sentinel.
- Tests: `tests/test_v14_cvm.py` (14 pass). v12/v13 regressions pass (12 passed, 3 skipped).

## Results (results/numerical_qk_cvm_t_20260926/)
- **Complete-forward cost** (`cost_budget.csv`, `complete_forward_profile.json`, `profiles/`):
  - AIME phase-weighted CVM_T c = 1.020 / 1.009 / 0.989 at prefixes 365 / 2157 / 6275. The request-level estimate is 1.003, and the pilot measured per-call 1.003.
  - LongBench-v2 16–18K regime (unscored): CVM_T 0.93–0.94, B8_P 0.90–0.92.
- **Adaptive diagnostic** (`adaptive_canvas_diagnostic.json`, 5 identical canvas-start states):
  - Calls to native stop: D 54, T_G 49, T_P 43, B8_P 43, CVM_T 46.
  - Executed prefix fraction: CVM ≈ T_P ≈ 0.70–0.75 (late), B8_P ≈ 0.52–0.55.
  - ~90% of divergent rows diverge while s = 1.
- **Pilot** (`complete_request_results.{csv,json}`): 120/120 executions, 0 failures, 60/60 warm accepted.
  - Correct out of 12: B8_P 9, T_G_original 8, D 6, CVM_T 6, T_P 5.
  - CVM_T/B8_P geometric time 1.23 [1.08, 1.50], from calls ×1.27 (per call 0.998). Paired outcomes 1 vs 4.
  - D_native and T_G_original tokens are identical to v13 in 12/12 cells each.
- **Launch inventory** (`launch_inventory.json`): on ordinary CVM steps, 0 v5/value-direction launches and 5 planner + 5 consumer launches; the anchor steps launch v5.
- **Overhead attribution** (`overhead_attribution.json`): the binding plus the fast-T controller cost ~0. Profiles drift ~2% within a process, so AIME forward differences ≤2% are unresolved. Use the pilot per-call factors (CVM_T 1.003, B8_P 1.005, T_P 1.013, T_G 1.018).
- **Sanitizer** on v5 export+protect: memcheck 0 errors, racecheck 0 hazards, synccheck 0 errors (`sanitizer_receipt.txt`).
- **Gates** (`gate_decisions.md`, recorded before scores): extension NOT run; no repair (no supported diagnosis).

## Reports
`morning_brief.md` (answers Q1–Q5), `fan_update.md`, `qualification.json`, `frozen_protocol*.json`, `smoke_protocol.json`, `private_receipt_index.json`.

## Reproduce
- Profiles: `scripts/v14_forward_profile.py`. Cost tables: `scripts/v14_cost_budget.py`.
- Adaptive diagnostic: `scripts/v14_adaptive_canvas.py` + `scripts/v14_adaptive_summary.py`.
- Protocol: `scripts/v14_build_protocol.py`. Runs: `scripts/v13_seed_runs.py`. Scoring: `scripts/v14_summarize.py`.
- Supervisors: `$R/cp3/run_pilot.sh`, `$R/post/run_post2.sh`.

## Next (single action)
Stop CVM-T on AIME (measured negative: no request-level execution saving; no quality or time gain over B8_P).
If Fan/PI keep the runtime claim: run the SAME frozen five arms unchanged on a small scored LongBench-v2 10–20K panel with
`scripts/v14_build_protocol.py` (new mode) + `scripts/v13_seed_runs.py`. That is the only regime with measured per-forward headroom
(c 0.90–0.94). It needs PI approval because it makes a long-context benchmark primary.
