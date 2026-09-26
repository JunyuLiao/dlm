# HANDOFF — v15 scored LongBench-v2 long-context diagnostic (current only; older rounds in docs/handoff_archive/)

## Identity
- Repo `coconight01/dlm_test`, branch `research/numerical-qk-reuse-native-20260924`, reviewed start `fa92a1e`.
- Junyu ref `query-sensitivity-aware-v3@053441c6` was verified and never written.
- Host mpk, 1× H100 80GB. Env: conda `ljy_dlm` with the user site ON; torch 2.6.0+cu124 and triton 3.2.0 from `~/.local`, transformers 5.11.0.
- Model DiffusionGemma-26B-A4B `f7f5b7f5`; the text context is 262144 positions, and peak memory in this panel was 58.4 GiB.
- Deploys (read-only): tests/setup `5f3e52c0decc`, panel `af1fc0d5f728`.
- Private root: `/media/volume/dllm-1/dyh/numerical_qk_longcontext_scored_20260926/`.
  - `private/`: generation manifest (no gold) and `gold_scorer_only.json`.
  - `setup/`, `panel/ledger.jsonl`, `private_panel/` receipts, `attribution/`.
  - `resource_ledger.jsonl`, `job_ledger.jsonl`.

## Contract (results/numerical_qk_longcontext_scored_20260926/)
- **Panel** (`panel_selection.json`):
  - Eligible = untruncated pinned NeMo `eval/longbench/default` prompt with the project's reserved-token escaping, 10–20K request tokens (thinking ON template). That gives 49 of 503 items, in 4 domains.
  - Selection: domain × bin round robin, seed 20260926; the two v14 profile items were excluded.
  - Result: 12 questions, 6 per bin; 3 have prior project exposure.
- **Protocol** (`frozen_protocol.json`, id `v15_lbv2_scored_28f30b0d703d`): the five v14 arms copied verbatim, the v13 schedule (seed 20260929), 240 executions.
- **Scoring** (`scripts/v15_longbench_task.py`): NeMo `eval_mcq` default on the final channel only. Task score counts a valid final choice even at a cap; strict score also requires EOS.
- **Code:**
  - driver `scripts/v15_seed_runs.py` (v13 plus gold-free manifest check, new-.so detection, deadline stop);
  - summarizer `scripts/v15_summarize.py` (strict ledger, exact pooled decomposition);
  - `scripts/v15_phase_accounting.py`;
  - tests `tests/test_v15_longbench.py` (11 CPU). 34 tests passed on the deploy.

## Results (see morning_brief.md)
- 240/240 executions, 0 failures, 120/120 warm accepted, 0 caps.
- Correct / 24: D 12, T_G 14, T_P 14, B8_P 14, CVM_T 14.
- vs D_native summed (calls × per-call): CVM_T 0.876 (0.930 × 0.942), B8_P 0.893 (0.967 × 0.923).
- **CVM_T / B8_P:** summed 0.981 [0.82, 1.18], geometric 0.953 [0.82, 1.12]; per-call 1.021; quality 2 vs 2.
- **Decision A:** simple reuse beats dense on this panel; CVM's increment is unestablished.
- **Attribution twins** (separate diagnostic): executed prunable prefix CVM_T 0.48–0.52, fresh T ~0.40, B8_P 0.33–0.36.
- v14 corrections appended in `results/numerical_qk_cvm_t_20260926/corrections_v15.md` (the LongBench phase weighting used the AIME canvas mix, plus wording fixes).

## Budget
- GPU 2.87 of 6 h; wall ~3.2 of 8 h.
- 243 of 250 executions: 240 panel + 3 setup.

## Next (single action)
Take Decision A to Fan:
- stop CVM-T as the contribution;
- keep the long-context held-bitmap (B8_P) per-call saving as an engineering result, noting it is prior art;
- start a new round only for a mechanism with a plausible increment over B8_P.

Reproduce the panel (resumable):
`$P/panel/run_panel.sh af1fc0d5f728`, then
`python -m scripts.v15_summarize --protocol results/numerical_qk_longcontext_scored_20260926/frozen_protocol.json --ledger $P/panel/ledger.jsonl --gold $P/private/gold_scorer_only.json --out <dir>`.
