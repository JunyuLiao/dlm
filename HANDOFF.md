# Native numerical QK reuse — checkpoint 2

## Identity and authority
- Sole specification: user v5, 2026-09-24; transcript not read/copied.
- Own branch: `research/numerical-qk-reuse-native-20260924`.
- Junyu base/ref: `053441c6c6741ada6728dfc97d8faa6ea2be72aa`.
- Haowei read-only ref: `b23f969a3c52417ba84a999ff7a31e3dd00bb697`.
- Both fetched newest refs matched the pins. Peer worktrees remain unchanged.
- Worker module hashes in each config/receipt are execution identity; report HEAD is not.
- Previous vLLM P0 is independent, unresolved, and not qualified by these tests.

## Completed
- Math/cache/runner CPU checks; six summary pytest tests passed on remote CPU.
- GPU: six math + two dispatch/lifecycle tests; twelve fresh Junyu comparisons PASS.
- Original dense four AIME attempt-0 outputs complete: 3/4 correct.
- Fresh Junyu T four complete: 3/4 correct. One seed, four distinct questions only.
- Observer parity: same-load plain/counts/counts+events tokens and calls exact.
- Separate dense timing controls match original tokens/calls across loads, all four.
- M1 first two complete: 908/1513 calls vs dense146/132. Severe trajectory inflation.
- Their actual per-layer dispatch clocks pass: anchors0/8/16; reuse current-QK count0.
- No unsupported-mask refresh in these requests. No CUDA Graph qualification claimed.
- Partial redacted evidence: checkpoint2_partial.json and checkpoint2_dispatch.json.
- Quality attempt0 is immutable; event-on retries remain separate timing-only tables.

## Contract / limits
- `method_contract.md`: numerical_reuse_current_V, not routing-only current-QK output.
- Geometry per-query-head Q128/KV64; rank32/seed1729; pinned T_s50 thresholds.
- FP32 cached BF16-transformed scores; current projected V for route, current original V for PV.
- Full numerical anchors at actual calls0,8,16; M1 route each call; M3 route every2.
- Cached executor has no Q/K inputs; GPU throwing-spy confirms refresh is not called on reuse.
- Dispatch counts are not measured DRAM bytes. All allocation/guards/bookkeeping charged.
- Native sampler/SC/EOS preserved; actual stop stable AND confident; ST1 in installed native config.
- Installed native LOCAL mask differs from pinned Junyu window rule; T/M1 inherit Junyu rule.
- Do not attribute all dense/T differences to numerical reuse; see method_contract.md.
- Events measure initial-encoder-end to final CUDA-event timeline, including host gaps/later commits.
- CPU generation boundary and output-ready burst events unavailable: TBT=N/A.
- Initial smoke includes JIT/cache/setup; no warm performance qualification.

## Live execution — do not duplicate
- Host exouser@149.165.151.254, only H100; own PID1413795, PGID/SID1413792.
- Root `/home/exouser/dyh/numerical_qk_reuse_native_20260924`.
- Immutable worker source `code_cp2`; active `runs/integrated_smoke_01/process.log`.
- Batch does observer parity, dense timing controls, T4, M1 four. M3 NOT queued.
- Python `/home/exouser/miniconda3/envs/ljy_dlm/bin/python`.
- Full model snapshot f7f5b7f5fa82ffc52addd066915886d497f5517b; no copied weights.
- Private rebuilt libraries and native model hashes: build_identity.json / per-run config.
- Last disk61GiB free; completed GPU-process472.32s plus current active batch.
- Never edit active source or shared build/environment. CPU analysis uses separate analysis_cp1.

## Resume
1. Inspect current batch atomic receipts/log and process exit; no self-matching pgrep.
2. After all M1: native_reuse_score.py with frozen manifest; native_reuse_audit_receipts.py.
3. Redacted native_reuse_summarize.py: original dense quality + separate timing-control-root.
4. Push four-question M1 checkpoint, then run same immutable code M3 decision-interval2.
5. Complete four M3 requests; score/audit/report/push. No240 expansion this round.
6. Separate exact stop-signal diagnostic is being prepared; never label it formal timing.
7. I-DLM preflight NOT_RUN applicability: no supported adapter/checkpoint found; do not invent stride.

Local WSL directory `/mnt/e/dlm/numerical_qk_reuse_native_20260924`.
Next exact CPU command (remote code_cp2):
`CUDA_VISIBLE_DEVICES= PYTHONPATH=src:. <python> scripts/native_reuse_score.py --manifest results/numerical_qk_reuse_20260924/private/smoke_manifest.json --output ../runs/integrated_smoke_01 --phase smoke --condition M1 --ids aime26/2 aime26/8 aime26/14 aime26/20`
