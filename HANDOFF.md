# Native numerical QK reuse — bounded first delivery complete

## Identity / authority
- Sole user execution spec v5; transcript and old v3/v4 task prompts not used.
- Branch `research/numerical-qk-reuse-native-20260924`.
- Junyu base/newest fetched ref: `053441c6c6741ada6728dfc97d8faa6ea2be72aa`.
- Haowei reference/newest fetched ref: `b23f969a3c52417ba84a999ff7a31e3dd00bb697`.
- Original dirty worktrees/peer branches/history untouched. Normal pushes, no force.
- Last verified delivery before final report: `58fa077816805dbdb90a07aeb0d90e057fe101ba`.
- Current report SHA: use git HEAD and ls-remote. Loaded module/binary hashes, not report HEAD,
  are execution identity: results/numerical_qk_reuse_20260924/execution_identity.json.

## Result and scope
- Native Torch; old vLLM P0 remains independently unresolved.
- Four frozen AIME26 IDs2/8/14/20, seed42,8192/thinkingON/native adaptive/EOS.
- D/T/M1/M3 all4 complete. Quality3/4,3/4,0/4,0/4 respectively.
- Mean request wall41.3/39.9/254.2/166.0s; initial smoke includes incurred JIT/setup.
- M1/M3 each3 caps and4 unparsed. Do not change scorer to recover analysis-channel numbers.
- Four questions are not quality noninferiority; no paper claim from this result.
- Original quality attempt0 immutable; four D timing-only retries separate and token/call matched.
- Main concise result: results/numerical_qk_reuse_20260924/paper_decision.md.
- Detailed tables: smoke_report.md, request_table.csv, timing-control CSVs, smoke_summary.json.

## Implementation / tests
- method_contract.md: numerical_reuse_current_V, per-head Q128/KV64, current Gaussian32 V.
- Real score anchors calls0/8/16; M1 decision each call; M3 R2 every2 actual calls.
- FP32 cached BF16-transformed scores and routing; BF16 output, current original V.
- Cached consumer takes no Q/K. GPU throwing-spy test forbids current-QK producer on reuse.
- Both arms86.9% target QK-element reuse; PV skip59.5%/57.0%; no DRAM-byte claim.
- All8 actual M1/M3 dispatch traces pass clocks/current-QK counts; no unsupported-mask refresh.
- Current-V update, R1 equality, mask/partial/GQA, invalidation tests passed.
- GPU:6 math,2 lifecycle,12 fresh-history cases PASS. CPU scope in qualification_current.json.
- Detailed stop observer:6 CPU tests and dense GPU parity PASS.
- One diagnostic ID2: all4 arms match primary tokens/calls. M1 stable-not-confident670/908;
  M3 855/1289, D1/146, T6/208. Descriptive trajectories, not complete causal attribution.
- Native stop stable AND confident; ST1. No acceptance/temperature/SC/EOS modification.
- Pinned Junyu LOCAL query-relative window differs from installed native dense SDPA support.
  T/M1/M3 preserve Junyu support. Do not call all-kept native-mask equivalence.
- Same-stream dynamic cache only; CUDA Graph replay and Haowei executor NOT qualified/integrated.
- TBT=N/A: no output-ready burst events. Event timeline excludes only initial prefill;
  includes host gaps and later commit/encoder work, not CPU generation or pure GPU activity.

## Remote evidence / resource status
- Host exouser@149.165.151.254; GPU idle; own PIDs1413795/1417003/1417877 exited0.
- Root /home/exouser/dyh/numerical_qk_reuse_native_20260924.
- Immutable worker source code_cp2; CPU tools analysis_cp1; shared environment untouched.
- Runs: dense_smoke_02, integrated_smoke_01, m3_smoke_01, stop_diagnostic_01.
- Raw answers/gold remain private; only hashes/redacted tables committed.
- Stop session adb80a5a22a646b4a40ac423593fb1b9; source and raw hashes in stop_summary.json.
- Python /home/exouser/miniconda3/envs/ljy_dlm/bin/python, Torch2.6.0+cu124.
- Full model snapshot f7f5b7f5fa82ffc52addd066915886d497f5517b, no copied/downloaded weights.
- GPU-process3056.85s, measured process CPU2881.48s; build/analysis metering limits recorded.
- Own remote storage662MiB, free roughly60GiB. Failures and raw preserved; resource_ledger.json.
- Initial transfer rejection explicitly approved by user; later build-script rejection not bypassed.

## Next unique question / reproducibility
- Do not start240 requests, broad sweep, M2, new sampler, or old value/kernel/guard directions.
- R3, warm PROFILE, same-support current-output comparator, I-DLM inference are NOT_RUN.
- I-DLM preflight found no supported checkpoint/adapter; no guessed stride/verification semantics.
- Next diagnostic before natural expansion: same state/cache/support, cached final scores versus
  routing_only_current_output, with actual current-QK cost charged. It is not implemented yet.
- Local WSL: /mnt/e/dlm/numerical_qk_reuse_native_20260924.
- Exact CPU report reproduction command is STATE.json next_command.
- CPU reports do not restart models. GPU launch scripts reject busy device/existing run directory.
- git push git@github.com:coconight01/dlm_test.git HEAD:refs/heads/research/numerical-qk-reuse-native-20260924
- git ls-remote git@github.com:coconight01/dlm_test.git refs/heads/research/numerical-qk-reuse-native-20260924
