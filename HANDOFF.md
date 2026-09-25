# v11 (DONE): preselected-support Hopper consumer -> H1/H3 (negative E2E)

- Kernel `experiments/value_direction_hopper/csrc/support_consumer.{cu,h}`, ATen `vd_support_v1`, build/verify
  `experiments/value_direction_hopper/support.py` (build e3c283b8cbb9b79c on /media/volume/dllm-1/dyh/...bridge_20260925/build).
  Support read before any tile work (poisoning/counters/scaling); bit-identical to fresh T on its support;
  sanitizer clean. vs O Triton consumer: rel-Fro <= 0.0025, max-abs envelope failed 60/660 -> H1 numerically != O.
- Router: `integration.Attention(consumer='hopper', support_build=...)` (ordinary/held consumer only).
  D_mask plug-in: `experiments/numerical_qk_reuse/dense_mask.py` (legacy-mask dense diagnostic).
- Same inputs: consumer 0-25% faster, complete M1 call H1 ~= O (selector 0.6-1.4 ms/call = 2-3x fresh T whole call);
  step: no gain; fresh-T step +14% vs native.
- Panel (39 of 64 requests, 0.56 GPU-h): quality Dn 3/4, T 3/4, H1 2/4, H3 3/4.
  H1/T geo 0.851 only via fewer calls (x0.81, per-call x1.10, two lost answers).
  H3/T geo 1.225 (summed 1.348).
- Bugs fixed: aligned cross-role barrier; `support` param shadowing; v10 guard_mode drop (v10 notes appended).
- Results: `results/numerical_qk_hopper_support_bridge_20260925/` (decision.md, morning_brief.md).
- Next: fused historical decision in the Hopper producer (anchor summaries + live T), GLOBAL layers first,
  target complete call <= fresh T (`scripts/v11_same_support_calls.py`).

# (previous) v10 overnight: bounded specialization + one warm repair + 4-question panel (DONE)

## Identity
- Spec: user-supplied v10. Resumed from `af6dc69` (v9). Branch `research/numerical-qk-reuse-native-20260924`.
- Host mpk (149.165.151.254), 1x H100. Interpreter: conda `ljy_dlm` with user site ENABLED:
  torch 2.6.0+cu124 and triton 3.2.0 from `~/.local`, transformers 5.11.0 from conda.
  Details: `results/numerical_qk_overnight_runtime_20260925/environment_and_storage_identity.json`.
- Every model job ran from a read-only `../deploy/<sha12>/` snapshot. Triton caches were private dirs on the
  large disk; `~/.triton` was never touched.
- Started 06:57:30Z. Used 56/72 complete requests and 1.14/5 GPU-h (ledger:
  `/media/volume/dllm-1/dyh/numerical_qk_overnight_runtime_20260925/resource_ledger.jsonl`).
  No live jobs. Root 56 GB free; nothing migrated or deleted.

## CP1: length-generic kernels (commits 901e360, f03bdf6)
- `experiments/numerical_qk_reuse/generic_kernels.py`: bodies textually identical to the static kernels, with
  runtime K/KT/PREFIX_TILES (do_not_specialize). K is passed as K/KDIV, where KDIV (pow2 <= 16) is a constexpr.
  This is the one fallback after the first mismatch: K%4==0, K%16!=0 lost constexpr alignment and changed the
  reduction order by 1 ulp. `tl.multiple_of` on a scalar had no effect.
- Flags: `kernel_variant='static'|'generic'` (runner `--kernel-variant`); `warmup_generic()` compiles the bounded
  production set.
- 35 tests: bit identity of every entry point/buffer/counter; variants bounded; 0 compiles for unseen lengths.
- Cold (new process, empty cache) /2: 94.7 -> 32.5 s; /8 first-seen 85.2 -> 38.6 s; warm unchanged. Tokens
  identical to static and v9.

## CP2: one warm repair (commits 947cf49, 48ad289, 01efe39, e955c29)
- Matched trace D vs P (4 windows): the largest added device cost is `_route` (32 CTAs, sequential tiles;
  late ordinary 15 ms, anchor 26 ms). About +1.0-1.4k device events per step. LOCAL layers dominate. Syncs equal;
  no allocations.
- Repair: `telemetry='minimal'` + `guard_mode='fused'` (one Triton guard pass, same coverage) + no memset for
  unused counters. Bit-identical. Same-state step -3.0 ms (~2%).
- Counterfactual (routing removed, same support): still ~11% slower per step than native.
- Not taken: T fast path (~1.6 ms), `_route` tile-parallel restructure (a new kernel design).

## CP3: panel (48 runs, 0 failures; all repeats token-identical to attempt 0)
| warm s | D | T | P | O |
|---|---:|---:|---:|---:|
| /2 | 22.8 | 35.0 | 19.5 | 19.3 |
| /8 | 20.3 | 32.7 | 33.5 | 33.3 |
| /14 | 64.9 cap | 64.1 | 119.5 cap | 118.5 cap |
| /20 | 55.6 | 21.7 wrong | 29.7 | 29.5 |
Quality 3/4 in every arm. O/P geometric 0.992; O/D 1.078 (summed 1.226); O/T 1.091 (summed 1.307).
P==O tokens on all 4 questions. The native-mask vs legacy-mask discrepancy remains.
The 8-ID expansion was NOT triggered (P->O < 5%, not competitive).

## Files (Git)
`results/numerical_qk_overnight_runtime_20260925/`: environment_and_storage_identity.json,
specialization_contract_and_tests.{json,md}, cold_disk_warm_process_comparison.{csv,json},
matched_native_sparse_trace_delta.{json,md}, chosen_warm_repair.md, complete_request_results.{csv,json},
decision.md, morning_brief.md. Tests: tests/test_v10_generic_kernels.py, tests/test_v10_minimal_telemetry.py.
Private/bulky: `/media/volume/dllm-1/dyh/numerical_qk_overnight_runtime_20260925/`
{cp1,cp2_trace,cp2_replay,cp3}/ (raw attempt-0 receipts under */private, never in Git).
Job scripts: `../run_v10_{cp1,trace,replay,counterfactual,cp3}.sh`.

## Next (single decision; decision.md)
Test historical (M1/M3) selection INSIDE Junyu's fused Hopper kernel: bitmap in place of fresh routing, same
mask, same kernel, head-to-head vs fresh T. Stop tuning the Triton M1 pre-QK path. If that does not beat fresh T
per call, the contribution must rest on selection quality vs a simple previous-bitmap baseline, not runtime.
Reproduce the panel: `./run_v10_cp3.sh 01efe3960713` (from `numerical_qk_reuse_recovery_20260924/`).
