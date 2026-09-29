# v27c (M1/M2/M3 frontier): strongest dense = FA4; the eager pipeline hides attention savings; move to a graph-captured substrate

Authority: user v27 doc + A64 doc + chat. GPUs are the user's own: no time window and no budget stop. Record GPU seconds only.
Results: `results/m1_m2_m3_frontier_v27_20260929/`.

## Fan's methods (always present; variants are layered on top, never substituted)
- **M1**: historical (observed) QK + current rank-32 projected V, current reference and causal T decide which KV64 tiles to keep; redecide every call.
- **M2**: M1 with the tile mean of V (pooling V) in place of the weighted projected V (`pooled` reference, `pooled_compact` execution).
- **M3**: M1 every R calls, the bitmap held in between.
- Output always uses current QK and original V. Junyu's fresh T (current QK + projected V) and B (bootstrap bitmap held) are separate arms.

## Done (before this handoff)
- Tiers 1–3, D_fast, compact M2, R6/R12/A64/B-hold, length gate, layer subsets, cross-layer shared support (see `tier3_report.md`, `direct_cost_report.md`).
- **c64 consumer** (`v27_consumer64.py`): 64-row, split-KV. The old 16-row consumer reloaded each K/V tile about 128 times. **D_c64** (the same kernel, all tiles kept) is the strongest dense baseline.
- **Fused observation**: the call-1 dense pass writes the M1 prefix summaries plus a compact FP32 score tail.
- **Long context vs D_c64**, per call, from `long_context/`:
  - 32K: held −6.7%.
  - 64K: held −15% to −17%; M1 shared D −5%.
  - BO fused: +5.5% at 32K, +9% at 64K.
- **Disk.** mpk root went from 97% to 76% by moving an inactive dyh directory to the attached volume (verified, symlinked).

## New in this session
- **Long RULER panel** (`specs/v27_long_ruler.json`): 1,144 executions, 11 arms including D_native, D_c64, T_scope and plain M1/M2c/M3, plus A64 shared/fused variants and B.
  - Two plumbing bugs cost two launches, both fixed with tests: long-RULER rows failed the ruler4k-only task check, and D_c64 lacked the runner branch and its bound fingerprint.
  - `scripts/v27_smoke_arm.py` now runs one real request per bound arm before every launch.
  - Run-dir `v27_long_ruler_006`: all 11 arms passed the smoke.
- **Fused fresh T** (`consume64_fresh_t`, v21 key `fresh_fused`): Junyu's fresh-T information selected inside the 64-row output kernel. QK is always computed; V load and PV are skipped per tile. It is a named variant, not M1. GPU tests pass against an independent torch reference. Profile queued (arm sets `v27fresh`/`v27freshlong`).
- **Selector latency is a root cause of weak M1 D-steps.** The generic summary-LOAD route has only 32 programs and a branchy loop that Triton cannot pipeline.
  - 64K: 5.1 ms per routed layer, vs 5.4 ms for dense attention of that layer.
  - 17.5K: 1.8 ms vs 1.5 ms.
  - `v27_route.py` implements the same decision as two branch-free loops (v21 key `route_pipeline`).
  - Bit-identity tests (`tests/test_v27_route_pipelined.py`) pass on GPU.
  - Bench (`selector/route_bench.json`): 1.7–1.9× faster; 64K exact mu 5.08 → 2.95 ms, which is still 0.54× a dense layer.
- **M1-DP** (`v27_dense_prefix.py`, v21 key `risk_state='dense_prefix'`): a named variant. The risk is computed against the dense prefix state and precomputed per summary, so decision calls run in parallel. GPU tests pass against a torch reference. Profiles are queued (`v27dp*`, `v27ns*`). Its threshold must be calibrated separately.
- **64K memory fix** (`v27_long.py`): generate() kept the O(n²) prefill mask mapping (7.9 GiB) through the first canvas. It is now elided after a semantic check. Tokens are unchanged.
- **Long RULER panel complete** (572 cells / 1,144 executions; `long_ruler_panel/README.md`):
  - No end-to-end effect: the initial prefill dominates and a request needs about 5 calls.
  - Per call vs D_c64: M3 R6/A64 shared+fused is 0.965 at 64K.
  - Quality at 64K: the plain A8 arms keep it. The shared-support A64 variants lose it (4–5 of 26 cells worse, none better); the unshared B does not.
  - Redacted records are in `generation_records_v27_long_ruler/`.
- **LongBench-v2 32K/64K bins** (24 each, natural length, no truncation) built by a subagent. The long-generation panel `specs/v27_long_lb.json` (432 executions) is frozen and running.
- `scripts/v27_datasets.py`: one map from each dataset to its base task, used by freeze/run/score.

- **LB-long panel** scored (420/432; `long_lb_panel/README.md`). Cross-layer shared support collapses quality. Unshared B/A64 at 64K: per call 0.935 vs D_c64, wall 0.94 [0.87, 1.02].
- **Official SOTA dense = FlashAttention-4** (vLLM fork, CuTe DSL, SM90 hd512). `official_baseline/README.md`: 64K kernel 2.91 ms, vs 4.9–5.5 ms for D_c64.
  - `v27_fa4.py` loads it from a dyh overlay. D_fa4 control; `fa4_consumer` makes M1/M2/M3 maps into FA4 block-sparse lists.
- **Substrate diagnosis** (`substrate/README.md`; read this first).
  - In-model FA4 sparse saves about 11 ms of GPU per call at 64K. But every eager forward is host-bound, so none of it reaches wall time.
  - On the pinned torch-2.6 `.local` overlay, transformers' MoE falls back to a per-expert loop with 60 host syncs per forward.
  - Under CUDA graphs the saving appears: common-state keep 0.1 is 0.77× per forward at 60K keys.
  - Piecewise inductor graphs (GLOBAL attention eager, vLLM split) on the conda env's own torch 2.12, 64K: in-harness FA4 dense 37.6 ms, held FA4 sparse 29.4 ms, observation call 66–78 ms.
  - The official HF compiled path is 44/55.7 ms per step at 17K/32K and OOMs at 64K.
  - All v21–v27 end-to-end ratios hold for the eager substrate only.

## Running
- Nothing. Both GPUs are idle.

## Next
1. **Make the piecewise substrate a fingerprinted runner option.**
   - `v27_piecewise_bench.piecewise` becomes a plugin-independent install recorded in configs and receipts.
   - Tests: LOCAL binding is semantically the native path; no recompiles across requests; tokens are stable across repeats.
   - Also compile the post-prefill encoder (the official path does); it is currently eager.
2. **Dense reference.** Use in-harness FA4 dense, which avoids HF's 60K-key concat. Report the D_fa4 plugin and the official compiled path as context.
3. **Re-measure quality on the new substrate.** The numerics change, so no earlier quality result transfers. Then re-freeze the unshared LB follow-up (`specs/v27_long_lb_unshared.json`) with FA4 execution and `substrate=piecewise`:
   - plain M1/M2c/M3 always;
   - B, M3 R6/R3 A64, M1-DP, pairs/no_last;
   - density gate, G75 F/S15/S30.
4. **Cut the observation-call overhead** (+28–40 ms at 64K). This is now the main cost of B/M3.
5. **Ported sparse baselines** (MAGE, SparseD, Quest, BLASST rules) on FA4 block-sparse, labelled as ports. Keep the MAGE/PulseCol/VATP collision in view.

Operational notes:
- On Windows, TaskStop leaves the chain's bash script running as an orphan. After every stop, list `bash` processes with `v27_` in the command line and kill the leftovers.
- **Diagnostics that must use a frozen config's source hashes** run with cwd in that deploy dir, via the private `v27_deploy_diag.py`.
- **Caches outside dyh.** Every host run must pin TMPDIR, TORCHINDUCTOR_CACHE_DIR, CUDA_CACHE_PATH and XDG_CACHE_HOME inside dyh (both helpers now do). Earlier runs left files in /tmp/torchinductor_exouser, ~/.nv/ComputeCache, /tmp/pytest-of-exouser and /tmp/exouser; they are awaiting the user's decision.
