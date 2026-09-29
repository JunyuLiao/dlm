# v27b (M1/M2/M3 frontier): strongest dense = D_c64; long-context panel running; selector latency found

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

## Running
- dllm and mpk: the LB-long panel (run-dir `v27_long_lb_001`, 216 executions per host).
- Afterwards, dllm runs the rp/dp/fresh direct-cost profiles, and mpk runs the unshared A64 profiles (`v27nslong`/`v27ns`).

## Next
1. Score the LB-long panel on mpk: the v15 LB scorer needs the NeMo checkout, which only mpk has. Report decode span, calls, quality and wall vs D_c64.
2. From the ns/dp profiles, choose an unshared A64 variant with a cheap decision call. Calibrate the M1-DP threshold and freeze a follow-up panel.
3. Short-context no-regression panel (AIME, LB) with the length gate.

Operational note: on Windows, TaskStop leaves the chain's bash script running as an orphan. After every stop, list `bash` processes with `v27_` in the command line and kill the leftovers.
