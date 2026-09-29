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
  - Bit-identity tests (`tests/test_v27_route_pipelined.py`) and `scripts/v27_route_bench.py` are **pending a free GPU**.

## Running
- dllm: long RULER panel (its half: ruler32k, then ruler64k), then the fused fresh-T profiles (16K LB, 32K, 64K ×2).
- mpk: its half of the panel starts when another user's job frees the GPU.

## Next
1. GPU: pipelined-route bit-identity tests and bench. If it holds, run the `v27rp`/`v27rplong` profiles and add `_rp` arms to the next panel.
2. Score the long panel (`--extra-gold`) and summarize decode-only and E2E against D_c64 with clustered CIs.
3. Short-context no-regression panel (AIME, LB) with plain M1/M2/M3, variants and D_c64/D_fast.
