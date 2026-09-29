# v27b: selector latency, fused fresh T, long-context memory

H100 80GB. All numbers are single-GPU measurements with the scope stated for each. Kernel-level numbers are **not** end-to-end effects.

## 1. The M1/M2/M3 decision-call selector is latency-bound (kernel level)

`route_bench.json` (`scripts/v27_route_bench.py`) covers one GLOBAL layer (16 Q heads, 2 KV heads, head_dim 512, 256 canvas queries) with synthetic tensors. The selector is the summary-LOAD route that every M1 decision call, and every M3 call at R, runs per routed layer. Times are median ms.

| keys | dense64 attention of the layer | exact mu: generic → pipelined (w4) | compact M2 pool: generic → pipelined (w4) |
|---:|---:|---:|---:|
| 17,536 | 1.49 | 1.77 → 1.00 | 0.95 → 0.52 |
| 32,768 | 2.87 | 2.84 → 1.57 | 1.54 → 0.85 |
| 65,536 | 5.43 | 5.08 → 2.95 | 3.11 → 1.65 |

**Cause.** The generic route runs only QB × H = 2 × 16 = 32 programs. Each scans every KV64 tile in order. The loop body branches on "summarized tile?" and updates the retained state conditionally, so Triton cannot prefetch the summary loads, and `num_stages=3` changes nothing (5.15 vs 5.08 ms at 64K).

**Fix.** `experiments/numerical_qk_reuse/v27_route.py` implements the same decision in two branch-free loops: prefix summaries, then the tail scores. The state update is a select of the identical expressions, and FP fusion is off.
- `tests/test_v27_route_pipelined.py` shows the decisions are **bit-identical** to the generic kernel for exact, compact-pool and row-pooled mu, for fused-observation tail scores, and for NaN/inf tiles, at 3 key extents and 2 thresholds.
- The bench re-checks equality on every timed configuration.
- The win comes from the branch-free body with 4 warps: 8 warps is slower, and the stage count does not matter.

**What remains.** At 64K the exact-mu selector still costs 0.54× a dense layer, because the scan is sequential: about 2.9 µs per tile per program. Whether this shows up per forward is being measured with the `v27rp` / `v27rplong` profiles. Those include Fan's plain M1/M2c/M3 (A8, c64), where all 5 GLOBAL layers select on each decision call.

## 2. Fused fresh T at 16K (direct per-call cost)

`fresh_t_fused_16k.csv`: LongBench-v2 target at about 17.5K keys, canvas 0, 8 calls, model-forward boundary, one host.

- **Arm.** `T_fused_c64` is Junyu's fresh-T information selected inside the 64-row output kernel. QK is always computed; V load and PV are skipped per tile. It is a named variant, not M1.
- **Result.** At every threshold shift (−ln2 … +2ln2) it costs **1.04× D_c64 per call**. The shift barely changes the cost, so the per-tile QK, softmax and risk work dominates the PV that is saved.
- **Other arms at this length.**
  - M3 R6/A64 (shared, fused observation) held calls: 1.01× D_c64.
  - Its decision calls: 1.07×.
  - No arm is cheaper per call than D_c64 at 16K.

## 3. Long-context memory (64K), fixed

In the long RULER panel (run 006), the 64K stage failed with OOM for plain M1/M3 (A8, c64) and for T_scope. `scripts/v27_memory_probe.py` found the cause:
- `generate()` keeps the prefill mask mapping, two `[1,1,n,n]` bool masks, **7.9 GiB at 64K**, in a local variable through the whole first canvas.
- Our prefill wrapper also pinned the last spot-checked mask.
- D_c64 alone already peaked at 77.9 GiB.

`v27_long.prefill_dense64` now does two things:
- It elides the mapping for a batch-1, unpadded, empty-cache prefill longer than 256. It does this only after the model's own mask builder is verified once per process on a 1,500-token prefill (`tests/test_v27_long_masks.py`).
- It keeps no reference to the mask.

Effect on a 64K RULER request:

| | before | after |
|---|---:|---:|
| prefill peak | 77.9 GiB | 70.0 GiB |
| resident at the first decoder call | 70.1 GiB | 62.2 GiB |

- The D_c64 completion tokens are identical to run 006 (same id, seed 202).
- All 11 panel arms pass a 64K smoke; the peak is 71.7 GiB, for plain M1/M3 A8.
- The 64K stage is re-run in run-dir 008. The 32K stage of the dllm half stays in run 006, because only memory changed.

## 4. Long RULER 32K, dllm half: preview only

These numbers are a preview and are **not** carried over to the full panel. They cover 13 warm blocks, the dllm half of run 006, with paired geometric means vs D_c64:
- **Request wall.** 0.95–1.03 for every arm. The prefill dominates, and RULER answers need only about 4–5 decoder calls.
- **Decode span per call.**
  - A64 shared/fused variants: 0.99–1.01.
  - Plain M1/M2c/M3 (A8, unfused observation): 1.10–1.14.
  - T_scope: 1.20.
  - D_native: 1.24.
- **Interpretation.** With about 5 calls, B0 and the BO observation take 2 of them, which leaves about 3 calls to benefit.
  - Long-prompt, short-answer tasks bound what sparse attention can give end to end.
  - A long-context, long-generation task (LongBench-v2 at 32K/64K with thinking) is being prepared.
