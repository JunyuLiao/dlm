# v27 execution substrate: why held sparse steps showed no in-model gain, and what a strong substrate looks like

**Question.** On the FA4 profiles, a held (block-sparse) decoder call cost about the same as a dense bootstrap call
(133 vs 131 ms at 64K), although the FA4 block-sparse kernel is about 7x cheaper than FA4 dense at the kept
fraction used. This directory answers where the saving went. All numbers are one H100, LongBench-v2 64K pool row
2 (59,895 prompt tokens) unless stated otherwise. They are diagnostics; none of them is a scored panel.

## 1. The attention saving is real inside the model

`scripts/v27_consume_timing.py` (CUDA events around the model's own calls; `in_model_forward_timing.jsonl`).
- FA4 dense takes 2.63 ms per GLOBAL layer call.
- FA4 block-sparse takes 0.34–0.41 ms at a median kept fraction of 0.09–0.10.
- The FA4 block-list build (0.05 ms) is cached per map (20 builds per 40 calls for B).

So about 11 ms of GPU time per decoder call is saved (5 GLOBAL layers).

## 2. The eager pipeline hides it: host-bound forwards

Per decoder call, CUDA-event span vs CPU enqueue time, same request:

| substrate | dense call | held FA4-sparse call | host time |
|---|---:|---:|---:|
| pinned env (torch 2.6.0 via `bridge/runtime/.local` overlay) | 130.5 ms (B0), 149.5 (D_fa4 plugin) | 145.4 ms | ≈ span − 1 ms |
| same conda env, its own torch 2.12.1 (overlay dropped from PYTHONPATH) | 66.3 ms (B0), 64.7 (D_fa4) | 65.6 ms | ≈ span − 1–4 ms |

- **Host time equals the span.** The GPU waits for the CPU, so GPU savings cannot shorten the call.
- **Pinned env** (`torch_profiler_summary.json`):
  - torch 2.6 has no `grouped_mm`, so transformers' MoE falls back to `_grouped_mm_fallback`: a per-expert `torch.mm` loop with one `offs.tolist()` host sync per call.
  - That is 60 syncs, about 3.6K–5.5K matmuls and about 7K kernel launches per forward.
  - Per-forward time also varies with how many experts the canvas hits. This confounds any comparison of B0 (all-mask canvas) with later calls.
- **torch 2.12:** no syncs, but still about 4.5K launches per forward. Still host-bound.

**Consequence.** Every v21–v27 end-to-end ratio was measured on a host-bound substrate. They remain valid statements about that substrate. They are not statements about a strong one, and must not be carried over.

## 3. Same forward on a common real state, eager vs CUDA graph

`scripts/v27_graph_forward_bench.py`, `graph_forward_bench.jsonl`.
- The exact arguments of a real decoder call are reused.
- GLOBAL attention runs as FA4 dense, or as the same FA4 kernel through its block-sparse interface at a fixed density.
- The sparse runs are **timing only**: synthetic seeded maps, first tile and canvas tiles always kept, outputs discarded.

| keys | graph dense | graph all-kept (block-sparse API) | keep 0.5 | keep 0.3 | keep 0.1 | eager dense / keep 0.1 |
|---:|---:|---:|---:|---:|---:|---:|
| 17K | 40.7 ms | 40.4 | 38.4 (0.94) | 37.8 (0.93) | 37.2 (0.91) | 63.1 / 63.7 |
| 32K | 43.9 | 43.5 | 40.1 (0.91) | 38.9 (0.89) | 37.8 (0.86) | 62.3 / 62.2 |
| 60K | 51.3 | 50.4 | 44.1 (0.86) | 42.1 (0.82) | 39.7 (0.77) | 63.8 / 63.3 |

(call index 3; call 14 agrees within 1.5 ms). The block-sparse interface costs nothing at all-kept.

## 4. Official HF compiled path (reference)

`../official_baseline/official_compiled_path.jsonl`.
- **The official path.** DiffusionGemma's own `generate` with `cache_implementation='static'` compiles the decoder, the post-prefill encoder and the sampler with `torch.compile(mode='reduce-overhead', fullgraph=True)`.
- **Per denoising step, official attention (SDPA):**

  | Prompt | Eager | Compiled |
  |---|---:|---:|
  | 17K | 77.1 ms | 44.0 ms |
  | 32K | 101.1 ms | 55.7 ms |

- **64K.** The official path runs out of memory even in eager mode. Its prefill materializes the O(n²) mask; ours elides it (`v27_long.prefill_dense64`).
- **Numerics.** The compiled static path is numerically different from eager (tokens diverge after 5 and 64 tokens). Its static sliding cache also sees one extra token, per its own source comment.

## 5. A strong substrate: piecewise CUDA graphs (the vLLM split) + FA4

`scripts/v27_piecewise_bench.py`, `piecewise_64k.jsonl`. This is still a diagnostic, not a runner option yet.
- **Compiled region.** The decoder forward is `torch.compile(mode='reduce-overhead')`. It includes the 25 LOCAL layers: they are bound to the same pure native SDPA function they already use, so no plugin state is traced.
- **Eager boundaries.** The 5 GLOBAL attention modules sit behind `torch._dynamo.disable`. The variable-length KV concat, FA4 dense/sparse and all M1/M2/M3 logic therefore run unchanged.
- **Sampler.** Compiled exactly as the official path does.

Results (warm):

| Prompt | Call | Forward | Host |
|---|---|---:|---:|
| 17K | D_fa4 dense | 29.3 ms | 14 ms |
| 17K | B held (first, still-compiling request) | 27.2 ms (0.93× of the D_fa4 plugin) | — |
| 64K | B0 dense bootstrap (call 0 of each canvas) | 37.6 ms | — |
| 64K | D_fa4 plugin dense (pays HF's 60K-key concat) | 40.5 ms | — |
| 64K | held FA4 block-sparse (calls 2+) | 29.3–29.9 ms | — |
| 64K | observation call, B | 77.8 ms | — |
| 64K | observation call, M3 R6 A64 | 66 ms | — |

At 64K the forward is GPU-bound (host about 15–18 ms).
- **Correction (2026-09-30).** An earlier version called B0 an "in-harness dense that avoids HF's concat" and quoted held/B0 = 0.78×.
  - That was wrong. The integration receives the same concatenated keys as the D_fa4 plugin.
  - B0 is cheaper because it is call 0 of each canvas: an all-mask canvas hits fewer MoE experts.
  - Medians over different call indices are confounded by this. They are amortized numbers, not per-forward prices.
  - The clean per-forward number is the common-state bench in section 3: 0.77× at keep 0.1 and 60K keys.
  - The per-canvas ratios B 0.93× and M3 0.87× were computed against B0, so they carry the same bias.
  - The fair dense reference is D_fa4.
- **Request wall (1,024-token budget)** is dominated by the 60K prefill and the eager per-canvas encoder. Call counts also move ±20% between runs of the same arm, because small numeric changes alter the trajectory. Single requests therefore say nothing about end-to-end effects; that needs the paired panel.

## What changes

1. **Paper-grade timing needs a graph-captured substrate.**
   - Our piecewise one is faster than the official compiled path. At 17K its forward is 29.3 ms, plus about 4.5 ms for the rest of the step (measured at 64K), against 44 ms per official step.
   - It also runs 64K, where the official path does not.
   - Baseline and methods must share it. The contribution is only the increment over its dense FA4 forward (D_fa4).
   - FA4 dense runs in its fastest configuration: its own block-sparse interface with every tile kept. That output is bitwise identical to its dense path and 4–6% faster; see `../official_baseline/README.md`.
2. **Quality must be re-measured on the new substrate.** torch 2.12's `grouped_mm` and inductor change numerics, so no v21–v27 quality result carries over.
3. **Host-side cost of our method matters again.** The observation call (+28–40 ms at 64K) is now the main overhead to cut. The selection logic runs eagerly at the GLOBAL boundary.
4. **Environment.** The substrate uses the existing `ljy_dlm` conda env's own torch 2.12.1 (no new env, nothing installed) by leaving the `.local` torch-2.6 overlay off PYTHONPATH. Junyu's torch-2.6 value-direction binaries are not used by the FA4 paths.
