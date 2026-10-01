# Batch-size scaling of one decoder forward (diagnostic, timing only)

`scripts/v27_batch_step_bench.py`, commit 02eead519. dlm2 (H100 80GB), 2026-10-01 13:26 local.
- **State.** The E4 run dir (protocol `v27_lb_confirm_e4_d3a7edb87a2b800a`, deploy `v27_e4_9e47ddb`) with the dense
  control D_fa4_allkept, LongBench-v2 64K item 0 (71,772 keys at the captured call), seed 404, decoder call 6.
- **Batching.** The exact call arguments are replicated to batch B: canvas, self-conditioning logits, positions,
  masks and every encoder-cache layer.
- **Timing.** The whole eager decoder forward is captured in one CUDA graph and replayed (CUDA-event median of 15).
- **GLOBAL attention arms:**
  - `dense`: FA4 block-sparse interface with every tile kept;
  - `0.12` / `0.2`: seeded synthetic keep maps of that prefix fraction, with the first and canvas tiles kept;
  - `canvasonly`: prefix attention removed.
- **Not included.** Selection overhead. The eager path's per-call K/V concat (`kv_concat`) is inside every forward
  number and is reported separately. `sampler` is the official step's logits work on random logits.

| B | forward dense (ms) | keep 0.12 | keep 0.20 | canvas only | prefix-attention share | saving at keep 0.12 | sampler | kv concat |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54.4 | 41.8 | 42.8 | 40.0 | 26% | 23% | 2.5 | 1.2 |
| 2 | 70.8 | 57.6 | 58.8 | 55.3 | 22% | 19% | 4.8 | 2.4 |
| 4 | 117.7 | 93.7 | 96.0 | 88.7 | 25% | 20% | 9.4 | 4.7 |
| 8 | OOM | | | | | | | |

- **Reading.** At 64K the GLOBAL prefix attention stays at 22–26% of the forward from B=1 to B=4, and the
  block-sparse saving stays at about 20%.
  - At B=1, FA4 with 256 queries does not fill the GPU. Per-request attention time at B=4 is about half that of B=1.
  - MoE turns compute-bound as tokens grow.
  - So batching does not make attention dominate at this context length. The expectation that it would was wrong.
- **Limits.**
  - One state, 64K only. B=8 does not fit with the replicated 64K caches.
  - Synthetic keep maps; no selection overhead.
  - Not an end-to-end or serving measurement.
