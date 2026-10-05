# Official-serving dense check: vLLM's native DiffusionGemma vs our dense control (2026-10-02)

**Question.** Is our dense control (`D_fa4_allkept`: HF model code + piecewise_v5 substrate + the official FA4 kernel)
as fast as the official serving system? The attention kernel was already the official SOTA (FA4); the rest of the
step (MoE, CUDA graphs, sampler, scheduling) had never been compared.

**Method** (`scripts/v27_vllm_dense_bench.py`, dllm).
- vLLM 0.30.0 (torch 2.13.0+cu130), native DiffusionGemma. FA4 is auto-selected for diffusion models on SM90; the
  CuTe-DSL FA4 kernels compile in its log.
- Batch 1, bf16, in-process engine, every engine step timed.
- Input: the exact prompt token ids and 8192-token budgets of 18 E14 dense cells that ran on dllm (6 per bin).
  Ours: the same cells' E14 ledger, same host.
- Prefill is chunked: 16,384-token chunks at 32K/64K and 8,192 at 96K, `gpu_memory_utilization` 0.92 / 0.95.
  Without chunking vLLM cannot serve 96K on one H100: it allocates full-length KV for every layer (about 224 KB per
  token) plus a whole-prompt activation reservation.
- vLLM rejects per-request seeds for diffusion models, so trajectories are not paired. Per-step time is compared per
  item; step counts and walls only in distribution.
- Per-step = decode span / steps (vLLM includes canvas commit steps; our S/N includes encoder appends).

| bin | cells | prompt tokens | per-step ms, vLLM / ours (ratio) | prefill s, vLLM / ours (ratio) | steps per request, vLLM / ours | request wall s, vLLM / ours |
|---|---:|---|---|---|---|---|
| 32K | 6 | 30728–34879 | 29.1 / 38.5 (0.76) | 0.89 / 2.03 (0.44) | 226 / 198 | 7.3 / 9.6 |
| 64K | 6 | 59895–71516 | 40.5 / 45.0 (0.90) | 2.48 / 5.05 (0.49) | 143 / 116 | 8.2 / 10.3 |
| 96K | 6 | 92763–94270 | 48.0 / 49.2 (0.98) | 4.05 / 7.20 (0.57) | 322 / 278 | 19.5 / 20.8 |

**Result.** The official vLLM serving system is faster than our dense control:
- per step 0.76× at 32K, 0.90× at 64K, 0.98× at 96K;
- prompt prefill 0.44–0.57×.

Our attention kernel is the same FA4, so the difference is in the rest of the system: MoE kernels, whole-step CUDA
graphs, the compiled sampler and prefill.

**Consequence.**
- All end-to-end speed ratios so far hold against our HF-based dense substrate, not against the fastest official
  serving system.
- For paper-grade claims, the method must run inside vLLM's DiffusionGemma path and be measured against vLLM dense,
  where dense and sparse differ only in skipped blocks of the same FA4 call.
- vLLM's FA4 varlen interface accepts `page_table` and `block_sparse_tensors` together, so a port is feasible in
  principle. To verify:
  - paged-KV block-sparse correctness;
  - the observation kernel reading the paged cache;
  - sampler hooks for T.
