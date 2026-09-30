# v27 attention speedup at kernel and module level (paper-style)

Two levels in the way sparse-attention papers report them, both on one H100 (mpk), torch 2.12.1, official
FlashAttention-4 (vLLM flash-attn cute build), 2026-09-30.

## 1. Kernel level (`kernel_bench.jsonl`, `scripts/v27_kernel_bench.py`)

One GLOBAL decoder layer call of DiffusionGemma: 16 query heads over the 256 canvas queries, 2 KV heads,
head_dim 512, bf16, keys = context + 256 canvas keys, no mask. FA4 block-sparse with uniform random kept tiles
(Q128 x KV64) per (head, query block); canvas tiles always kept. Median of 50 timed launches after 10 warm-ups.
Speedup is relative to FA4 with every tile kept (bitwise equal to FA4 plain dense; checked per length).

| context | FA4 dense (ms) | SDPA (ms) | 5% kept | 10% | 20% | 30% | 50% | 100% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 0.441 | 2.56 | 3.9x | 3.3x | 2.7x | 2.3x | 1.7x | 1.02x |
| 16K | 0.766 | 5.01 | 5.6x | 4.5x | 3.3x | 2.6x | 1.9x | 1.00x |
| 32K | 1.410 | 10.07 | 8.3x | 5.9x | 3.9x | 2.9x | 2.0x | 0.98x |
| 64K | 2.691 | 19.87 | 11.3x | 7.3x | 4.4x | 3.2x | 2.0x | 0.98x |
| 128K | 5.273 | 39.50 | 14.1x | 8.3x | 4.8x | 3.2x | 2.1x | 0.99x |

- Actual kept fractions include the always-kept canvas tiles (e.g. 7.6% at the 5% setting for 8K).
- Building the FA4 block lists takes 0.10-0.11 ms per map and is paid once per held map.
- PyTorch SDPA with enable_gqa (the HF model's default path at head_dim 512) is 5.8-7.5x slower than FA4 dense.

## 2. Attention-module level inside real requests (`module_profile_32k.jsonl`, `module_profile_64k.jsonl`)

`scripts/v27_observe_profile.py` on panel protocol v27_long_lb_variants_v5 (piecewise_v3), request 0 of each bin
(34,879 and 71,516 prompt tokens), CUDA events around every GLOBAL attention call and its parts. The module time per
layer call averages everything a method spends per GLOBAL layer: the dense first call of each canvas, the observation
call, decisions, and the held sparse calls. Dense reference: the same FA4 dense call measured in the same requests
(median 1.46 ms at 32K, 2.95 ms at 64K).

| method | 32K ms/layer call | 32K speedup | 64K ms/layer call | 64K speedup | sparse call median (64K) | selector call median (64K) |
|---|---:|---:|---:|---:|---:|---:|
| B (A64, hold) | 0.594 | 2.46x | 0.676 | 4.36x | 0.31 ms | 4.98 ms, once per canvas, async |
| M3 R6 DP, -ln2 | 0.864 | 1.69x | 1.093 | 2.70x | 0.55 ms | 0.27 ms (DP decision) |
| M3 R6 DP, -ln2, first layer dense | 0.970 | 1.51x | 1.506 | 1.96x | 0.52 ms | 0.27 ms |
| Fan M1 R1 A8 | 4.019 | 0.36x | 7.179 | 0.41x | 0.62 ms | 4.64 ms, every step |
| Fan M2c R1 A8 | 5.102 | 0.29x | 9.498 | 0.31x | 0.50 ms | 7.93 ms, every step |
| Fan M3 R3 A8 | 2.801 | 0.52x | 4.459 | 0.66x | 0.54 ms | 4.66 ms, every 3 steps |

- The main-stream event totals exclude the observation-call selector that B and the async M3 variant run on a side
  stream. Counting it serially as an upper bound, B is 1.89x (32K) and 3.33x (64K).
- Fan's plain methods are slower than dense at module level. Their sequential kept-state selector costs 2.3-7.9 ms
  per layer per decision, more than dense attention itself (1.46-2.95 ms), and they re-observe full QK scores every
  8 steps (6.2-11.4 ms per layer call). The variants remove this: one fused observation per canvas, the dense-prefix
  decision (0.27 ms), the side-stream selector and held maps.

## How the levels relate

Kernel speedups of about 7x at the kept fractions actually used (10-12% of tiles at 64K) become 2.7-4.4x at module
level once the dense first call, the observation and the decisions are paid, and 1.15-1.16x (request) or about 1.27x
(decode excluding prefill) end to end, because GLOBAL attention is 14% of a dense 64K request and 35% of a 64K
decoder forward (`../substrate/time_breakdown.json`).
