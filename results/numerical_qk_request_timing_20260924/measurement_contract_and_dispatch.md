# v9 measurement contract and dispatch

Deployed source `669858060b12060c098423aa719f63ecb3789637`. Evidence: bounded dispatch spy (one untimed probe per row) and execution flags read at the step boundary inside `adapter.generate` vs in each replay.

## aime26/2@canvas1 (absolute position 365, local stored prefix 365, global 365)

| row | registry entry | native SDPA | dense_eager | selector | median ms | min | first |
|---|---|---:|---:|---:|---:|---:|---:|
| dense_eager_same_mask.anchor_inputs | `dispatcher` | 0 | 30 | 0 | 129.00 | 128.51 | 128.89 |
| native_decoder_forward | `sdpa_attention_forward` | 30 | 0 | 0 | 113.82 | 113.65 | 114.49 |
| native_denoising_step.anchor_inputs | `sdpa_attention_forward` | 30 | 0 | 0 | 119.87 | 119.44 | 119.95 |
| native_denoising_step.step1_inputs | `sdpa_attention_forward` | 30 | 0 | 0 | 122.46 | 122.24 | 122.95 |
| sparse_legacy_recompute.anchor | `dispatcher` | 0 | 0 | 30 | 160.35 | 159.52 | 159.95 |
| sparse_legacy_recompute.ordinary | `dispatcher` | 0 | 0 | 30 | 153.30 | 152.53 | 165.29 |
| sparse_prefix_block_summary.anchor | `dispatcher` | 0 | 0 | 30 | 161.56 | 160.60 | 161.55 |
| sparse_prefix_block_summary.ordinary | `dispatcher` | 0 | 0 | 30 | 153.35 | 153.12 | 160.30 |

Context mismatches vs production: {'native_decoder_forward': {}, 'native_denoising_step.anchor_inputs': {}, 'native_denoising_step.step1_inputs': {}, 'sparse_legacy_recompute': {}, 'sparse_prefix_block_summary': {}}

Production flags: `{"inference_mode": true, "grad_enabled": false, "autocast_cuda": false, "attn_implementation": "sdpa", "tf32_matmul": false, "current_stream": 0}`

## aime26/2@canvas6 (absolute position 1645, local stored prefix 1023, global 1645)

| row | registry entry | native SDPA | dense_eager | selector | median ms | min | first |
|---|---|---:|---:|---:|---:|---:|---:|
| dense_eager_same_mask.anchor_inputs | `dispatcher` | 0 | 30 | 0 | 131.46 | 131.04 | 131.22 |
| native_decoder_forward | `sdpa_attention_forward` | 30 | 0 | 0 | 115.67 | 115.30 | 115.62 |
| native_denoising_step.anchor_inputs | `sdpa_attention_forward` | 30 | 0 | 0 | 121.29 | 120.79 | 121.37 |
| native_denoising_step.step1_inputs | `sdpa_attention_forward` | 30 | 0 | 0 | 132.17 | 131.71 | 132.67 |
| sparse_legacy_recompute.anchor | `dispatcher` | 0 | 0 | 30 | 165.92 | 165.13 | 166.07 |
| sparse_legacy_recompute.ordinary | `dispatcher` | 0 | 0 | 30 | 165.75 | 164.39 | 1425.95 |
| sparse_prefix_block_summary.anchor | `dispatcher` | 0 | 0 | 30 | 167.01 | 165.91 | 167.28 |
| sparse_prefix_block_summary.ordinary | `dispatcher` | 0 | 0 | 30 | 167.35 | 166.20 | 174.29 |

Context mismatches vs production: {'native_decoder_forward': {}, 'native_denoising_step.anchor_inputs': {}, 'native_denoising_step.step1_inputs': {}, 'sparse_legacy_recompute': {}, 'sparse_prefix_block_summary': {}}

Production flags: `{"inference_mode": true, "grad_enabled": false, "autocast_cuda": false, "attn_implementation": "sdpa", "tf32_matmul": false, "current_stream": 0}`

## Definitions

- **native** = no binding at all; `ALL_ATTENTION_FUNCTIONS["sdpa"]` is `transformers.integrations.sdpa_attention.sdpa_attention_forward`; no T observer. Used for D in the request timing (runner condition `native_dense`, diagnostic off).
- **dense_eager_same_mask** = `_install_dense` + `attention_override=None` -> the BLASST dispatcher -> `dense_eager_attention_forward`. This is what v8 labeled `native_dense` (183.24 ms). Kept only as a separately named reference.
- **sparse_<selector>** = production `integration.install` (M1, preqk current output) plus exactly one `observe` wrapper; T bookkeeping is inside the timed step. Native pays no T work.
- Mask disclosure: L/S use `legacy_junyu_mask` (query-relative local window crop); native SDPA receives no sliding-window crop beyond the cache's own window. L vs S is algorithm-preserving; L/S vs D additionally differs in support.
- Event spans include host launch gaps; they are not GPU-active unions.
