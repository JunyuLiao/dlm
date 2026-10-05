# Official SOTA dense baseline for DiffusionGemma's 512-dim GLOBAL layers

**Geometry:** the canvas decode call of a GLOBAL layer, with 256 queries, 16 Q heads, 2 KV heads, head_dim 512, bidirectional, bf16, on an H100 80GB. All times are CUDA-event medians of 20 repetitions after 3 warm-ups, in milliseconds. These are kernel-level numbers, not end-to-end.

## Which official implementation is SOTA here

`baseline_survey.md` gives the sources. The DiffusionGemma technical report serves the canvas with **FlashAttention-4**, and vLLM's flash-attention fork accepts head_dim ≤ 512 on SM90. We verified this in the installed `cute/interface.py`: `is_sm90_range = 8 <= head_dim <= 512`. It is loaded unchanged from a dyh overlay (`experiments/numerical_qk_reuse/v27_fa4.py`). The only change is a documented import shim for missing FP8/FP4 dtypes, whose paths bf16 attention never reaches.

| keys | **FA4 dense** | our dense64 (D_c64) | HF default path: SDPA mem-efficient on K/V repeated 8× | FlexAttention (best tile) | SDPA flash / cuDNN | FlashInfer |
|---:|---:|---:|---:|---:|---|---|
| 16,640 | **0.79** | 1.40–1.46 | 1.64 | – | no kernel (head_dim > 256) | not runnable here (JIT needs ninja) |
| 32,768 | **1.50** | 2.67–2.91 | 3.10 | 6.18 | no kernel | not runnable here |
| 65,536 | **2.91** (≈190 TFLOPS) | 4.90–5.47 | 6.16 | 20.8 | no kernel | not runnable here |

- Files: `fa4_bench_dllm_pinned_env.json`, `dense_bench_mpk_pinned_env.json`, `dense_bench_mpk_vllm_env.json`.
- **FA4 is the SOTA official dense reference.** It is 1.9× faster than our dense64 and 2.1× faster than the HF default path.
- All earlier per-forward ratios quoted against D_c64 must be re-based on FA4 before they are claimed.
- **Environment effect on Triton.** Our dense64 Triton kernel runs 1.7–1.9× faster in the vLLM environment than in the pinned environment: 2.88 ms vs 4.90 ms at 64K, both with Triton 3.7.1. The pinned environment's toolchain therefore handicaps every self-written Triton kernel. This is being investigated. FA4 numbers come from the pinned environment.

## Update 2026-09-30: provenance, same-process SOTA check, fastest FA4 configuration

**Provenance.**
- The kernel is FlashAttention-4 (CuTe DSL, BSD-3-Clause), as vendored in vLLM's official `vllm-project/flash-attention` fork.
- All 53 `vllm_flash_attn/cute` files in our overlay match, byte for byte, the RECORD hashes of the official vLLM nightly wheel (`wheels.vllm.ai/46d2b23ac…`).
- The SM90 head_dim-512 range (`is_sm90_range = 8 <= head_dim <= 512`) is in the vLLM fork's `main`. The Dao-AILab upstream caps SM90 at 256, and papers must say which one was used.

**Same process** (`dense_sota_same_process.json`; torch 2.13, one H100; CUDA-event median of 20; every configuration checked against FP32):

| keys | FA4 dense (num_splits=1) | FA4 best split-KV | FA4 block-sparse API, all tiles kept | FlashInfer 0.6.18 batch-ragged (FA2 backend) | FlashInfer single_prefill (FA2) | FlashInfer FA3 backend |
|---:|---:|---:|---:|---:|---:|---|
| 16,640 | 0.782 | 0.804 (s4) | **0.756** | 0.784 | 4.32 | no hd512 (static_assert hd ∈ {64,128,256}) |
| 32,768 | 1.462 | 1.493 (s3) | **1.391** | 1.467 | 8.99 | no hd512 |
| 65,536 | 2.851 | 2.913 (s4) | **2.672** | 2.891 | 19.25 | no hd512 |

- **FA4 and FlashInfer are tied** (within 1.4%). Split-KV does not help FA4 at this shape: the heuristic is 2–12% slower than `num_splits=1`.
- **FA4's block-sparse API with every tile kept is the fastest dense configuration.** It is 4–6% faster than FA4 dense and bitwise identical to it. `allkept_check.json` shows this at the model's real key counts (17,284 / 32,389 / 60,151 / 60,407, where the last tile is partial): max |dense − all-kept| = 0.0.
- **The dense baseline (D_fa4) therefore runs through that all-kept path.** Dense and sparse then differ only in the skipped tiles, and no share of the gain comes from the sparse code path itself.
- An earlier cross-environment run had FlashInfer 5% faster than FA4 (`flashinfer_bench_fan_env.json`). That was an environment artifact; the same-process numbers above supersede it.

## FA4's official block-sparse interface (same kernel, Q128 × KV64 keep maps)

Random keep maps, with the first tile of each row always kept. Errors against a masked FP32 reference are ≤ 9e-4.

| keys | keep 100% | 50% | 25% | 10% |
|---:|---:|---:|---:|---:|
| 16,640 | 0.78 | 0.42 | 0.28 | 0.18 |
| 32,768 | 1.42 | 0.73 | 0.46 | 0.25 |
| 65,536 | 2.69 | 1.32 (0.45× dense) | 0.79 (0.27×) | 0.40 (0.14×) |

- **Execution scales almost proportionally with the kept tiles.** Implementing M1/M2/M3 as block maps passed to the official kernel is therefore a clean "modify the official baseline" design. Dense and sparse differ only in the skipped blocks.
- **Per-call map build.** With the model's own non-contiguous K/V views, the per-call conversion of the keep map (argsort into FA4 block lists) adds about 0.15 ms. Building the lists once per decision and reusing them while the map is held removes this cost.

## Consequence

With sparse execution on FA4, the method's cost is dominated by its selector and observation. At 64K, the exact pipelined selector costs about 2.95 ms per routed layer, about equal to one dense FA4 layer (`../selector/README.md`). Per-call gains therefore require a cheap decision call (held M3 maps, M1-DP) and a cheap observation, measured against FA4 dense.
