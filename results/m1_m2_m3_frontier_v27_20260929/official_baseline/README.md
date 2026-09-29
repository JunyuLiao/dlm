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
