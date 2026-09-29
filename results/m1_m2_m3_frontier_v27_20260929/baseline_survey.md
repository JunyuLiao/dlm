# Baseline survey: dense and sparse attention baselines for DiffusionGemma GLOBAL layers (2026-09-29)

## Scope, method and confidence

- **What this is.** A helper-agent web survey of official repositories, documentation and papers. It answers six questions:
  - Q1: which dense kernels can run the 512-dim GLOBAL layers;
  - Q2: what the official DiffusionGemma inference stacks use;
  - Q3: sparse and caching methods for diffusion LLMs;
  - Q4: sparse attention for autoregressive LLMs;
  - Q5: which baselines to use;
  - Q6: who has benchmarked DiffusionGemma, and against what.

  It also includes the comparison-protocol table the coordinator requested.
- **What it is not.** Nothing was run. No GPU host was touched. The survey adds no evidence about our method.
- **How it was read.**
  - Most facts come from fetched pages: GitHub source files, pull requests, docs, and arXiv HTML or PDF.
  - Several PDFs were read through `pdftotext`: SparseD, the Uno paper (2609.04010), DualKV and the Gemma 4 report.
  - Many paper facts passed through a summarising fetch tool. The ones that matter most were re-read verbatim: the FA4 head-dim checks, the vLLM version-selection code, SparseD's latency setup, and the MAGE and Quest implementation text.
- **Confidence markers.**
  - `NOT_VERIFIED` means the item could not be confirmed from a primary source in this session.
  - A "(summary)" tag means the number was read through the summarising fetch tool and not from verbatim text.
- **Relation to existing notes.** This extends `literature_eval_protocol.md` and `blasst_and_eval_practice.md` and does not repeat their BLASST analysis.

---

## 0. Executive summary

**Q1: dense kernels for head_dim 512 on H100.**
- **Excluded, verified from code or docs:** upstream FlashAttention-2, FlashAttention-3 (hopper), upstream FlashAttention-4 on SM90, cuDNN SDPA as shipped, PyTorch's flash and cuDNN SDPA backends, TRT-LLM skip-softmax, and ThunderKittens.
- **Paths that do run 512/512 bf16 GQA non-causal on H100:**
  1. **FA4 (CuTe-DSL) in vLLM's fork of flash-attention.** SM90 hd512 was merged on 2026-04-03 with a 64×64 tile. vLLM main now *auto-upgrades* SM90 to FA4 for `head_size > 256` and for diffusion models. The DiffusionGemma technical report says its official vLLM stack uses FlashAttention-4 for the canvas attention.
  2. **vLLM `TRITON_ATTN`** (`kernel_unified_attention`). This is the backend named in Google's developer-guide serve command and forced by SGLang's DiffusionGemma path. It is known to be badly tuned at hd512: 6–13% occupancy, and a tuning PR is still open.
  3. **PyTorch SDPA memory-efficient backend** (CUTLASS 2.x, from xFormers). This is what HF transformers' default `sdpa` path runs for the 512-dim layers, after `repeat_kv`.
  4. **FlexAttention, Triton template.** It works only with reduced block sizes and stages.
  5. **FlashInfer FA2-template hd512.** `DISPATCH_HEAD_DIM` includes 512. A third party measured it at 19–28% of roofline on GH200 (NOT_VERIFIED).
  6. **cuDNN-frontend FROST SM90 D512.** This exists only as an open PR (#1159) and has not shipped.

**Q5: which baselines to use.**
- **Dense.** The strongest defensible dense baseline for the GLOBAL layers is **FA4 from `vllm.vllm_flash_attn` (`fa_version=4`) on SM90 at hd512**, because it is the kernel of the official serving stack. Report vLLM `TRITON_ATTN` and our own Triton kernel next to it.
  - Our Triton 64-row kernel may stay as the *same-kernel* dense reference only if it is at least as fast as FA4-vLLM on our shapes.
  - Otherwise every speedup must be quoted against FA4-vLLM.
- **Sparse, four primary baselines plus one optional.** All four primary ones run as ports, because none of the official codes supports DiffusionGemma at hd512 with a bidirectional canvas.
  1. **SparseD** (ICLR 2026): port its selection rule.
  2. **MAGE** (block diffusion, H100, FlashInfer; no code): port. This is the closest prior art and the main collision risk for M1/M3.
  3. **Quest** (ICML 2024, FlashInfer): port its page-bound rule, as MAGE and LoSA did.
  4. **BLASST skip-softmax** (MLSys 2026): port the in-kernel rule. The official kernels are causal only and support head_dim ≤ 256.
  5. Optional: **PulseCol**, the periodic-refresh analogue of M3 (no code; port).
- **Cache methods** (Fast-dLLM, dKV-Cache, dLLM-Cache, Elastic-Cache, Sparse-dLLM, DPad) address a no-cache LLaDA/Dream regime that DiffusionGemma does not have. Cite them as orthogonal, not as baselines.

**Q6: who has benchmarked DiffusionGemma.**
- **No paper found (as of 2026-09-29) evaluates sparse attention or any attention-kernel speedup on DiffusionGemma.**
- Only two papers measure DiffusionGemma speed at all:
  - the Google technical report (vLLM, FlashAttention-4, H100, FP8, batch 1, 4K in / 1K out, 13.56 ms per step);
  - Uno (2609.04010), which ran it in its own Nano-vLLM on H200 in bf16. Uno does not state the attention kernel (NOT_VERIFIED).
- The closest analogues all use a FlashAttention-family or FlashInfer dense baseline:
  - MAGE: FlashInfer dense and sparse, H100.
  - LoSA: FlashInfer.
  - SparseD: FA2 dense, FlexAttention sparse.
  - PulseCol: FlashAttention.
  - FlashBlock: implemented inside FlashAttention.

---

## 1. Setting check against the official config

All values below come from [config.json](https://huggingface.co/google/diffusiongemma-26B-A4B-it/raw/main/config.json).

- `num_hidden_layers` 30. The `full_attention` layers are indices 5, 11, 17, 23 and 29, so 5 GLOBAL layers. The other 25 layers are `sliding_attention`.
- `global_head_dim` 512, `num_attention_heads` 16, `num_global_key_value_heads` 2. This matches our GQA 16/2.
- The sliding layers use `head_dim` 256, `num_key_value_heads` 8 and `sliding_window` 1024.
- `canvas_length` 256, `max_position_embeddings` 262144.
- The `full_attention` layers use proportional RoPE with `partial_rotary_factor` 0.25.
- The config was written by `transformers_version` 5.8.0.dev0.

Other sources:
- The FlashAttention issue [#2427](https://github.com/Dao-AILab/flash-attention/issues/2427) describes Gemma-4 26B-A4B global layers as 8 query heads and 4 KV heads, and 4 global layers. That text does **not** match the DiffusionGemma config, which is authoritative for us.
- The [model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it) states up to 48 steps per canvas. Stopping requires mean entropy below 0.005 *and* identical argmax predictions on two consecutive steps.
- The [Gemma docs page](https://ai.google.dev/gemma/docs/diffusiongemma) says adaptive stopping typically gives 12–16 steps.

---

## 2. Q1: dense attention implementations and head_dim 512 on H100 (bf16, GQA, non-causal)

| Implementation | Max head_dim on SM90 (H100) | Runs 512/512 bf16 GQA non-causal on H100? | Primary evidence | Notes |
|---|---|---|---|---|
| **FlashAttention-2** (Dao-AILab) | 256 | **No** | README: "All head dimensions up to 256" ([repo](https://github.com/Dao-AILab/flash-attention)) | HF `flash_attention_2` fails on Gemma-4's 512-dim layers ([transformers #45201](https://github.com/huggingface/transformers/issues/45201)). |
| **FlashAttention-3** (`hopper/`) | 256 (Q/K = V). Mixed dims allowed: Q/K ≤ 64 with V ≤ 512, or Q/K in (128,192] with V in (96,128] | **No** | [hopper/flash_api.cpp](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/hopper/flash_api.cpp): "at most 256"; the `q_v` argument needs head_size ≤ 64 and hdim_v ≥ 256. [hopper/setup.py](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/hopper/setup.py) builds `64_512` and `192_128` "hdimdiff" variants | The `64_512` / `q_v` path is DeepSeek-MLA absorption, where V also acts as part of K. It does not apply when K ≠ V. |
| **FlashAttention-4, upstream** (`flash_attn/cute`, CuTe-DSL) | SM90: 8–256. SM100/110: ≤128, plus (192,128), (256,256), and (64 or equal, **512**) | **No on H100.** Only Blackwell accepts (512,512) | [`_validate_head_dims` in cute/interface.py](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/flash_attn/cute/interface.py) | The SM90 hd512 PR [#2422](https://github.com/Dao-AILab/flash-attention/pull/2422) is **open**. Tri Dao suggested a two-warpgroup design, and the author noted "performance in decode is bad" with one warpgroup. |
| **FlashAttention-4, vLLM fork** (`vllm-project/flash-attention`, imported as `vllm.vllm_flash_attn`) | **SM90: 8–512** | **Yes** | [fork cute/interface.py](https://raw.githubusercontent.com/vllm-project/flash-attention/main/flash_attn/cute/interface.py): `is_sm90_range = 8 <= head_dim <= 512`, and `_tile_size_fwd_sm90` returns `FwdConfig(64, 64, False, True)` for 512. Merged in [fork PR #130](https://github.com/vllm-project/flash-attention/pull/130) (2026-04-03), which imports upstream #2422 | Forward only. The function also takes a `sparse_block_size_q` argument (FA4 block-sparse). The fork's assertion message still says "256", which is stale text. An SM90 FP8 hd512 MMA PR [#199](https://github.com/vllm-project/flash-attention/pulls?q=512) is open. |
| **FlashInfer** (FA2 template) | `DISPATCH_HEAD_DIM` covers 64, 128, 256, **512** on main | **Yes (NOT_VERIFIED on SM90 in bf16)** | [include/flashinfer/utils.cuh](https://raw.githubusercontent.com/flashinfer-ai/flashinfer/main/include/flashinfer/utils.cuh). The vLLM docs list FlashInfer head sizes "64, 128, 256, 512" for CC 8.x–9.x ([vLLM attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/)) | Release notes announce hd512 only for SM120/121 ([v0.6.14](https://flashinfer.ai/releases/)). [PR #5044](https://github.com/flashinfer-ai/flashinfer/pull/5044) (2026-09-14) enables FA2 large-head FP8 KV on SM80+. A third party measured the FA2-template hd512 at 19–28% of roofline on GH200, versus 44–49% for "FA4" ([pegainfer #1051](https://github.com/pegainfer-project/pegainfer/issues/1051); NOT_VERIFIED). The FA3 template (`backend="fa3"`) at 512 is NOT_VERIFIED. |
| **FlashInfer MLA** (`head_dim_ckv=512`, `head_dim_kpe=64`) | n/a | **Not applicable** | [FlashInfer attention API docs (0.7.0)](https://docs.flashinfer.ai/api/attention.html) | It needs a single shared latent that serves as both K and V. DiffusionGemma has K ≠ V and 2 KV heads. |
| **cuDNN SDPA** (backend and frontend) | Documented Hopper prefill limit d ≤ 256 (cuDNN 9.18.1 matrix) | **Not shipped** | [cuDNN frontend Attention docs](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html). The SM90 D512 FROST engine is an **open PR** ([cudnn-frontend #1159](https://github.com/NVIDIA/cudnn-frontend/pull/1159), milestone 1.31.0, opt-in) | The support matrix [issue #378](https://github.com/NVIDIA/cudnn-frontend/issues/378) says "no SM90 FROST SDPA engine in develop" as of 2026-09-27. The PR reports a 5.46–5.78× gain over cuDNN 9.24.0 on causal Gemma-4 full attention on H200, which implies a slow pre-existing D512 path (NOT_VERIFIED). TransformerEngine's FROST backend ([TE #3527](https://github.com/NVIDIA/TransformerEngine/pull/3527), open) targets SM100/103 and says "the C++ cuDNN fused path caps at 256". |
| **PyTorch SDPA, flash backend** | 256 | **No** | [sdp_utils.cpp](https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp) `check_head_dim_size_flash` | |
| **PyTorch SDPA, cuDNN backend** | 128, raised to 256 on SM90/100 with recent cuDNN | **No** | same file, `check_cudnn_tensor_shapes` | cuDNN is preferred first on SM90/100 when cuDNN > 9.15, so it is excluded only by head_dim. |
| **PyTorch SDPA, memory-efficient backend** (CUTLASS, from xFormers) | No upper bound (alignment only) | **Yes, after `repeat_kv`** | same file, `check_head_dim_size_mem_efficient`. [SDPA docs](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html): `enable_gqa` works only on the flash and math kernels | This is what **HF transformers `sdpa`** runs for the 512 layers. HF's [`use_gqa_in_sdpa`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/integrations/sdpa_attention.py) returns False when head_dim > 256, which forces `repeat_kv`, 8× here. The kernel is SM80-style CUTLASS 2.x, with no TMA or WGMMA. |
| **PyTorch FlexAttention, Triton template** | No hard limit | **Yes, with reduced tiles** | The default BLOCK_M = BLOCK_N = 128 needs 266,240 B of SMEM at d = 512, above the 232,448 B limit shared by B200 and H100 ([unsloth #11102](https://github.com/unslothai/unsloth/pull/11102)). The fix is smaller blocks or fewer stages ([pytorch #132075](https://github.com/pytorch/pytorch/issues/132075)) | SparseD's official code uses FlexAttention for its sparse steps. |
| **PyTorch FlexAttention, `BACKEND="FLASH"`** (FA4) | Inherits FA4: 256 on SM90 upstream | **No with upstream FA4** | [PyTorch blog, 2026-03-04](https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) and [flex docs 2.14](https://docs.pytorch.org/docs/2.14/nn.attention.flex_attention.html) ("Experimental … user needs to have flash installed") | FA4 gained block-sparse iteration and GQA packing on Hopper and Blackwell, tuned around 128×128 blocks on Hopper. It might become usable with the vLLM fork (NOT_VERIFIED). |
| **xFormers** (`memory_efficient_attention`) | cutlass `FwOp.SUPPORTED_MAX_K = 65536`; `flash3` `SUPPORTED_MAX_K = 256` | **Yes, via the cutlass op** (same kernel family as PyTorch mem-efficient) | [v0.0.29 cutlass.py](https://raw.githubusercontent.com/facebookresearch/xformers/v0.0.29/xformers/ops/fmha/cutlass.py), [flash3.py](https://raw.githubusercontent.com/facebookresearch/xformers/v0.0.29/xformers/ops/fmha/flash3.py) | The FMHA code has moved to Meta's `mslk` package ([xformers fmha/__init__.py](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/__init__.py)). The current mslk limits are NOT_VERIFIED. |
| **vLLM `TRITON_ATTN`** (`kernel_unified_attention`) | "Any" | **Yes** (also has per-request causality for DiffusionGemma) | [vLLM attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/), [vLLM DiffusionGemma blog](https://vllm.ai/blog/2026-06-10-diffusion-gemma) | At hd512 the tuning PR [#43257](https://github.com/vllm-project/vllm/pull/43257) is open and stale. It reports regs/thread up to 255 and occupancy of 6–13%. On H100 at 100k input it measured Gemma-4 A4B 19.7 s → 11.7 s with tuning (summary). |
| **vLLM `FLASH_ATTN` (FA4)** | 512 on SM90 through the fork | **Yes** | [fa_utils.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/attention/backends/fa_utils.py) upgrades FA3 → FA4 on SM90 when `head_size > 256` or `model_config.is_diffusion` ("Per-sequence causal (dynamic_causal) requires FA4") | The backend table still lists FA4 as "≥10.0". Which path a given vLLM build takes must be checked from its log line (NOT_VERIFIED for our build). |
| **vLLM `FLEX_ATTENTION`** | "Any" | Probably yes (NOT_VERIFIED at 512 on H100) | backend table | |
| **SGLang** | Triton for Gemma-4 and DiffusionGemma | Yes (Triton) | [SGLang DiffusionGemma cookbook](https://docs.sglang.io/cookbook/autoregressive/Google/DiffusionGemma): Triton, eager and unchunked prefill applied automatically "because the full-attention head_dim is 512 and the canvas uses bidirectional attention" | [PR #34061](https://github.com/sgl-project/sglang/pull/34061) (merged 2026-09-23) enables decode CUDA graphs. The FA2 NVFP4 VO-split route ([#29305](https://github.com/sgl-project/sglang/pull/29305)) targets SM120/121 and is closed. |
| **TensorRT-LLM** | Its MHA/FMHA cannot serve H512 on non-SM100 GPUs | Gemma-4: yes, via **FlashInfer FA2** on SM90. DiffusionGemma: **not supported** | [supported-models.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/models/supported-models.md) lists Gemma-4 but not DiffusionGemma. [PR #18547](https://github.com/NVIDIA/TensorRT-LLM/pull/18547) (open) says "Gemma4's H512 layers cannot fall back to native MMHA" and that FlashInfer FA2 is the default off SM100 | Skip-softmax (BLASST) in TRT-LLM supports head dims 64, 80, 128 and 256, causal self-attention only ([TRT-LLM sparse-attention docs](https://nvidia.github.io/TensorRT-LLM/latest/features/sparse-attention.html)). |
| **ThunderKittens** | 64 and 128 (attention demos) | **No** | [TK paper](https://arxiv.org/html/2410.20399v1), [repo](https://github.com/HazyResearch/ThunderKittens) (TK 2.0, 2026-01-11) | |
| **Custom FA2-derived hd512** (DualKV) | 512 (BM = BN = 64, 4 warps, causal, training) | Research code only | [DualKV 2605.15422](https://arxiv.org/abs/2605.15422), App. I; code on the [gemma4-dev branch](https://github.com/amazon-science/dualkv-flash-attn-for-rl/tree/gemma4-dev) | An analogue for "custom 64-tile hd512 kernel". It is not a serving baseline. |

**How much faster FA4 is than Triton at hd512, in the official stack.**
- vLLM [PR #53175](https://github.com/vllm-project/vllm/pull/53175) (merged 2026-09-24) moved Gemma-4-31B FP8-KV on H200 from `TRITON_ATTN` to the composite `TRITON_FLASH_ATTN`, which uses FA4 for the hd512 layers.
- Output throughput went from 546.84 to 791.60 tok/s, and mean TPOT from 195.10 to 129.67 ms, which is 1.45× (summary).
- This setup differs from ours (autoregressive, FP8 KV, H200). It still shows that a Triton-based dense reference at hd512 is not the strongest available.

**A back-of-envelope estimate (arithmetic only, not measured).**
- Setup: one GLOBAL layer, 256 canvas queries × 64K keys, 16 query heads, D = 512.
- Work: about 0.55 TFLOP per layer per step against about 0.27 GB of K and V (2 KV heads), an intensity of about 2,000 FLOP per byte. That is well above H100's ridge point.
- Consequence: the dense baseline is **compute-bound**, so dense-kernel TFLOPS translate directly into baseline time. This is why the choice between FA4-vLLM, Triton and our kernel matters.

---

## 3. Q2: official DiffusionGemma inference and the attention each path uses

| Path | Status and source | Attention path for the 512-dim GLOBAL layers |
|---|---|---|
| **HF transformers** `DiffusionGemmaForBlockDiffusion` | [modeling_diffusion_gemma.py](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py); [model doc](https://huggingface.co/docs/transformers/main/en/model_doc/diffusion_gemma) | Uses `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)`. The flags `_supports_sdpa`, `_supports_flash_attn` and `_supports_flex_attn` are all True. The decoder sets `is_causal = False` ("In the decoder, attention is bidirectional!") and "reads but does not update the KV cache". There is no special handling for head_dim > 256. So `sdpa` (the default) becomes `repeat_kv` plus the SDPA mem-efficient kernel, and `flash_attention_2` fails. The docs suggest `cache_implementation="static"` to trigger `torch.compile`. |
| **vLLM** (official; merged in [PR #45163](https://github.com/vllm-project/vllm/pull/45163), 2026-06-12) | [vLLM blog 2026-06-10](https://vllm.ai/blog/2026-06-10-diffusion-gemma); [recipe](https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it) (`vllm/vllm-openai:gemma`, BF16, `--max-num-seqs 4`) | The blog says per-request causality is supported in "`TRITON_ATTN` and FlashAttention 4 (`FLASH_ATTN`)". [Google's developer guide](https://developers.googleblog.com/diffusiongemma-the-developer-guide/) serve command passes **`--attention-backend TRITON_ATTN`**. The [DiffusionGemma technical report](https://arxiv.org/html/2608.00146v1), §6, says the canvas uses "the highly-optimized FlashAttention-4 kernel". Current vLLM main auto-upgrades diffusion models on SM90 to FA4 (see §2). If FA4 is unavailable, `Gemma4Config.verify_and_update_config` logs "FA4 not available, forcing TRITON_ATTN backend" ([vLLM config API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/config/)). |
| **SGLang** | [PR #34061](https://github.com/sgl-project/sglang/pull/34061) merged 2026-09-23; [cookbook](https://docs.sglang.io/cookbook/autoregressive/Google/DiffusionGemma) | Triton attention backend is forced. The cookbook states "Not benchmarked for speed". |
| **Google JAX reference** | [google-deepmind/gemma `gemma/diffusion`](https://github.com/google-deepmind/gemma/tree/main/gemma/diffusion) (sampler, early stopping, transformer). Fine-tuning is in [google/hackable_diffusion](https://github.com/google/hackable_diffusion) | Reuses `gemma.gm.nn.gemma4`. Attention is plain `jnp.einsum` for the logits plus `jax.nn.softmax`, with the full logits materialised ([gemma4 `_modules.py`](https://raw.githubusercontent.com/google-deepmind/gemma/main/gemma/gm/nn/gemma4/_modules.py)). No Pallas, splash or flash attention was found in the reference. A TPU serving recipe was **not found** (NOT_VERIFIED). |
| **Vertex AI, NVIDIA NIM, MLX, Ollama, llama.cpp** | Listed by the [developer guide](https://developers.googleblog.com/diffusiongemma-the-developer-guide/) and the [model card](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | Attention implementation NOT_VERIFIED. These are not relevant for an H100 bf16 baseline. |

---

## 4. Q6: papers and reports that benchmark DiffusionGemma, and the closest analogues

### 4.1 Direct DiffusionGemma speed measurements (all found)

| Source | Stack and dense kernel | Hardware and precision | Setup | What is reported |
|---|---|---|---|---|
| **DiffusionGemma Technical Report** ([arXiv 2608.00146](https://arxiv.org/html/2608.00146v1), 2026-08-04, Google DeepMind) | vLLM with async scheduling and a per-sequence causal flag. The canvas attention uses **FlashAttention-4**. "Reference inference implementations are available in HuggingFace Transformers and vLLM." | H100, **FP8** | batch 1, 4,096 input / 1,024 output | Per step: t_fwd = 13.56 ms (12.63 ms GPU), TPF about 19.7, about 1,456–1,479 TPS. This is 7.1× Gemma-4 AR (204 TPS) and 4.8× AR+MTP (303 TPS), all end-to-end. The report notes that attention is about 4× slower than in Gemma-4 AR, because decode kernels cannot be used on a 256-token bidirectional canvas (summary). |
| **Uno** ([arXiv 2609.04010](https://arxiv.org/abs/2609.04010), 2026-09-03) | "All experiments … using our Nano-vLLM implementation" (PDF §5). The DiffusionGemma attention kernel is **not stated (NOT_VERIFIED)** | H200, bf16 | Throughput at batch 1 and at the largest batch that fits; entropy-bounded sampler with bound 0.1 | End-to-end system and per-request throughput. The paper notes that DiffusionGemma "is faster at batch size 1" but slower at large batch. |
| vLLM blog ([2026-06-10](https://vllm.ai/blog/2026-06-10-diffusion-gemma)) | vLLM, `TRITON_ATTN` or FA4 (which one ran is not stated) | H100 and H200, FP8 | batch 1, `vllm bench serve` | 1,008 tok/s on H100 and 1,288 on H200, end-to-end. |
| SGLang [PR #34061](https://github.com/sgl-project/sglang/pull/34061) | SGLang, Triton | H200, TP2, BF16 | 128 or 2,048 tokens, concurrency 1 or 4 | Request latency (0.66–1.39 s). |

**Explicit finding.** No sparse-attention, KV-selection or attention-kernel paper that evaluates DiffusionGemma was found in searches as of 2026-09-29. The searches covered arXiv, Hugging Face papers, GitHub and the vLLM and SGLang trackers.
- [2606.14620](https://arxiv.org/abs/2606.14620) studies token-commit order and makes no speed claims.
- [2609.29505](https://arxiv.org/abs/2609.29505) does not evaluate DiffusionGemma.

### 4.2 Closest analogues (other diffusion LLMs) and their dense kernel

| Paper | Model family | Dense kernel used as baseline | 512-dim? | Speed level |
|---|---|---|---|---|
| MAGE ([2602.14209](https://arxiv.org/html/2602.14209)) | Block diffusion: Fast-dLLM v2 7B, SDAR-8B, LLaDA2.0-mini | **FlashInfer** exact attention. Sparse kernels are also FlashInfer. | No | End-to-end TPOB (time per output block) and attention-layer time, on **H100** |
| LoSA ([2604.12056](https://arxiv.org/html/2604.12056v1)) | Block diffusion: Trado-4B/8B, SDAR-8B | **FlashInfer**, plus custom CUDA/Triton | No | Attention only (RTX A6000 / 5090) |
| SparseD ([2509.24014](https://arxiv.org/abs/2509.24014)) | LLaDA-1.5, Dream-7B (full recompute) | **FlashAttention(-2)** in model code. Sparse steps use FlexAttention. | No | Whole-generation latency on one sample (A800) |
| PulseCol ([2605.20813](https://arxiv.org/html/2605.20813)) | LLaDA-1.5, Dream | **FlashAttention** (version NOT_VERIFIED), plus a custom column-sparse kernel | No | Attention only and end-to-end (RTX PRO 6000) |
| FlashBlock ([2602.05305](https://arxiv.org/html/2602.05305v2)) | Trado (block diffusion) | Implemented *inside* the **FlashAttention** kernel | No | Token throughput (A100) |
| Focus-dLLM ([2602.02159](https://arxiv.org/html/2602.02159)) | UltraLLaDA, Dream | **FlashAttention** dense, Triton sparse | No | End-to-end vs vanilla (H200) |
| dInfer ([2510.08666](https://arxiv.org/abs/2510.08666)) | LLaDA-MoE / LLaDA2 | SGLang `RadixAttention` backends in its SGLang path (local copy `third_party/dinfer/python/dinfer/model/modeling_llada2_moe_sglang.py`) | No | TPS at batch 1 (8×H800) |
| Nemotron-Labs-Diffusion ([2607.05722](https://arxiv.org/abs/2607.05722)) | Tri-mode AR/diffusion | SGLang (kernel NOT_VERIFIED) | No | Throughput (GB200) |
| Mercury ([2506.17298](https://arxiv.org/abs/2506.17298)) | Proprietary | Undisclosed engine | NOT_VERIFIED | Tokens/s (H100) |
| Uno ([2609.04010](https://arxiv.org/abs/2609.04010)) | Diffusion-augmented AR, head_dim 128 | Nano-vLLM (kernel not stated) | No | Throughput |

The only 512-dim precedents are Gemma-4 kernel work, not sparse-attention papers:
- vLLM PRs #43257 and #53175;
- cuDNN-frontend #1159 and TE #3527;
- DualKV's custom hd512 kernel.

---

## 5. Q3: sparse and caching methods for diffusion LLMs

| Method | Venue and date | Code | Models | Datasets and metrics | Baselines compared | Speedup type (dense reference) | Port to our setting (bidirectional canvas + prefix KV cache, adaptive steps, hd512) |
|---|---|---|---|---|---|---|---|
| **SparseD** | ICLR 2026 ([repo](https://github.com/INV-WZQ/SparseD)); arXiv [2509.24014](https://arxiv.org/abs/2509.24014) (2025-09) | Yes (FlexAttention) | LLaDA-1.5, Dream-7B-Instruct | MMLU, GSM8K, HumanEval, RULER 4K/8K; accuracy. Latency at 4K–64K. | Sliding window, StreamingLLM, dKV-Cache, Fast-dLLM, dense FA | Whole-generation latency, batch 1, one RULER sample. 1.23–1.25× at 64K / 128 steps and 1.48–1.50× at 64K / 1,024 steps, vs FA(-2), on A800 | **Medium; port the rule.** Dense for the first 20% of steps, then a head-specific block top-ρ pattern that is reused. Their regime has no KV cache, so attention is recomputed over all tokens every step. The "first 20% of steps" must be redefined per canvas under adaptive stopping. The FlexAttention kernel needs small tiles at hd512. Our "fresh T" arm is SparseD-like. |
| **Sparse-dLLM** | AAAI 2026 ([AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/download/40586/44547)); [2508.02558](https://arxiv.org/html/2508.02558) | [Yes](https://github.com/OpenMOSS/Sparse-dLLM), plain PyTorch eviction | LLaDA-8B, LLaDA-1.5, Dream-7B | MMLU, ARC-C, PIQA, GPQA, GSM8K, MATH, HumanEval; accuracy. Generation 256/512, context ≤ 4K. | Vanilla, dLLM-Cache, dKV-Cache, Fast-dLLM | End-to-end tokens/s vs vanilla HF, on RTX 4090 | Low relevance. Its gain is cache eviction in a no-cache model. The retention rule could be ported as a token mask on prefix keys. |
| **dKV-Cache** | NeurIPS 2025 ([repo](https://github.com/horseee/dKV-Cache)); [2505.15781](https://arxiv.org/html/2505.15781) | Yes | LLaDA-8B, Dream-7B | MMLU, GSM8K, MATH500, GPQA, HumanEval, MBPP | Vanilla, few-step | End-to-end tokens/s, 2–10× vs vanilla (A6000, H20) | **Not applicable.** DiffusionGemma already commits canvases to a KV cache. Orthogonal. |
| **dLLM-Cache** | ICML 2026 per the [repo](https://github.com/maomaocun/dLLM-cache); [2506.06295](https://arxiv.org/html/2506.06295) | Yes | LLaDA-8B, Dream-7B (+ LLaDA-V, MMaDA) | GSM8K, GPQA, MATH, MMLU(-pro), BBH, MBPP, HumanEval, LongBench | Vanilla, dKV-Cache, Fast-dLLM | FLOPs (up to 9.1×) and TPS vs vanilla (RTX 4090) | Not applicable (prompt/response cache). |
| **Fast-dLLM v1** | ICLR 2026 ([repo](https://github.com/NVlabs/Fast-dLLM)); [2505.22618](https://arxiv.org/html/2505.22618) | Yes | LLaDA, LLaDA-1.5, Dream (+ LLaDA-V) | GSM8K, MATH, HumanEval, MBPP; accuracy + tokens/s | Vanilla | End-to-end throughput, up to 27.6×, cache plus parallel decoding (A100) | Not applicable. DiffusionGemma has a native cache and its own sampler. |
| **Fast-dLLM v2** | ICLR 2026 ([repo](https://github.com/NVlabs/Fast-dLLM)); [2509.26328](https://arxiv.org/abs/2509.26328) | Yes | Qwen2.5-derived block dLLM (needs training) | Reasoning and code | AR | 2.5× vs AR | A model, not a baseline. |
| **DPad** | arXiv [2508.14148](https://arxiv.org/abs/2508.14148) (2025-08; venue NOT_VERIFIED) | [Yes](https://github.com/Crys-Chen/DPad) | LLaDA-1.5, Dream | Reasoning and code | Vanilla, + Fast-dLLM | Up to 61.4× vs vanilla | Not applicable (suffix dropout; DiffusionGemma's canvas is a fixed 256). |
| **LoSA** | arXiv [2604.12056](https://arxiv.org/html/2604.12056v1) (2026-04) | Not provided | Trado-4B/8B, SDAR-8B | LongBench subsets, HellaSwag, WinoGrande, BoolQ; accuracy; 32K/64K | Quest, SparseD, dense | **Attention only**, up to 4.14× vs FlashInfer, batch 1 (A6000, 5090) | **High relevance; port.** Block DLM with a prefix KV cache. Reuses prefix attention for stable tokens and applies Quest-style sparsity to active tokens. |
| **DARE** | [2604.04215](https://arxiv.org/html/2604.04215) | — | — | — | — | — | **Not a sparse or caching method.** It is a dLLM post-training and RL framework. No sparse method named DARE was found (NOT_VERIFIED that none exists). |
| **FlashBlock** | ICML 2026 ([project](https://caesarhhh.github.io/FlashBlock/)); [2602.05305](https://arxiv.org/html/2602.05305v2) | [Yes](https://github.com/Caesarhhh/Flash_Block) | Trado-4B/8B (+ video) | GSM8K, MATH500, AIME, MBPP, HumanEval, LiveCodeBench, LiveBench | Dense Trado, SparseD | Token throughput up to 1.44×; attention time up to 1.6× (video). Implemented inside FlashAttention (A100) | **Medium; orthogonal.** Caches block-external attention output plus LSE across steps. This is an approximation, and it composes with sparsity. |
| **MAGE** | arXiv [2602.14209](https://arxiv.org/html/2602.14209) (2026-02; v2 2026-06) | Not provided | Fast-dLLM v2 7B, SDAR-8B, LLaDA2.0-mini | LongBench (6 tasks; F1, ROUGE-L, EditSim), NIAH; 32K–128K | Quest (page 16, re-estimated every step), SparseD (dense for the first 20%), exact FlashInfer | Attention 2.85–5.39× and **end-to-end TPOB up to 6.82× at 128K**, vs FlashInfer, on **H100** | **Highest relevance; port.** Exact attention at the first All-[MASK] step of each block, and the top-k indices are reused for the remaining steps. The selection kernels run on a side stream, costing +1.9–6.4% at step 1 (summary). |
| **PulseCol** | arXiv [2605.20813](https://arxiv.org/html/2605.20813) (2026-05) | Not provided | LLaDA-1.5, Dream | GSM8K, HumanEval, RULER-4K/8K; 4K–128K contexts | SparseD, sliding window, StreamingLLM, dense FA | Attention up to 10.54× (90% sparsity, 128K); **end-to-end up to 1.95× at 64K / 1,024 steps** (summary), on RTX PRO 6000 | High relevance to M3 (periodic refresh); port. |
| **Elastic-Cache** | ICLR 2026 per the [repo](https://github.com/VILA-Lab/Elastic-Cache); [2510.14973](https://arxiv.org/html/2510.14973) | Yes | LLaDA, LLaDA-1.5, Dream, LLaDA-V | GSM8K, MATH, HumanEval, MBPP (+ MathVista, MathVerse) | LLaDA no-cache, Fast-dLLM, dKV-Cache, DeepCache | End-to-end tokens/s, up to 45.1× vs no-cache (A100) | Not applicable (cache refresh policy). Orthogonal. |
| **d2Cache** | [2509.23094](https://arxiv.org/abs/2509.23094) (details from the prior note, NOT re-verified) | — | LLaDA-8B, Dream-7B | GSM8K, MBPP, HumanEval, MATH-500, GPQA, MMLU-Pro | dLLM-Cache, Fast-dLLM | tokens/s | Not applicable. |
| 2026: **Focus-dLLM** | [2602.02159](https://arxiv.org/html/2602.02159) (2026-02) | [Yes](https://github.com/Longxmas/Focus-dLLM) | UltraLLaDA, Dream | LongBench, NIAH; 8K–32K | Vanilla, Fast-dLLM, Sparse-dLLM, SparseD | End-to-end 29.6× vs vanilla at 32K; 2.05× vs Fast-dLLM (H200) | Low. Confidence-guided pruning of the no-cache context. |
| 2026: **DyLLM** | ICML 2026; [2603.08026](https://arxiv.org/abs/2603.08026) | NOT_VERIFIED | LLaDA, Dream | Reasoning and code | — | Throughput up to 9.6× | Low. Token-level partial recompute. |
| 2026: **BA-Att** | CVPR 2026 Findings (summary); [2605.19726](https://arxiv.org/abs/2605.19726) | NOT_VERIFIED | LM, MLLM and video | — | FlashAttention | Attention up to 6.95× vs FA | Medium (downsampled block selection). |
| 2026: **Prefilling-dLLM** | EMNLP 2026; [2606.10537](https://arxiv.org/abs/2606.10537) | [Yes](https://github.com/menik1126/Prefilling-dLLM) | dLLMs | LongBench, InfiniteBench | — | 9.1–28× at 8–32K | Low (chunk selection for the no-cache regime). |
| 2026: **HERALD** | [2606.21633](https://arxiv.org/abs/2606.21633) (same group as MAGE) | NOT_VERIFIED | Block DLMs | — | GPU-only serving | Decode throughput 2.28× (CPU offload) | Systems work, orthogonal. |
| 2026: **Affix Cache**, **Flash-dLLM** | [2608.26140](https://arxiv.org/abs/2608.26140); [2609.26796](https://arxiv.org/abs/2609.26796) ([code](https://github.com/VILA-Lab/Flash-dLLM)) | — | dLLMs | — | Elastic-Cache | End-to-end | Orthogonal (cross-request cache; IO-aware cache plus drafting). |
| Trained, not training-free: MRSA | [repo](https://github.com/NRHLYM/Mask-Routed-Sparse-Attention-for-Long-Context-Diffusion-Language-Models) | Yes | Dream-7B (trained) | — | NSA, DSA | — | Requires training. Out of scope. |

---

## 6. Comparison-protocol table for the closest papers

"Same kernel?" asks whether the sparse method is built by modifying the dense-baseline kernel (**Same**) or runs a different kernel or library (**Separate**). For cache methods the attention kernel is unchanged and fewer tokens are computed; these are marked **Same (fewer tokens)**.

| Paper | Dense baseline implementation (version) | Same kernel? | Reported speed level | Batch and sequence lengths | Quality benchmarks and metrics | Seeds or repeats | GPU |
|---|---|---|---|---|---|---|---|
| **SparseD** | "FlashAttention" (FA2 cited) inside the LLaDA/Dream model code; no KV cache | **Separate.** FA for the dense early steps, FlexAttention for the sparse steps | Whole-generation latency (s) on one RULER sample; speedup at 16K–64K and 128–1,024 steps | Latency: batch 1, 4K–64K. RULER quality: batch 16 (4K) or 8 (8K) | MMLU, GSM8K, HumanEval, RULER 4K/8K; accuracy | Not stated | A800 |
| **Sparse-dLLM** | Vanilla official LLaDA/Dream code (HF; LLaDA and Dream model code use `F.scaled_dot_product_attention`, is_causal=False: [LLaDA](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct/raw/main/modeling_llada.py), [Dream](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/raw/main/modeling_dream.py)) | Same (fewer tokens; PyTorch eviction) | End-to-end throughput (tokens/s), peak memory | Generation 256/512; context ≤ 4K; batch not stated | 7 tasks; accuracy via OpenCompass | Seed 2025; mean of 3 trials; efficiency averaged over 10 sampled instances | RTX 4090 |
| **dKV-Cache** | Vanilla HF LLaDA/Dream (no FA mention) | Same (fewer tokens), plus a custom `concat_reorder` op | End-to-end tokens/s | Batch 1 (larger batches in App. D); generation 8–512 | MMLU, GSM8K, MATH500, GPQA, HumanEval, MBPP | Not stated | A6000, H20 |
| **dLLM-Cache** | PyTorch + HF implementation | Same (fewer tokens) | TPS, FLOPs, latency (end-to-end) | Batch 8 in the main results, batch 1 in the comparison table | GSM8K, GPQA, MATH, MMLU(-pro), BBH, MBPP, HumanEval, LongBench | Not stated | RTX 4090 |
| **Fast-dLLM v1** | Vanilla LLaDA/Dream | Same (fewer tokens) | End-to-end throughput | Batch 1 (1–32 in App. C.5); generation 256–1,024 | GSM8K (5-shot), MATH (4-shot), HumanEval, MBPP; lm-eval | Not stated | A100 |
| **Elastic-Cache** | LLaDA without cache; Fast-dLLM | Same (fewer tokens) | End-to-end throughput to completion | Batch 1; generation 256/512/1,024 | GSM8K, MATH, HumanEval, MBPP (+ multimodal) | Not stated | A100 |
| **MAGE** | Exact attention in **FlashInfer** (version not stated) | **Same library** (all sparse kernels are FlashInfer) | Attention-layer time and **end-to-end TPOB** | 32K / 64K / 128K; KV budget k ∈ {256, 512, 1024}; batch not stated | LongBench (6 tasks: F1, ROUGE-L, EditSim), NIAH | Not stated | **H100** |
| **LoSA** | **FlashInfer** | Partly (FlashInfer plus custom CUDA/Triton) | Attention only | Batch 1; 32K / 64K | LongBench subsets plus 3 commonsense tasks; accuracy | Not stated | A6000, RTX 5090 |
| **PulseCol** | FlashAttention (version NOT_VERIFIED) | Separate (custom column-sparse tile kernel) | Attention and end-to-end | 4K–128K; 64–1,024 steps; batch not stated | GSM8K, HumanEval, RULER-4K/8K | Not stated | RTX PRO 6000 |
| **FlashBlock** | FlashAttention | **Same** ("implemented directly inside the FlashAttention kernel") | Token throughput, attention time | Batch 128 (primary); up to 800K tokens (video) | GSM8K, MATH500, AIME, MBPP, HumanEval, LCB, LiveBench | Not stated | 4×A100 |
| **Focus-dLLM** | FlashAttention | Separate (Triton sparse operator) | End-to-end vs vanilla | 8K–32K; batch not stated | LongBench, NIAH | Not stated | H200 |
| **BLASST** | **FlashAttention-3 BF16** ("All speedups are measured against FlashAttention-3 BF16 baselines"). Kernels ship in TRT-LLM fmha_v2/XQA (Hopper) and trtllm-gen (Blackwell) | **Same family** (in-kernel skip inside FA-style kernels; the FA3 denominator is a different code base) | Kernel prefill and decode, plus end-to-end TTFT/TPOT in TRT-LLM (about 1.1×) | Kernel: prefill batch 1 at 64K; decode batch 148 at 32K (B200) or 128 at 16K (H200). End-to-end: in-flight batching, concurrency 64 | RULER 4–128K, LongBench v1/v2, MATH500, AIME24, GPQA, LiveCodeBench | Timing runs not stated. Multiple samples per reasoning problem (summary; NOT_VERIFIED) | H200, B200 |
| **Quest** | "A normal attention implementation from the original **FlashInfer**" (2024; version not stated) | **Same library** ("dedicated CUDA kernels based on FlashInfer") | Kernel (self-attention up to 7.03×) and end-to-end decode latency per token (1.74× FP16) | Single batch; 32K (primary) up to 128K | PG19 perplexity, passkey 10K/100K, LongBench (6 sets) | Not stated | RTX 4090 (kernel), RTX 6000 Ada (end-to-end) |

**What the table shows.**
1. **The dense reference is always a FlashAttention-family or FlashInfer kernel, never eager attention.** The one exception is the cache-method line, whose "vanilla" is the HF model code with SDPA or FA and *no KV cache*.
   - The strongest sparse-attention papers (MAGE, Quest, FlashBlock, BLASST) measure dense and sparse **in the same library or kernel family**.
   - SparseD, PulseCol and Focus-dLLM compare a separate sparse kernel against FA.
2. **The speed level varies by paper.**
   - Cache methods report end-to-end tokens/s against a no-cache baseline. That regime does not apply to DiffusionGemma.
   - Sparse-attention papers report attention-only speedups plus one end-to-end number, at 32K–128K: MAGE's TPOB, PulseCol's end-to-end, SparseD's whole-generation latency.
   - BLASST alone reports the kernel-to-end-to-end compression honestly, about 1.1× end-to-end.
3. **Batch size.** Batch 1 dominates for dLLM sparse attention. Throughput papers use large batches.
4. **Quality is benchmarked two ways.** Long-context quality uses LongBench (MAGE, LoSA, Focus, BLASST), RULER (SparseD, PulseCol, BLASST) and NIAH. Short reasoning and code uses GSM8K, MATH, HumanEval and MBPP.
5. **Seeds and repeats are almost never reported.** Only Sparse-dLLM states a seed and 3 trials. No paper reports timing repeats or CIs.

---

## 7. Q4: training-free sparse attention for AR LLMs (candidate ported baselines)

| Method | Venue and code | Selection signal | Granularity | Training? | Dense baseline reported | Kernel vs end-to-end speed | Benchmarks and metrics | Runs on our hd512 bidirectional canvas on H100? |
|---|---|---|---|---|---|---|---|---|
| **BLASST / skip-softmax** | MLSys 2026 ([2512.12087](https://arxiv.org/html/2512.12087)); [TRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/latest/features/sparse-attention.html), [TRT-LLM blog](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.md), FlashInfer [PR #4991](https://github.com/flashinfer-ai/flashinfer/pull/4991) | In-kernel `local_max − running_max < ln λ`, with λ scaled by 1/L | KV tile | No (a threshold calibration fit) | FA3 BF16 | Kernel about 1.1–1.5×; end-to-end about 1.1× (see `blasst_and_eval_practice.md`). TRT-LLM blog: LongBench v2 on H200 at 70% sparsity, TTFT 16.5 s → about 13.9 s | RULER, LongBench v1/v2, MATH500, AIME24, GPQA, LCB | **No.** TRT-LLM: causal only, head dims 64/80/128/256. Port the rule. |
| **Quest** | ICML 2024 ([2406.10774](https://arxiv.org/html/2406.10774)); [code](https://github.com/mit-han-lab/Quest) | Per-page channel min/max of K; upper bound of q·k | Page (16) | No | FlashInfer | Kernel 7.03× (32K, budget 2,048); end-to-end decode 1.74× (FP16) | PG19 perplexity, passkey, LongBench | No (AR decode, Llama). Port (MAGE and LoSA did). |
| **MInference** | NeurIPS 2024 spotlight ([repo](https://github.com/microsoft/MInference)); [2407.02490](https://arxiv.org/html/2407.02490) | Offline per-head pattern (A-shape, vertical-slash, block) plus online estimate from the last 64 queries | 64-blocks, 1×64 verticals | No | FlashAttention-2 | Prefill latency up to 10× at 1M (A100). Index overhead matters below about 50K | InfiniteBench, RULER, PG-19, NIAH | Prefill-oriented; the head_dim of its kernels at 512 is NOT_VERIFIED. Low priority. |
| **XAttention** | ICML 2025 ([repo](https://github.com/mit-han-lab/x-attention)); [2503.16428](https://arxiv.org/html/2503.16428) | Antidiagonal block scoring plus a cumulative threshold | Block | No | "FlashAttention … within the FlashInfer framework" | Attention prefill up to 13.5× | RULER, LongBench, VideoMME, VBench | Port the scoring rule if needed. Low priority. |
| **FlexPrefill** | ICLR 2025 oral ([repo](https://github.com/bytedance/FlexPrefill)); [2502.20766](https://arxiv.org/html/2502.20766) | JS-divergence choice between query-aware and vertical-slash, then a cumulative-attention threshold γ | 128-block | No | FlashAttention | Attention-level (App. C/G) | RULER, InfiniteBench | Batch 1, bf16, Triton. Port the rule only. |
| **SpargeAttn** | ICML 2025 ([repo](https://github.com/thu-ml/SpargeAttn)); [2502.18137](https://arxiv.org/html/2502.18137) | Mean-pooled Q/K block prediction with a self-similarity filter, plus a second in-kernel online-softmax skip | 128×64 blocks | No (per-layer hyperparameter grid search) | FA2 (with the SageAttention base) | Kernel 2.5–5×; end-to-end e.g. 1.83× on Mochi | Language, image and video | The SageAttention base head dims are likely 64/128 (NOT_VERIFIED). Port. |
| **SeerAttention / SeerAttention-R** | [2410.13276](https://arxiv.org/html/2410.13276) ([repo](https://github.com/microsoft/SeerAttention)); venue NOT_VERIFIED | Learned AttnGate on pooled Q/K | 64-block | **Yes** (gate distillation; base frozen) | FA2 | Kernel 7.3× at 128K / 90%; end-to-end 1.41× on RULER | PG19, LongBench, RULER; R: AIME, GPQA, MATH-500 | Needs training. Out of scope for a training-free comparison. |
| **Twilight** | NeurIPS 2025 spotlight ([repo](https://github.com/tsinghua-ideal/Twilight)); [2502.02770](https://arxiv.org/html/2502.02770) | Top-p pruning after a base selector (Quest, DS), with 4-bit K estimates | Token | No | FlashInfer, FA2 | Kernel 2.4× vs FlashInfer; end-to-end decode 3.9× vs FlashInfer at batch 32–256 | LongBench, RULER, GSM8K, CoQA, PG-19 | Port (adaptive budget). |
| **RetrievalAttention** (now RetroInfer) | NeurIPS 2025 per the [repo](https://github.com/microsoft/RetrievalAttention) (venue mapping NOT_VERIFIED); [2409.10516](https://arxiv.org/html/2409.10516) | Attention-aware ANNS on CPU | Token | No (index build) | Exact KNN, ANNS, full attention | Decode 0.188 s/token at 128K (RTX 4090) | ∞-Bench, RULER, NIAH | CPU offload. Not comparable. |
| **DuoAttention** | ICLR 2025 ([repo](https://github.com/mit-han-lab/duo-attention)); [2410.10819](https://arxiv.org/html/2410.10819) | Per-head retrieval vs streaming split | Head | **Yes** (gate optimisation on synthetic data) | FA2, FlashInfer | Decode up to 2.18× (MHA), 1.50× (GQA) | NIAH, LongBench | Needs optimisation. Out of scope. |
| **ShadowKV** | ICML 2025 spotlight ([repo](https://github.com/bytedance/ShadowKV)); [2410.21465](https://arxiv.org/html/2410.21465) | Chunk-mean landmarks plus outliers; low-rank pre-RoPE K; V on CPU | 8-token chunk | No (per-sequence SVD) | Full attention | Decode throughput up to 3.04× (A100) | RULER, LongBench, NIAH | Offload plus low-rank K. Not comparable at batch 1 on H100. |
| **MagicPIG** | ICLR 2025 spotlight ([repo](https://github.com/Infini-AI-Lab/MagicPIG)); [2410.16179](https://arxiv.org/html/2410.16179) | LSH sampling (importance sampling) | Token | No | GPU-only attention | Throughput 1.5–5× | Various; <2% degradation | CPU. Llama only. Not comparable. |
| **RocketKV** | ICML 2025 ([repo](https://github.com/NVlabs/RocketKV)); [2502.14051](https://arxiv.org/html/2502.14051) | SnapKV++ eviction, then hybrid sparse attention (sequence and head reductions) | Page and token | No | Full KV (FA 2.6.3; gpt-fast) | End-to-end decode up to 3.7× (A100) | LongBench, NIAH, RULER | Port the HSA rule. It is also a TRT-LLM sparse algorithm. |
| 2026: **UNIQUE** | [2605.27740](https://arxiv.org/abs/2605.27740) | Page key mean plus std offset | Page | No (optional training) | FlashInfer, vLLM | Kernel 11.4×; end-to-end ≥ 5.3× | LongBench Pro, speech | Code NOT_VERIFIED. |
| 2026: **CoSA** | [2607.25291](https://arxiv.org/abs/2607.25291) (Tencent AngelSlim) | Kernel-aware proxy selects blocks, plus in-kernel online-softmax skipping | Page and block | No | NOT_VERIFIED | Attention 4.93×; TTFT 2.53× at 128K | Long-context | **Collision watch** (proxy plus in-kernel skip). |
| 2026: **PRR** | [2606.30389](https://arxiv.org/abs/2606.30389) ([code](https://github.com/Tianyu9748/Incremental_FlashAttention)) | EMA prediction from historical selections, plus an FA "repair" kernel using online-softmax statistics | KV block | No | NOT_VERIFIED | Up to 40% per-token latency | — | **Collision watch** (reuse of historical selections). |
| 2026: **S2O** | [2602.22575](https://arxiv.org/abs/2602.22575) | Online permutation plus early stop | Token-guided | No | FlashAttention | Attention 7.51×; end-to-end 3.81× at 128K | — | Code NOT_VERIFIED. |
| 2026: **FlashPrefill V2** | [2608.19758](https://arxiv.org/abs/2608.19758) | Max-based dynamic threshold | Block | No | FA2 and FA3/4-aligned | Kernel up to 27× (bf16) vs FA2; end-to-end vs FA3/4-aligned | — | Prefill. Integrated with SGLang. |
| 2026: **HiSparse** | [2608.07009](https://arxiv.org/abs/2608.07009) | Indexer-agnostic (DSA, NSA, Quest) | — | No | Full-HBM KV | Throughput 4.7× | — | Merged into SGLang. Memory management, not selection. |

---

## 8. Novelty and collision watch-list (per the project's evidence standard)

Our methods decide which 64-key tiles to skip:
- **M1:** historical QK scores plus a rank-32 projection of V;
- **M2:** block-mean V instead of the projection;
- **M3:** re-decides every R steps.

Prior work that a reviewer would map onto these components:
- **MAGE.** It selects once from exact attention at the first denoising step of each block and reuses the selection within the block. This is on block diffusion and on H100. It is the closest prior art to "historical QK scores, decided early, reused".
- **SparseD.** It uses a head-specific pattern computed after the dense early steps and reuses it across steps.
- **PulseCol.** It uses periodic refresh, which is the direct analogue of M3's re-deciding every R steps.
- **PRR.** It predicts blocks from historical selections.
- **Value-aware importance.** VATP ([2406.12335](https://arxiv.org/abs/2406.12335), EMNLP 2024) scores tokens by attention times ‖v‖₁. This is prior art for using V in the selection risk, relevant to M1 and M2.
- **CoSA.** It combines a proxy selection with in-kernel online-softmax skipping. This matches the "hybrid" idea listed in `blasst_and_eval_practice.md`.

The distinguishing claims must therefore rest on specific differences that the paper demonstrates, not on the general "reuse historical attention" idea:
- the V-projection risk bound;
- tile-level skipping under adaptive stopping, where the step count is not known in advance;
- operation on a native-cache block-diffusion model at hd512.

---

## 9. Q5: recommendation

### 9.1 Dense baseline for the 512-dim GLOBAL layers (H100, bf16, GQA 16/2, non-causal)

1. **Primary (strongest defensible): FlashAttention-4 (CuTe-DSL) from vLLM's fork, `vllm.vllm_flash_attn` with `fa_version=4`, SM90 hd512 path.**
   - Why it is the right reference:
     - It is the attention kernel that the DiffusionGemma technical report names for its official vLLM serving stack.
     - vLLM main selects it automatically on SM90 for `head_size > 256` and for diffusion models.
     - vLLM's own Gemma-4 measurement shows it beating `TRITON_ATTN` by 1.45× end to end. That measurement was autoregressive Gemma-4-31B with FP8 KV on H200, not our setting.
     - It belongs to the FA4 code base, which already carries block-sparse iteration. That makes a *same-kernel* sparse variant possible, though NOT_VERIFIED for SM90 at hd512.
   - Caveats:
     - It is forward-only.
     - It is a vLLM-fork kernel: upstream Dao-AILab has not merged SM90 hd512 (PR #2422 is open).
     - Its 64×64 tile might not be fast, so it must be **measured** on our shapes before being called "strongest":
       - q_len 256 (canvas) × kv_len 16K / 32K / 64K + 256;
       - Hq 16, Hkv 2, D 512, bf16, non-causal.
2. **Always report next to it:**
   - vLLM `TRITON_ATTN`. It is the backend in Google's developer-guide command and SGLang's only path. Label it "official default command", not "strongest".
   - PyTorch SDPA mem-efficient, the HF transformers reference path.
   - Our Triton 64-row kernel.
3. **Rule for our own kernel.**
   - Our Triton dense kernel may be the *same-kernel* reference for our sparse kernel. That follows the BLASST, Quest, MAGE and FlashBlock precedent.
   - Gains are attributed against the strongest dense kernel, following the project's evidence standard. If FA4-vLLM is faster than our Triton dense kernel, every headline speedup is quoted against FA4-vLLM.
   - The preferred alternative is to put the sparsity inside FA4 (block-sparse metadata or FlexAttention `BACKEND="FLASH"` on the fork), giving a same-kernel comparison on the strongest substrate.
4. **Not usable on H100 today:** FA2, FA3, upstream FA4, cuDNN (SDPA backend or shipped frontend), TRT-LLM (no DiffusionGemma support, and its MMHA cannot do H512), ThunderKittens. Say so with the links in §2, so a reviewer's "why not FlashAttention?" is answered.
5. **End-to-end substrate.** Where possible, measure request latency in vLLM with FA4, the official serving stack; the report's own figure of 13.56 ms per step at 4K on H100 FP8 is a sanity anchor. If our pipeline is HF-based, swap its GLOBAL attention to FA4-vLLM dense for the dense arm, so the per-forward baseline is not the mem-efficient SDPA path.

### 9.2 Sparse baselines a reviewer would expect (2026 diffusion-LLM sparse-attention paper)

| # | Baseline | Why a reviewer expects it | Run official code or port? |
|---|---|---|---|
| 1 | **SparseD** (ICLR 2026) | The canonical dLLM sparse-attention paper. It is cited as a baseline by MAGE, LoSA, PulseCol, FlashBlock and Focus-dLLM. | **Port the selection rule, labelled "SparseD-rule (port)".** The official code targets no-cache LLaDA and Dream with FA2 plus FlexAttention and cannot load DiffusionGemma. The port: dense attention for the first s₀ steps of each canvas (s₀ fixed, because the "20% of T" rule is undefined under adaptive stopping), then a head-specific top-ρ block pattern at our 64-tile, reused, with ρ swept. Our existing "fresh T" arm should be renamed or aligned to this. |
| 2 | **MAGE** (arXiv 2602.14209) | The closest setting: block diffusion with a prefix KV cache, long context, H100, and FlashInfer on both sides. It is also the closest prior art to M1/M3. | **Port (no official code), labelled "MAGE-rule (port)".** Per canvas, compute exact attention at the first denoising step, take the top-k KV tiles per KV head (budget k), and reuse them for all remaining steps. Report its selection overhead separately, as MAGE does. |
| 3 | **Quest** (ICML 2024) | The standard query-aware per-step baseline. MAGE and LoSA both adapt it to block diffusion. | **Port, labelled "Quest-rule (port)".** Per-step page min/max upper bound with page = 64 to match our tile (MAGE used 16), re-estimated every step. The official code is FlashInfer-2024, AR decode, causal, Llama. |
| 4 | **BLASST skip-softmax** (MLSys 2026) | The zero-selector-cost, in-kernel baseline, already discussed in the notes. | **Port the rule into the dense kernel used as baseline** (FA4-vLLM if feasible, otherwise our Triton kernel), labelled "BLASST-rule (port)". The official TRT-LLM and FlashInfer kernels are causal-only and support head_dim ≤ 256. Calibrate λ against context length as in their Algorithm 2. |
| 5 (optional) | **PulseCol** (May 2026) | The periodic-refresh analogue of M3. | Port (no code). Alternatively cite **FlashBlock** (ICML 2026, official code) as an orthogonal, composable cross-step cache, without making it a primary arm. |

**Not recommended as primary baselines.** Fast-dLLM v1/v2, dKV-Cache, dLLM-Cache, Elastic-Cache, d2Cache, DPad, Sparse-dLLM and Focus-dLLM:
- They accelerate a no-cache LLaDA/Dream regime.
- DiffusionGemma's encoder already commits every canvas to a KV cache, and its decoder reads that cache read-only.
- State this explicitly and cite them as orthogonal.

AR prefill methods (MInference, XAttention, FlexPrefill, SpargeAttn) are lower priority, because each step has only 256 canvas queries.

### 9.3 Recommended comparison protocol for our paper (derived from §6)

1. **Two dense references in every speed table:**
   - (a) the strongest official dense kernel: FA4-vLLM at hd512, measured;
   - (b) the same-kernel dense arm (our sparse kernel with keep-all).

   Report gains against (a). Report the (a)→(b) gap as a substrate fact, never as a contribution.
2. **Four speed levels, side by side**, following BLASST, MAGE and PulseCol:
   - GLOBAL-attention kernel time per step;
   - per-forward (denoising-step) time on common states;
   - **per-canvas time**, the analogue of MAGE's TPOB;
   - whole-request latency and tokens/s under **native adaptive stopping**, with steps per canvas and total calls reported.

   Include the selector cost as its own line.
3. **Contexts and batch.** 16K, 32K and 64K, plus 128K if memory allows (SparseD, MAGE, LoSA and PulseCol use 32K–128K). Batch 1 as primary, which is what all dLLM sparse papers use; optionally batch 4, the vLLM recipe's max-num-seqs.
4. **Quality.**
   - LongBench subsets (MAGE and LoSA use HotpotQA, NarrativeQA, Qasper, QMSum, RepoBench-P, TriviaQA; metrics F1, ROUGE-L and EditSim).
   - RULER at 16K–64K.
   - NIAH.
   - A short-reasoning no-regression set: GSM8K, MATH-500 and HumanEval, plus our AIME constraint.
   - An accuracy-vs-budget sweep: k ∈ {256, 512, 1024} tokens, or ρ.
5. **Statistics beyond the literature norm.** Most papers state none.
   - Fixed seeds with ≥ 3 generation trials, following Sparse-dLLM's precedent.
   - Timing with warm-up and ≥ N paired repeats, reported with paired CIs.
   - Raw receipts showing that the realised sparsity actually exercised the sparse path.
6. **Hardware and precision.** H100 80GB in bf16, matching MAGE's H100. State that the official DiffusionGemma speed claims use FP8, and are therefore not directly comparable.

---

## 10. Open items (NOT_VERIFIED)

1. Whether FA4-vLLM hd512 on SM90 supports FA4 block-sparse metadata or FlexAttention `BACKEND="FLASH"`.
2. Which backend our pinned vLLM build actually selects. Check the log for "upgrading FlashAttention 3 -> 4" versus "forcing TRITON_ATTN".
3. FlashInfer's FA3-template (`backend="fa3"`) support for hd512, and its measured bf16 performance on H100.
4. The shipped cuDNN (backend 9.2x) status for D512 on SM90. cuDNN docs 9.18.1 say d ≤ 256, but the FROST PR compares against a cuDNN 9.24.0 D512 number.
5. The current mslk head-dim limits for the xFormers cutlass op.
6. Which attention kernel Uno used for DiffusionGemma in Nano-vLLM.
7. The venues of SeerAttention, DPad and HERALD, and the code availability of MAGE, LoSA, PulseCol, UNIQUE and S2O (the papers list none).
8. Whether a dLLM sparse or caching method named "DARE" exists. Only a post-training framework was found.
9. BLASST's number of timing repeats and reasoning samples.
10. PulseCol's end-to-end 1.95× figure and MAGE's selection-overhead figures were read through the summarising fetch tool.

Detailed sources are linked inline in each table.
