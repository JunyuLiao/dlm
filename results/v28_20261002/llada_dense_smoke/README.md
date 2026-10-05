# LLaDA official dense environment smoke

The official SGLang JointThreshold path completed three toy generations on one H100 80GB, with exit code 0. Each generated six tokens and contained the expected arithmetic answer. This is environment qualification only; it provides no benchmark accuracy, speedup, or sparse-equivalence claim. Forward counts are unavailable because the official generation metadata has no verified denoising-forward counter.

Sampling was seed 0, temperature 1, top-p 1, top-k -1, maximum 64 new tokens, one running request, and FlashInfer attention. No raw prompt or generated output is published.

The successful attempt reserved 88.199930 GPU seconds, including imports, loading, JIT, graph capture, generation, and teardown. All six attempts reserved 345.549518 GPU seconds; five failed or were stopped and remain in the accounting. The GPU was released to zero used MiB and zero utilization after completion. See `summary.json` for every exit code and reservation duration.

The toolchain uses SGLang 0.5.21, torch 2.13.0+cu130, FlashInfer 0.6.18, FlashInfer JIT cache 0.6.18+cu130, ninja 1.13.2, nvcc/CRT/NVVM 13.0.88, CUDA runtime headers 13.0.96, and CCCL 13.0.85. NVIDIA official archive SHA-256 checks passed. CPU qualification covered BF16/runtime/CCCL, cooperative groups and cublasLt, and the actual FlashInfer source that had failed. The pip compiler target had pulled newer runtime headers into its implicit include directory; the final compiler wrapper uses the matched official compiler and runtime headers.

Model: [inclusionAI/LLaDA2.1-mini](https://huggingface.co/inclusionAI/LLaDA2.1-mini), revision `20e64e2ad21644d0e5248586ed9c942cdd45de0f`. Toolchain archive metadata: [NVIDIA CUDA 13.0.2 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.0.2.json).
