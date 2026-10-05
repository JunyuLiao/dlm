# I-DLM official algorithm environment smoke

The pinned I-DLM bundled SGLang `IDLMBlockN` path completed three toy generations on one H100 80GB with exit code 0. This establishes loading, compilation, and execution only. All generations reached the requested 64-token length limit, and only one of three toy answer checks passed; task quality is not qualified. No raw prompt, output, or expected answer is published.

The official configuration uses block size 7, generation block size 4, confidence threshold 0, temperature 1, top-k 50, top-p 0.95, and speculative verification. Request parameters used seed 0, temperature 1, top-p 1, top-k -1, and maximum 64 new tokens. Precedence between algorithm and request sampling has not been verified. No M3 sparse variant was applied.

The run reserved 187.207282 GPU seconds from idle confirmation through complete worker exit, including imports, loading, JIT, graph capture, and teardown. The first generation included 135.445704 seconds of initial JIT/execution; subsequent timings are not benchmark evidence. Output lengths differ from the LLaDA smoke, so their latencies must not be compared. The GPU was released to zero used MiB and zero utilization. Forward counts are unavailable because the official generation metadata has no verified denoising-forward counter.

Environment versions: bundled SGLang 0.0.0.dev0, torch 2.9.1+cu128, FlashInfer 0.6.3, transformers 4.57.1, sgl-kernel 0.3.21. The independent official CUDA toolkit closure is nvcc 12.8.93, runtime 12.8.90, CCCL 12.8.90, and NVRTC 12.8.93. Archive SHA-256 checks, pip dependency checks, CPU algorithm import, and cooperative-groups/BLAS/BF16 compilation passed. Model weight shards were checked for ordinary-file structure and parseable safetensors headers.

The fixed source has a Python 3.10 typing annotation import error. Its isolated clone received exactly one `from __future__ import annotations` line to postpone annotation evaluation. No algorithm logic changed; original/new file hashes and the source pin are recorded in `summary.json`.

Sources: [I-DLM pinned source](https://github.com/Introspective-Diffusion/I-DLM/tree/a23c1a12ef997c7f3ad616b25bcfb62db39ded68), [public I-DLM weights](https://huggingface.co/yifanyu/I-DLM-8B/tree/3cecd8cd39b9b32486b20aa80f28e81206387720), [NVIDIA CUDA 12.8.1 redistribution manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json).
