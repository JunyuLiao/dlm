# Fused observation BF16 correctness

Frozen source `54163f4e7c8766aaa4187807420b3cf3167dba81`: all44 tests passed on one idle H100. CUDA-hidden collection passed first, and the GPU process exited0. Full process-reserved GPU time was32.737210 seconds; pytest reported31.90 seconds. These durations document resource use, not kernel speed.

Four original default-tf32x3 STORE comparisons were retained. Forty new main-BF16 tests cover GQA16/2, head512, Q256, rank32, splits1/2, prefix0/31/65/844/1793, scales512**-0.5 and official GLOBAL1.0, and dense-output/observe-only modes. The independent eager FP32 oracle disables TF32 and checks all tail scores, full attention output, all prefix-tile log masses and rank32 means, plus finite-input flags. Existing tolerances were frozen before execution and unchanged. See the protocol in `docs/V29_FUSED_OBSERVATION_AUDIT_20261002.md`.

Attempt001 stopped at CPU preflight because pytest was missing, with0 GPU launches/seconds. Its frozen24case source and logs are retained. Attempt002 used a separate own runner pytest9.1.1 target, Torch2.13.0+cu130, Triton3.7.1, vLLM0.30.0 and CUDA13.0; qualified environment packages were unchanged. Only six exact committed source/test files were deployed, and all caches/logs remain private and isolated. No model, prompt, gold or generated output was transferred.

This is scalar/geometry numerical qualification on synthetic random tensors, not real-model QKNorm-distribution quality or bit/trajectory equivalence. The low-level primitive handles zero wholly-prefix tiles; generic core fused bootstrap still rejects prefix<64 unless an earlier configured gate bypasses it. No speed or end-to-end quality conclusion follows.
