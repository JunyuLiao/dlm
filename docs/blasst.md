# BLASST sparse attention experiment

This branch implements BLASST Algorithm 1 in a tiled online-softmax reference
path. It intentionally omits pipeline, warp, batched-load, and architecture-
specific optimizations.

## LLaDA calibration design

The diffusion-specific calibration uses:

- LLaDA's native 4096-token context limit;
- two heterogeneous natural-text windows rather than a repeated phrase;
- 15%, 50%, and 90% random masking to represent different denoising stages;
- relative accuracy measured as top-1 agreement with the lambda-zero dense
  algorithm path, only at masked positions;
- ground-truth recovery retained only as a model-capability diagnostic;
- FlashAttention 2.8's reverse KV traversal and 128-by-64 tile shape;
- separate row-decision and physically skippable tile counts.

The lambda-zero pass records the physical score
`R_j = max_i exp(local_max_ij - running_max_i)` for every 2D attention tile.
For each remaining-mask-ratio bucket and target physical sparsity `s`, the
calibrator proposes `lambda_s = quantile_s({R_j})`. It then selects the largest
proposed lambda in each denoising bucket that satisfies the configurable
masked-position disagreement limit. This calibrates realizable kernel
sparsity directly instead of using row sparsity as a proxy.

Reproduce the calibration:

```bash
conda run -n ljy_dlm python scripts/llada_blasst_calibrate.py \
  --context-length 4096 --num-contexts 2 \
  --mask-ratios 0.15,0.5,0.9 \
  --target-physical-sparsities 0.05,0.1,0.2,0.3,0.4,0.5
```

Here `lambda=0` is a separate exact no-pruning control. All nonzero thresholds
are derived from the observed physical-score quantiles rather than a fixed
lambda sweep.

Ground-truth recovery at `lambda=0` is not expected to be 100%: it asks whether
one denoising forward reconstructs inherently ambiguous hidden text. The
primary relative metric instead compares predictions with lambda zero, whose
self-agreement is 100%.

The output is `blasst_calibration_4096.json`. Masked-token recovery is a much
better calibration signal than whole-sequence dense-logit agreement, but final
validation should additionally measure iterative generation on a task dataset.

The unfused PyTorch path is for correctness and calibration, not performance.
Only a CUDA-kernel integration can turn physically skipped tiles into speedup.
