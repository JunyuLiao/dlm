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

Threshold selection maximizes physical sparsity subject to a configurable
masked-position disagreement limit relative to lambda zero at every noise
level. This measures fidelity to dense attention rather than the model's
ability to reconstruct inherently ambiguous hidden text.

With the default 5% disagreement limit, the selected threshold is
`lambda=0.001`. It yields 9.10% row-level sparsity and 0.049% physically
skippable tiles. Agreement with dense is 99.76%, 98.65%, and 96.32% at 15%,
50%, and 90% masking. `lambda=0.003` fails because agreement at 90% masking
falls to 93.22%.

Reproduce the calibration:

```bash
conda run -n ljy_dlm python scripts/llada_blasst_calibrate.py \
  --context-length 4096 --num-contexts 2 \
  --mask-ratios 0.15,0.5,0.9 \
  --lambdas 0,0.001,0.003,0.01,0.03,0.1,0.3,1.0
```

Here `lambda=0` is a separate exact no-pruning control. The remaining values
are readable, approximately logarithmic values from `1e-3` through `1`.

Ground-truth recovery at `lambda=0` is not expected to be 100%: it asks whether
one denoising forward reconstructs inherently ambiguous hidden text. The
primary relative metric instead compares predictions with lambda zero, whose
self-agreement is 100%.

The output is `blasst_calibration_4096.json`. Masked-token recovery is a much
better calibration signal than whole-sequence dense-logit agreement, but final
validation should additionally measure iterative generation on a task dataset.

The unfused PyTorch path is for correctness and calibration, not performance.
Only a CUDA-kernel integration can turn physically skipped tiles into speedup.
