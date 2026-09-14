# DiffusionGemma Sol-Attn MATH500 region experiment

This isolated, resumable experiment compares universal-Gaussian 64×64
Sol-Attn routing when only prefix tiles are skippable and when prefix and
canvas tiles share one candidate population. It runs 50 paired MATH500
problems at target skipped-tile fractions 25%, 50%, 75%, and 90%, plus dense.
The dense condition uses the same eager reference attention backend as the
sparse conditions, with masking disabled, so deltas isolate routing.

The primary sparsity is count-weighted across every denoising call:
`sum(skipped physical tiles) / sum(all structurally valid physical tiles)`.
The report also exposes sparsity within the routed region and separate local,
global, prefix, and canvas counters.

```bash
PYTHONPATH=src:. python -m experiments.diffusion_gemma_solattn_math500 prepare \
  --nemo-gym-root /tmp/nemo-gym-math500
PYTHONPATH=src:. HF_HUB_OFFLINE=1 python -m experiments.diffusion_gemma_solattn_math500 smoke
PYTHONPATH=src:. HF_HUB_OFFLINE=1 python -m experiments.diffusion_gemma_solattn_math500 run
PYTHONPATH=src:. python -m experiments.diffusion_gemma_solattn_math500 report
```

Predictions are fsync'd after each problem. Re-running `run` resumes each
condition by request ID after validating its run fingerprint.
