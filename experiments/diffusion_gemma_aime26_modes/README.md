# Paper-aligned AIME2026 BLASST comparison

This experiment is explicitly **not an exact reproduction** of
[DiffusionGemma Table 3/4](https://arxiv.org/html/2608.00146v1).
The public sources checked did not specify exact AIME prompts, output cap or
repetition/seeding protocol. User approved a 32,768-output-token safety cap,
seed42, one generation per question, and a custom fixed five-shot extension.

All30 AIME2026 questions from cached math-ai/aime26 revision
79037aebdb6580008fb960d17cb21fd3099083e3; both native thinking settings;
both zero-shot and fixed five-shot AIME2025 worked demonstrations. The five
demonstrations and six disjoint calibration problems are reused from the prior
AIME30 manifest, not selected using AIME2026 performance. No final problem is
used to select thresholds. All four groups have dense + BLASST25/50/75/90:
600 final generations, plus smoke and 24dense/96sparse calibration generations.

Reuse the original64×64 physical_tile_v1 BLASST implementation, lambda<=1,
prefix+canvas eligible, separately calibrated local/global scalars per group.
The earlier unrelated AIME26 aggressive lambda>1/128×64 experiment is not reused.
The original scalar physical-margin calibration grid, bisection and six-problem
sparse verification are reused; no refitting from test labels. Targets beyond
lambda1 ceilings deploy lambda1 and remain labelled unattainable.

Pinned BF16 model revision and native sampler match the prior run. The paper
sampler settings match; the optimized FP8 performance implementation is not
reproduced. No timing is presented as sparse-kernel speedup. Dense statistics
count structurally eligible physical tiles with zero skipped; no calibration margins are
retained for final dense samples. Calibration alone collects compact physical
margin arrays, without saving full attention matrices.

Scoring ignores the thought channel and grades only the emitted final response.
An unfinished thought is a failure even if it contains the reference number.
The existing flexible numeric extraction is reused. Budget stops, missing final
answers, raw token IDs and raw text including channel separators are saved.
Reports use sums of physical tile counters, not averages of sample sparsities.

Run from repository root in ljy_dlm with PYTHONPATH=src:.:

```
python -m experiments.diffusion_gemma_aime26_modes.run prepare
python -m experiments.diffusion_gemma_aime26_modes.run run
python -m experiments.diffusion_gemma_aime26_modes.run report
```

Default output: results/diffusion_gemma_aime26_blasst_thinking_shots/.
Immutable fingerprints prevent mixing changed code or settings. Completed shards
resume without inference. CUDA smoke checks plain-vs-instrumented dense token
parity in all four groups, valid local/global and prefix/canvas tile coverage,
finite attention and actual skipped tiles. Launch via a systemd user service;
the inherited monitor_controlled.py logs progress every900seconds independently.
Individual final sparse errors are saved and independent samples continue.
