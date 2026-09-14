# DiffusionGemma oracle block-selection study

Run from the repository root with the existing `ljy_dlm` environment:

```bash
export PYTHONPATH=src
python -m experiments.diffusion_gemma_oracle.run prepare
python -m experiments.diffusion_gemma_oracle.run smoke
python -m experiments.diffusion_gemma_oracle.run run
python -m experiments.diffusion_gemma_oracle.run report
```

`--output` chooses an isolated bundle; `run --conditions NAME ...` resumes selected
configurations. Completed shards are fingerprinted; mismatched code is rejected.
The CUDA smoke must pass before the sweep. Individual sample errors are persisted
and independent samples continue. Rerunning retries only missing shards.

The default bundle is `results/diffusion_gemma_oracle`. It contains the exact sample
manifest and cached baseline hashes in `setup.json`, a smoke audit, raw per-sample
generations, summed per-step/layer/head statistics, representative block rankings,
CSV/JSON summaries, ranking overlays, and scientific tradeoff figures.

There are 16 prompts (10 LongBench, 6 AIME) and 26 new conditions: a dense observer,
19 top-k configurations (10/25/50/75/90% for QK, mass, contribution; 25/50/75/90%
for BLASST margin ranking), and six top-p configurations (90/95/99% cumulative
mass or contribution). The existing nine dense/Sol/BLASST conditions are reused
for exactly these prompts. No new calibration is performed.

Top-k uses the identical integer budget per head/query block for every signal.
Top-p intentionally allows the budget to vary; comparisons must use measured
physical sparsity. QK uses maximum valid token score, mass uses summed normalized
probability, contribution uses `||P_tile V_tile||_F`, and BLASST uses maximum
row-wise running-max margin. Tiles with no valid structural pairs are excluded.
Nonempty support rescues are counted. Prefix and canvas are both eligible.

Dense counterfactual diagnostics sample the first call of every layer at steps
0/1/4/12/24/47 if reached. All heads and Q blocks are measured; rank arrays retain
the first Q block. Sparse rollout statistics cover all attention calls. Output
error uses FP32 same-state P/V and exact renormalization, before output projection.
It includes multi-tile interactions; contribution norm alone does not.

The current models are *stronger ranking heuristics*, not a downstream optimal
oracle. A failure cannot establish that no safe sparse subset exists. Small-subset
accuracy preservation is exploratory, not proof of statistical equivalence.
Dynamic allocation is motivated only if the observed heterogeneity supports it;
this study does not claim measured dynamic-policy accuracy or kernel speedup.

During a long run, check the actual process and progress approximately every
30 minutes. A progress file alone is not proof the process remains alive.
