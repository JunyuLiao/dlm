# RULER8K version14: active experiment

Resumable130-question evaluation (10/task,13 tasks),26 disjoint calibration questions,13 development questions. Nine conditions: dense, plus BLASST/mass-only/full-dimensional centered/Gaussian2 centered at50% and75% target physical sparsity. Pinned DiffusionGemma,128×64 physical tiles, prefix+canvas eligible, seed42, projection seed1729 and official RULER task-specific output budgets.

BLASST defaults to λ≤1. Only failure of the joint constant λlocal=λglobal=1 calibration run to reach the requested **whole-model** sparsity permits above-one thresholds. A local-only shortfall does not unlock aggressive mode. Existing inverse-valid-KV-length calibration and routing are preserved. See `setup.json` and `execution_contract.json` for frozen settings and provenance.

Version13 failed at smoke bookkeeping after its first dense smoke, before any calibration or final runs. All its files remain preserved. Version14 reloads completed fresh shards to expose the same lossless metadata as resumed shards. No numerical algorithm/kernel changed. All19 CPU/CUDA tests passed, including fresh/cache-hit equivalence and synthetic1170-output report regeneration.

From `/home/exouser/ljy/dlm`, launch or resume only when the prior worker is no longer running:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=src:. OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false /home/exouser/miniconda3/envs/ljy_dlm/bin/python -m experiments.diffusion_gemma_ruler8k_jl_v14 launch
```

After completion, `report` and `verify` subcommands regenerate solely from completed raw shards, with `CUDA_VISIBLE_DEVICES=''`. Do not use the version13 entry point on this bundle. The worker performs both automatically after the sweep. A complete result requires1170 outputs, full diagnostic coverage and a matching independent regeneration audit.

Monitor `job.json`, `phase.json`, `progress.json`, `completed.jsonl`, `run.log`, and `supervisor_terminal.json`. The requested agent/supervisor health-check interval is30 minutes. Failed independent configurations are preserved while other configurations continue.

This is an accuracy/physical-sparsity study, not a hardware-speedup benchmark. The650-example source pool was previously used; the new disjoint calibration/final split is not fresh held-out confirmation.
