# FDFO final-commit profiling experiment

This benchmark pre-validates a specialized final KV-cache commit path without
changing FDFO scheduling, token selection, or model outputs. Profiling is
injected into child SGLang processes only when `SGLANG_FDFO_PROFILE_DIR` is set
by the runner.

Use the same isolated SGLang 0.5.16 environment as
`benchmarks/sglang_fdfo`:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/sglang_fdfo_commit/run_experiment.py \
  --output-dir results/systems/sglang_fdfo/commit
```

The default run uses LLaDA2.1-mini at its pinned revision and sweeps block
sizes 8/16/32/64, exact context lengths 128/1024/4096, and request counts
1/4/16 twice in eager mode. It also measures HBM bandwidth and repeats the
block-32, context-1024 cells with CUDA graphs. Raw CUDA-event, scheduler,
workload, server, and `nvidia-smi` telemetry artifacts are retained.
The analysis and Markdown report are generated automatically after the sweep.

Re-run only the analysis for a completed timestamped directory with:

```bash
python benchmarks/sglang_fdfo_commit/analyze.py \
  results/systems/sglang_fdfo/commit/YYYYMMDD_HHMMSS
```

This creates `iterations.csv`, `breakdown.csv`, `shape_sweeps.csv`,
`commit_normal_ratios.csv`, `mixed_batches.csv`, `scheduler.csv`,
`opportunity.json`, and `report.md`.

For an Nsight Systems capture, add `--nvtx` and wrap the runner (or a reduced
single-cell invocation) with `nsys profile`. NVTX ranges include the complete
FDFO iteration, model forward, token-selection step, attention backend,
transformer blocks, logits processing, and paged KV writes. Nsight is optional;
the main report uses CUDA events.
