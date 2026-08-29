# RULER experiment scripts

Scripts are grouped by role. Run them from the repository root unless a script
states otherwise.

| Directory | Purpose | Typical results |
|---|---|---|
| [`evaluation/`](evaluation/) | Manifest preparation, single evaluations, adapter wrappers, and H100 smoke checks | `results/ruler/` or an explicit `OUTPUT_DIR` |
| [`blasst/`](blasst/) | BLASST calibration, context/λ sweeps, DiffusionGemma reports, and Gemma 4 AR validation | `results/blasst/` |
| [`kv_pruning/`](kv_pruning/) | Quantile/block-maximum pruning runs, reports, and random baselines | `results/kv_pruning/` |
| [`diagnostics/`](diagnostics/) | Attention-distribution collection and balanced manifest utilities | `results/attention/` |

Model-specific legacy/reference programs remain in `fast_dllm_v2/scripts/` so
their imports and paper-reproduction commands stay self-contained.
