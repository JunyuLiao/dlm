# DiffusionGemma denoising-attention analysis

This experiment observes the native dense DiffusionGemma attention path. It
intercepts the registered attention call after q/k normalization and RoPE,
applies the native scale and mask to diagnostic QK scores, and returns the
original SDPA result unchanged.

Smoke run on one H100:

```bash
PYTHONPATH=src /home/exouser/miniconda3/envs/ljy_dlm/bin/python -m \
  experiments.diffusion_attention_analysis.run_analysis \
  --output-dir results/attention/analysis/main \
  --num-samples 4 --max-new-tokens 64 --steps 8
```

The default explicit layer sample (`0,5,12,17,24,29`) covers local and global
attention at early, middle, and late depths. An empty `--heads` selection means
all heads. Use `--layers` and `--heads` to record any other explicit sample.

Offline regeneration of tables, plots, gates, and `report.md`:

```bash
PYTHONPATH=src /home/exouser/miniconda3/envs/ljy_dlm/bin/python -m \
  experiments.diffusion_attention_analysis.analyze \
  results/attention/analysis/main
```

For a balanced profiling subset of an existing cached RULER manifest, add
`--prompts-jsonl PATH --balanced-by-task --num-samples 32`. The runner preserves
each selected row's exact prompt, `inference_seed`, and `tokens_to_generate`.

`--replay-mask`, `--refresh-interval`, and `--uncertainty-band` keep the
observation-run configuration stable, but `run_analysis` rejects live replay.
After the Phase 3 gate passes, replay is a separate, explicitly gated command:

```bash
PYTHONPATH=src /home/exouser/miniconda3/envs/ljy_dlm/bin/python -m \
  experiments.diffusion_attention_analysis.run_replay \
  --observation-dir results/attention/analysis/main \
  --output-dir results/attention/analysis/main/replay \
  --prompts-jsonl PATH --ruler-root /tmp/NVIDIA-RULER \
  --num-samples 100 --balanced-by-task --steps 8 \
  --threshold-mode empirical_quantile --target-density 0.75 \
  --proxies max --no-eager-dense
```

Run a 2–4 prompt smoke with eager dense enabled before any validation run.
The canonical result used 32 observation prompts and 100 held-out replay
prompts. Its replay configuration is a scientific emulation: it reports
theoretical routing work avoided and never claims kernel or wall-clock speedup.
