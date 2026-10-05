# Query-adaptive RULER4K reproduction

Run from `/home/exouser/ljy/dlm` in the `ljy_dlm` Python environment on one H100.

```bash
export PYTHONPATH=src:.
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=4

# The v4 kernel and matching ATen bridge are already built at the paths in
# configs/configuration.json. Rebuilding is only needed after source changes.
python -m experiments.value_direction_hopper.query_adaptive_study run \
  --root /home/exouser/ljy/dlm/results/query_adaptive_v3

# Regenerate final tables/plots from completed final shards only.
python -m experiments.value_direction_hopper.query_adaptive_study report \
  --root /home/exouser/ljy/dlm/results/query_adaptive_v3

# After the primary matrix, run the separate small same-state diagnostic.
python -m experiments.value_direction_hopper.query_adaptive_replay \
  --root /home/exouser/ljy/dlm/results/query_adaptive_v3 --limit 1  # smoke
python -m experiments.value_direction_hopper.query_adaptive_replay \
  --root /home/exouser/ljy/dlm/results/query_adaptive_v3

# If the predeclared selection gate is met, confirm with seeds 43 and 44.
python -m experiments.value_direction_hopper.query_adaptive_confirmation \
  --root /home/exouser/ljy/dlm/results/query_adaptive_v3

# Separate matched-warmup, interleaved, untraced runtime benchmark.
python -m experiments.value_direction_hopper.query_adaptive_timing \
  --root /home/exouser/ljy/dlm/results/query_adaptive_v3 --limit 1 --repeats 1  # smoke
python -m experiments.value_direction_hopper.query_adaptive_timing \
  --root /home/exouser/ljy/dlm/results/query_adaptive_v3
```

The worker writes one shard per example/condition and checks immutable source,
manifest, model, and policy provenance on resume. `progress.md` is a lightweight
15-minute heartbeat. Diagnostic replays and timing use the same frozen routing
thresholds but never overwrite the primary generation shards.

The v1 and v2 directories are preserved failed attempts. v1 stopped on strict
JSON serialization of an internal all-retained `−∞` threshold; v2 smoke passed
but its detached continuation stopped at JSON-normalization in the frozen-config
comparison. Both were corrected before launching v3.
