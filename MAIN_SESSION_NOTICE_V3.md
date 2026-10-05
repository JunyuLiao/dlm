# Main-session notice: v3 phase-aware calibration

The temporal+Gaussian-32 calibration rule was updated for this run.

- Separate local/global thresholds are calibrated for call 1, call 2, and later calls.
- Each phase targets the requested whole/local/global physical sparsity within ±2 percentage points when feasible.
- Early protection is an ordering constraint (`threshold(call1/call2) <= threshold(late)`), not a fixed early sparsity ceiling.
- Complete trajectories are rerun after threshold changes; no histogram interpolation is used.
- Unattainable policies retain their measured operating point and are not relabelled as the requested target.

Run root: `query_adaptive_aime_temporal_v9`.
Targets: 40% and 70%; seeds: 42, 43, 44; method: temporal + Gaussian-32.

Handoff status at 2026-09-26 16:34 UTC: the detached calibration worker is no
longer running after 36 calibration shard files; no threshold files,
calibration-complete marker, or final shards have been written yet. Resume with
the same environment and root (the run is resumable):

```bash
AIME_TARGETS=40,70 AIME_METHODS=temporal \
AIME_PRIOR_ROOT=/home/exouser/ljy/dlm/results/query_adaptive_aime_temporal_v8 \
AIME_FINAL_SEEDS=42,43,44 PYTHONPATH=src:. \
python -m experiments.value_direction_hopper.aime_temporal_sweep \
launch-calibrate --root /home/exouser/ljy/dlm/results/query_adaptive_aime_temporal_v9
```
