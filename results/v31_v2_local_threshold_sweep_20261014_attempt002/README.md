# V2 local threshold sweep

Scope amendment, 2026-10-14: finish the remaining LongBench FULL dense,
PIECEWISE dense, and all-kept FA4 controls, then pause before any full
LongBench V2 threshold arm. The five initial LongBench thresholds and both
finer thresholds across all three workloads are deferred by the user.
The original frozen protocol is retained; the amendment is recorded in
`scope_amendment_controls_only_20261014.json`. An independent boundary monitor
stops the coordinator after the last control's complete official scoring.
Do not execute the resume command below until the user authorizes resumption.
`controls_complete.json` will certify the control panel only;
`PAUSED_USER_REQUEST.json` will preserve the remaining campaign state.

Five initial uniform GLOBAL thresholds: 0.018, 0.0195, 0.021, 0.0225, 0.024. Two adjacent midpoints are selected after the complete stage-1 grid. Workloads: official 503-item LongBench-v2 (seed 1), 30 AIME26 problems (seeds 42/43/44), deterministic 100-item MATH500 subset (sampling seed 20261011, generation seed 1).

This is exploratory accuracy tuning on the evaluation panels. Fresh V31, fixed FULL dense, PIECEWISE dense, and all-kept FA4 controls are source/runtime matched. Native decoding and V2 formula are unchanged. Clean timing remains paused. All prompts, gold, tokens, and completions are private.

Sampling detail: the inherited MATH500 builder uses `Random(base_seed + sample_count)` on the source sorted by unique_id. For base 20261011 and 100 samples, the effective sampling seed is 20261111. All sampled prompts and gold answers were independently verified against the 500-item source; see `receipts/math500_source_integrity.json`.

Progress: `state.json`, `coordinator.log`, and the 30-minute `monitor.jsonl`. Partial per-workload curves are published to `curve.csv` and `summary.md` after a complete arm is officially scored. Resume with the committed entrypoint `python scripts/v31_v2_local_threshold_sweep.py --stage run` under the pinned runtime; complete shards are validated and skipped, while interrupted attempts are preserved and rerun in new directories.
