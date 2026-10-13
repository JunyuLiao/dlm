# V31 refresh factor isolation — 2026-10-14

The user requested factor isolation because the 64-problem LongBench-v2 subset is noisy. This study analyzes validated existing native cells and completes operator replay for raw/stable × progress dense on/off × cheap intervals 1/2/4/8. No new native generation is authorized in this step; choose 503-problem arms afterward. The previous development promotion decision remains historical and does not constrain the new selection exercise.

The source study is `results/v31_cached_refresh_stable_20261013/`, complete at commit `a136b40d`. Its result leaves and frozen specifications remain untouched. The selector parameters are frozen from its operator-only choice: zero hysteresis, persistence 2, 2 swaps/event, 8 swaps/anchor epoch, removal guard 0.025. Raw uses unrestricted hybrid top-K and does not activate those constraints. Both retain K=128 prefix tiles per query head/Q128 block, capped for short prefixes, with mandatory tail outside the budget.

Only 8 of the 16 factorial settings have native development results. No matched progress-on/off comparison exists with cheap refresh enabled. The two held-map controls provide a progress comparison at interval zero. Existing native contrasts are exploratory; unadjusted bootstrap intervals are not confirmation or noninferiority evidence.

Operator replay evaluates all 16 settings and both held references on the same frozen Q/K/V, with imposed source V31 progress events. All 670 input files are validated before replay. An independent NumPy oracle checks every cheap-refresh map and score. All post-anchor calls share the same denominator; errors are also reported on common non-selection calls. Pre-anchor bootstrap/carry frames are excluded identically. This diagnostic cannot predict native candidate progress clocks, calls, output lengths, or accuracy.

Outputs: `reports/native_factor_contrasts.json`, `operator/attempt001/summary.json`, and a final human-readable report after completion. One H100 worker. Failed attempts remain immutable. Clean timing stays paused.
