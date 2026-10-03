# V29 LongBench qualification evidence

This archive records generation/scorer pipeline qualification only. All 12/12 workers completed with exit code 0 and exactly one timed request: four arms (dense, native, allkept, method) at each of 32K, 64K and 96K. Each arm used the predeclared engine seed 29001, one request per length, and an 8192-token budget. The source deployment was `54163f4e7c8766aaa4187807420b3cf3167dba81`; the exact protocol ID is retained in `summary.json`.

This is not an accuracy or speed evaluation. There is only one question per length and one engine seed, and the arms can have different decoding trajectories. Completion of infrastructure qualification does not mean every answer was correct. Wrong, capped, unparsed and missing-final outputs remain scored failures without replacement or filtering. The 64K dense capped/unparsed output and all four wrong 96K outputs remain in the archive. Timed CUDA graph captures were zero for all 12 requests. Any scorer-generated ratios or statistics are descriptive qualification details, not scientific evidence or independent-request confidence intervals.

## Files

- `requests.csv`: all 12 per-request settings, timing, forward counts, stopping and scoring flags; token counts only.
- `request_receipts.jsonl`: all 12 path/counter receipts and compilation deltas.
- `workers.json`: all worker completion records and both timing boundaries, including the prior unsuccessful cleanup attempt.
- `arms.csv`: absolute per-arm scoring and request measurements.
- `summary.csv`, `summary.json`, `summary.md`: scorer-produced aggregates and comparisons. Only the Markdown heading and introductory qualification warning were changed for publication; measured values were preserved.

The current completed attempt reserved 2169.317956341 outer-supervisor GPU seconds. Prior attempt 001 reserved 244.358443453 seconds and is retained despite a cleanup-residual failure after its completed worker. Campaign resource accounting is therefore **2413.676399794 GPU seconds**. Inner terminal timings cover a narrower boundary; they are not added to the outer durations and do not replace them. Resource reservation includes setup, model loading, compilation/capture, execution and cleanup; it is not request speed.

## Privacy audit

The seven published detail files were inspected for private absolute paths, private hashes/fingerprints, credentials, prompt content, input/output token IDs, gold/parsed answer literals and generated text. None were found. Fields containing `token` are scalar counts, not token arrays. Hex digest-shaped values are public source/deployment Git commits. Public dataset labels, indices, the host alias and software/configuration fields are retained for reproducibility. No private qualification-proof file, raw completion or private log is included. Private originals were left unchanged; the files above were copied into this new directory.
