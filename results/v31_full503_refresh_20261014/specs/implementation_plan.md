# Execution plan

1. Freeze copied sources, all model hashes, official full503 input bindings, native sampler, budgets and arm definitions.
2. Run focused CPU/CUDA tests and current-Q/K/V full-information CPU-map and sparse-output checks for every-step baseline.
3. Repeat V31 control and compare updates-disabled observations; smoke every-step output receipts on the same two cells.
4. Run 13 arms in four rotated shards with one H100 worker, source checks and immutable attempts.
5. Officially score all 503 cells per arm, audit matching sampler/runtime/source, publish accuracy/trajectory/physical work/selector overhead separately.
6. Create final_complete only when coverage, receipts, official scorer identity and all hashes agree. Timing remains paused.
