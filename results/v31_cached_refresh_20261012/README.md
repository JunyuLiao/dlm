# V31 cached-statistics partial refresh — 2026-10-12

The native accuracy panel is complete for 16 runnable arms under the frozen V31 contracts: 154 cells per arm (64 LongBench-v2 subset cells and 90 AIME26 cells). The primary bounded-swap `M=256` arm is in `attempt009`; the matched controls and the remaining valid mass, decay, phase, C_gate, and refresh-frequency arms are in `attempt010`. Every runnable arm has 154/154 records, complete receipts, official LongBench/AIME summaries, and matching frozen source hashes.

The headline control on this panel is `current_v31_control`: 53.1% LongBench-v2 subset overall and 54.44% AIME avg@k. The cached arms do not establish an accuracy improvement. Global top-K has near-control mean denoising counts but still has cells above V31's per-problem call count; bounded-swap variants have larger trajectory increases. Promotion is blocked by the fixed-trajectory gate and by the fact that this is a 154-cell panel rather than the 503-cell audit.

Operator checks remain clean: fixed `K=128`, bounded `B=8`, LOCAL dense accounting, zero order errors, zero non-finite attention calls, zero unexpected LOCAL skips, and zero timed CUDA graph captures. Selector audit time is diagnostic instrumentation, not clean timing. Clean timing remains paused.

Four requested variants remain diagnostic-only because native map wiring was not available: separate online-value-only tie-break, full-information/full-V tie-break, strict intersection, and filled intersection. They are covered by synthetic/reference replay and are not presented as matched native production arms.

See [`summary.md`](/home/exouser/ljy/dlm/results/v31_cached_refresh_20261012/summary.md) and [`final_report.json`](/home/exouser/ljy/dlm/results/v31_cached_refresh_20261012/final_report.json) for tables and machine-readable receipts. Failed setup/resource attempts remain preserved under `attempt001`–`attempt009`; no killed attempt was overwritten.
