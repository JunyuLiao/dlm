# Stable cheap-refresh implementation and execution plan

The user attachment is the governing task. Preserve Yuhan's inherited V31 dense/carry selection and native FA4 output; derive the new selector independently. K=8192 tokens/128 prefix KV64 tiles per query head and Q128 block, capped on short prefixes. All LOCAL calls, prefill and commits remain native dense.

1. Preserve predecessor sources byte-for-byte; append interpretation correction outside its frozen result root.
2. Independently qualify the explicit row/tile transpose, immutable last-dense cache, CUDA no-op budget handling, protected tiles, zero-change observations, native output and sampler RNG invariance.
3. Collect real frozen Q/K/V on two canvases of four requests spanning both workloads and all five GLOBAL layers. Prefix K/V are stored once per canvas; each call stores current Q and tail K/V. Raw tensors remain private.
4. Replay every mask sequentially and evaluate at the subsequent call. Use an operator-only predeclared hysteresis/removal-guard grid. Count duplicated selected QK and bounded outside probes separately. Keep full support dense work offline.
5. Repeat three native cells twice for progress control and observation-only control. Native panel waits on tensor, operator and reproducibility gates.
6. Freeze source, parameter, model, runtime and input identities. Launch 13 required arms in four resumable mixed-workload shards, with rotated arm order. Every failed leaf is retained and resumes use a new attempt.
7. Run official scorers after generation. Analyze N, N/C, cap hits and tails by workload; cluster AIME seeds by problem. Use the predeclared aggregate trajectory criteria. Do not reject on a single longer problem. The 503-cell expansion is a subsequent decision.

Clean timing remains paused. The current implementation deliberately uses a bounded duplicate QK pass rather than altering FA4's output arithmetic; it computes exactly the selected per-query-head rectangles plus mandatory tail, without a KV-head union. One host read per observed layer is disclosed to avoid rebuilding unchanged FA4 lists. No per-tile Python loops occur on the production path.
