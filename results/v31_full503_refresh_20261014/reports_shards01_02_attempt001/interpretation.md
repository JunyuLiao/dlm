# Interpretation of the completed two-shard interim panel

The user paused generation after shard02 at 2026-10-15 03:54:07 UTC. All 13 arms have the same 252 seed-1 LongBench-v2 problems, 3,276 native cells total. Each shard contains 126 problems. Another 251 problems per arm remain; shard03 never launched. See [all tables and paired intervals](summary.md). This is a partial, previously examined benchmark audit.

## 1. Operator diagnostics

The frozen gates passed 74 CPU/CUDA/FA4 tests. Independently validated factorial replay matched the CPU oracle on 2,550 refresh maps, with maximum fresh-statistic / CPU-score differences 1.907e-6 / 1.639e-6. The added fresh-every-step diagnostic covered 20 identical-Q/K/V snapshots across both workloads and all five GLOBAL layers: every CPU map and direct native FA4 output matched, with zero captures. Its mean of snapshot mean row-output L2 errors versus dense was 1.7787; maximum row L2 error was 12.3816. These are absolute per-head output units on frozen snapshots, not native accuracy predictions. Qualification receipts and source audit remain in the parent study.

All 3,276 evaluated records pass source/input/runtime/sampler, completed-artifact, native routing and budget checks. LOCAL remained dense; request-time graph captures, budget errors and order errors are zero. Warmup captures are outside evaluated requests. K=128 applies to eligible prefix tiles, with short-prefix caps and inherited protected/structural tiles preserved.

## 2. Native accuracy

Fresh every step has the highest observed score, 134/252 (53.17%), versus V31 130/252 (51.59%). The gain is four answers: 22 paired wins and 18 losses. Its paired 95% bootstrap accuracy interval is -3.57 to +6.75 percentage points, so this does not establish superiority. All nine reported accuracy contrasts have intervals that include zero; intervals are descriptive and unadjusted.

No cheap-refresh arm exceeds V31's observed accuracy. Stable no-progress interval 2 is the highest-scoring cheap arm at 127/252, three below V31. Raw no-progress interval 2 scores 125/252 with the lowest observed cheap-arm mean N. Among stable no-progress intervals, 2 has the best observed accuracy and mean N; there is no monotonic interval trend.

At interval 2, stability changes correct answers by -6 with progress enabled and +2 without progress. Enabling progress changes raw by +1 answer with N +28.28%, and stable by -7 answers with N +3.43%. Stability effects also reverse across individual shards: progress-on +3/-9 answers, progress-off +6/-4. The data do not establish a general stability or progress accuracy benefit.

## 3. Denoising trajectory

Fresh every step has mean N 269.02 versus V31 291.66, an observed 7.76% reduction. Mean N/C is 18.02 versus 18.60; output length is 4.31% shorter. However, N increases on 129/252 problems and the median paired N change is +3. Its N-ratio interval is 0.749 to 1.146. Lower aggregate N is neither a uniform per-problem improvement nor an established reduction.

Stable no-progress interval 2 has mean N 4.15% below V31, but outputs are 8.16% shorter and mean N/C is 1.10% higher. Its p95 N is 811.65, versus 727.15 for V31. Its apparent mean-N saving therefore does not establish improved per-canvas convergence or tail behavior. The native sampler and output limits were unchanged throughout.

## 4. Selector overhead

| Arm | Selector event audit seconds / problem | Reported cached state MiB (max) |
|---|---:|---:|
| Raw / no progress / 2 | 3.723 | 348.25 |
| Stable / no progress / 2 | 4.119 | 348.25 |
| Fresh every step | 3.717 | 0.00 |

These are CUDA-event elapsed audit measurements, including instrumented dispatch; they are not kernel-only time or clean end-to-end latency. Reported state excludes transient workspaces and total model memory. The V31 selection path is not measured by these new selector fields, so zero there is not a zero-cost control.

Fresh every step additionally executes 16,327,904,160 full-support QK rectangles, the same number of PV rectangles, and 53,781,280 duplicate tail-QK rectangles. Its output-consumer sparsity must not conceal this observation work. It inserts no extra transformer/denoising forward. All per-arm counts, cache sizes, physical GLOBAL/LOCAL work and instrumented GPU reservation seconds are in native_results.json.

## 5. Clean timing

Intentionally absent and paused. No speedup is inferred from denoising counts, selector audit times or physical sparsity. Generation is paused with no study GPU worker remaining; resume requires explicit user instruction.
