# Corrected HumanEval result interpretation, 2026-10-20

Both frozen budgets are complete and exactly reproducible across audit/clean
on all164 tasks, seed42, both thinking settings. See the combined tables,
per-item records, paired intervals and source certificates in
`results/humaneval_corrected_k256_k128_20261020/attempt001/`.

V2 atK128 is the strongest mean result among the value methods: accuracy improves
by1/4 tasks on/off, forwards decline7.13%/9.13%, and mean request latency declines
5.69%/7.84% versus V31. Speedup95% intervals are[0.897,1.239] and[0.905,1.321],
so speedup is not established beyond the observed engines/questions. AtK256,
V2 gains3/1 correct but raises mean calls and latency. K128 thinking-on V2 reduces
N/C10.712→9.738 while output length grows1243→1277 tokens; its lower mean calls
therefore accompanies fewer steps per canvas. Its output caps rise1→3 and its
N_p95 rises126.7→133.85. Mean trajectory improvement is not uniform tail improvement.

First-canvas matching does not match whole work. V2/V31 GLOBAL kept-tile ratios
are1.478/1.143 for K256 on/off and1.075/0.890 for K128. K128 non-thinking V2 misses
the initial sparsity target by+0.53116pp and uses6.32% fewer first-canvas kept
tiles under the predeclared closest-threshold rule. Keep that limitation visible.

Full-value swaps and guarded residual are slower than V31 in every budget/mode
cell. Thinking-on swaps add62.4%/55.6% request time atK256/K128 despite only0.35%/
2.50% fewer forwards. Their extra anchor-module audit elapsed averages0.707/
0.783 seconds per problem. The device_exact_v1 initializer still constructs
FP32 full QK/PV moments in per-head/per-query-tile PyTorch loops and materializes
block numerators (experiments/numerical_qk_reuse/v31_value_fast_runtime.py).
This observed implementation overhead is substantial; the module audit timings
are not interchangeable with isolated clean kernel timing. Guarded residual
also lacks a consistent accuracy or trajectory gain in these cells.

Native FULL is faster than every tested sparse/value method. Dense implementations
have their own stable outputs/trajectories; all-kept reaches164/164 thinking-on,
where native FULL reaches161/164. Equality is required between repeated instances
of the same method, not between distinct attention consumers. Shared references
are not extra independent trials. Main seed42 and six examined seed43 tasks do
not support seed-robust superiority or a broad novelty claim.
