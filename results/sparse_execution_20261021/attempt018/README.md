# Best interior/tail kernel synthetic qualification, 2026-10-21

This is the current synthetic qualification attempt, not the native benchmark.
Best complete8K refresh so far429.32us uses interior/tail specialization of the
3D-TMA four-channel Q128 online kernel, library200354cb919fb6b8. It retains
ordered Gaussian32 threshold decisions and conditional full PV in one traversal,
with a generic irregular-query fallback and explicit irregular-key tail.
All16kernel tests pass, including Q256/1025keys with changing-input graph replay.
The fully-aligned predecessor passed40combined resident/V31/projection tests.

matrix001 freezes the full required Q256/H16/Hkv2/D512 BF16 grid:
K1024/4096/8192/16384 and40/50/60% actual physical GLOBAL tile sparsity. All
compilation/graph capture is outside timing. Actual capture count, fresh
projection/RMS, sketch packing, masks/lists, K/V movement, held consumer and
module/host times are included. Two-pass remains diagnostic only.

Native combined qualification, matched native controls, per-workload40-50%
held sparsity calibration and full benchmark relaunch remain gated. The old
optimized_refresh_benchmarks attempt004 stays paused and partial. User short
budgets are K128 for AIME26/MATH500/HumanEval, LongBench-v2 K8192. No speedup
is inferred from nominal sparsity or fewer forwards.
