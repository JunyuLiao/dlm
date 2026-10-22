# Optimized native comparison, attempt008

Source3db64cea retains the current selected attention kernels and all native
sampler invariants. All older attempts stay preserved as partial.

Native qualification already passed in attempt007; reuse only explicitly
fingerprinted source/runtime/input/threshold-identical pairs. AIME26's existing
full-request calibration achieved45.6943% held GLOBAL sparsity at threshold
0.059460355750136064. Reuse its computation-only evidence unchanged.
MATH500 extends thresholds to1.6/3.2/6.4/12.8; prior0.8 reached32.0452%.
HumanEval is fixed at0.5600716206066294 and LongBench-v2 at0.021 by user,
without new calibration. Report observed work/mismatch for all workloads.

Fresh full audit/clean generations begin after MATH calibration passes.
AIME26 30x42/43/44, MATH500500x42, HumanEval164x42 use V31K128;
LongBench503x42 uses V31K8192. Official scorers and clustered paired CIs follow.
No completed result before final_complete.json. Monitor every15minutes.
