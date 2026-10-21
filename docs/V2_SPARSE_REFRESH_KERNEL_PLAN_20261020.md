# V2 sparse refresh kernel study

This user-authorized study targets HumanEval164, K128, thinking on, seed42 on
one H100. The inherited Yuhan V31 reuse lifecycle and Junyu Gaussian32 family
retain their attribution. Existing published V2 returns dense observing FA4 on
refreshes; `v2_sparse_refresh` is a distinct variant that returns current masked
attention on refreshes as requested. Its ordinary held/carry consumer, progress
clock, sampler and LOCAL path stay inherited. No earlier paused queue resumes.

The first implementation candidates remove observing dense FA4 from discovery:
(1) parallel QK/PZ summaries, ordered V2 scan, current sparse FA4;
(2) fuse QK/PZ and the ordered scan in a Q128 program, then current sparse FA4.
The second removes summary traffic and repeated launches, at the cost of fewer
concurrent CTAs. The same Q128/KV64 decision, current FP32 Gaussian32, full-V RMS,
ascending physical key order, mass-inclusive skipped state, sticky bonus1.386,
first support and mandatory canvas/boundary tiles apply. Projection is charged.
Current Triton QK supplies prefix masses in these named candidates, whereas
published V2 overwrites them with observing FA4 masses. Audit mask differences;
do not assume bitwise equivalence. Full-dimensional model output is never PZ.

BLASST reference: official NVIDIA TensorRT-LLM
`cpp/kernels/fmha_v2/src/fmha/warpspec/{compute,epilogue}.h`, read-only local
reference and upstream https://github.com/NVIDIA/TensorRT-LLM. Its producer/
consumer vote and skipped-PV synchronization are relevant; its pre-softmax
mass test cannot replace the V2 post-PZ decision. A Q128 x D512 FP32 output
alone occupies65536 registers, so a monolithic CTA has no room for QK/PZ state;
split roles or staging must be measured, not assumed to win.

Freeze source/spec before generation; store private native tensors, manifests,
outputs and caches under the new results campaign. Start with relevant oracle
tests and controlled H100 kernel timing, then capture native HumanEval K128
thinking-on refresh inputs from already examined prompts. Compare complete
projection/discovery/output/list-construction cost against matched V31 discovery
and separately against native dense FA4. Zero timed graph captures, interleaved
warm timing, exact geometry, source/runtime receipts and observed sparsity are
required. A synthetic or matched-state replay is a kernel diagnostic and does
not establish native task accuracy or request speedup.

If an optimized refresh beats V31's complete dense refresh on the native states,
qualify all-kept/masked operator correctness and reproducible native audit/clean
execution, then run HumanEval164 official scoring with source-matched V31,
dense FULL fix51994, PIECEWISE and all-kept controls. Expand only after that gate
to AIME26 seeds42/43/44 (K1728) and LongBench-v2 full503 seed42 (K8192), using
new resumable attempts and workload-specific uniform thresholds calibrated to
absolute first-canvas GLOBAL kept tiles. First-canvas calibration does not imply
whole-trajectory equal work. Report N,C,T,S/N,S,W and eligible/kept GLOBAL and
LOCAL denominators. Negative kernel evidence blocks expensive expansion.

Every GPU attempt checks the live idle device and holds an exclusive project
lock. Retain failed attempts and measured reservation seconds. Any initial time
estimate is an engineering estimate, not a new user authorization limit.

## Implementation and execution addendum

The first generic Triton pipeline exceeded H100 shared memory; `num_stages=1`
fits. Further named candidates include max/scaled-sum routing normalization and
squared risk comparison, a normalized parallel-summary scan, D64 streaming QK,
and four/eight/sixteen-warp layouts. Skipped routing mass always advances and
projected routing state advances only for retained blocks. Arithmetic changes
are source-identified candidates, not bitwise substitutions by assertion.

A fully fused conditional-PV prototype uses duplicate Q128 discovery in two
Q64 output CTAs. Both make the same Q128 vote, avoiding inter-CTA waiting, while
each owns half the D512 output rows. It retains an independent retained-only
softmax normalizer for the full-dimensional output. This trades extra QK/PZ for
lower accumulator pressure and fewer kernel launches; it uses native BF16 P
rounding with different accumulation order from FA4. The attempted multi-CTA
cluster lowering fails Triton's CTA planner and remains a recorded negative.

Automatic approval review rejected a repository commit and both persisted-QKV
capture and nonpersistent live HumanEval attention replay. No native worker was
launched. Use separately hashed execution snapshots for source freezing; the
native replay action remains pending the explicit user approval requested in
the conversation. Synthetic diagnostics may continue. Report them as synthetic
and do not use them to promote AIME/LongBench generation. Private-input capture
support was removed from the active benchmark source. An unlaunched prepared
capture attempt is preserved as evidence of the rejected action.
