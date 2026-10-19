# Framework deterministic guard, without memory poisoning

Fixed-BLAS attempt005 completed872.066/867.061s and receipts actual33554432-byte
cuBLAS,1048576-byte cuBLASLt and preferred cuBLAS. It does not fix the remaining
variation. V2 question145,seed42 has its first changed raw logits at call529;
one engine also fails unobserved repeats. Keep this negative; do not promote it.

A second raw state difference is control-after145/42's initial argmax only,
with identical logit hashes and subsequent trajectory. At its first denoise
call, native phase5 overwrites argmax before any emitted use; the prior value
is preserved only for commit requests. Preserve that raw difference and explain
its inactive status rather than treating it as another model divergence.
This distinction does not excuse the genuine V2 failure.

The explicit Inductor policy does not enable PyTorch's framework-wide
deterministic-algorithm guard. Freeze attempt006 adding only that guard relative
to attempt005: torch.use_deterministic_algorithms(True, warn_only=False). Keep
the supported fixed BLAS workspace. Explicitly disable
torch.utils.deterministic.fill_uninitialized_memory so the new guard does not
also change allocation initialization. Preserve all native Python model/helper
functions, sampler settings, methods, thresholds, budgets and graph modes.
Require actual guard=True and fill=False in exit receipts alongside effective
workspace bytes; fail closed on an unsupported operation or lost setting.

First rerun the four source-bound operator-definedness checks in a new immutable
preflight attempt, under these same flags. No native model or sampler forwards.
Run the native pair only if that preflight passes. Then use the same two
full-cohort clean diagnostic schedules:80 observed requests and72 unobserved
repeats, seed42/43, all methods. Compare every trace and repeat. The guard is a
diagnostic candidate, not proof of the original cause; do not silently whitelist
an operator or replace native stopping to make a gate pass.

One180-second operator reservation and at most two1000-second native reservations
charge existing phasek256. No phase or total43200s expansion. Stop before full
accuracy while genuine native reproducibility still fails. Any claimed causal
operator needs tensor evidence/replay; any empirical runtime qualification must
be named separately and cover both thinking settings and matched controls.
