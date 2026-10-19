# Bounded guarded native compatibility and repeatability probe

The frozen operator regression at
results/humaneval_index_put_fallback_20261019/attempt001 passed: the original
guarded compiled BF16-to-FP32 assignment raises, and the process-local compiler
adapter exactly matches explicit cast/copy without changing the Python callable.
Its reservation charged7.082s. This establishes compatibility only.

Freeze native mixed attempt007 before launch. Relative to failed guard
attempt006, install only that source-bound adapter before native compilation and
use fresh private compiler-cache namespaces. Both independent engines share the
new namespace; no unpatched old AOT artifact may be reused. Keep actual framework
guard=True, fill=False and cuBLAS32MiB, the explicit model/helper compiler policy,
native FULL/PIECEWISE settings and all original Python helper source identities.
The guard's fallback implementation is a disclosed numerical/runtime variant;
it is neither an accuracy improvement nor a proven cause of earlier variation.

Run the same152-request diagnostic schedule in each of two independent engines:
the original80-request qualification schedule with sampler/logit observation,
then72 unobserved measured repeats. Compare all methods, both seeds42/43 on the
six previously examined thinking-enabled questions. Preserve every difference,
including inactive initial-state fields, and distinguish those from changed
model logits or native trajectories. Require budget/layer/graph/finite receipts,
original sampler callable identity and effective adapter/guard/workspace receipts.
These observation-heavy passes are not timing evidence.

Allow at most1200s per engine,2400s total reservation ceiling, within unchanged
phasek256 and total43200s caps. The larger per-engine ceiling allows fresh-cache
compilation. Stop and preserve the attempt on unsupported native operations or
failed receipts. No full accuracy matrix starts from this diagnostic alone:
independent repeatability, qualified audit/clean behavior, both thinking modes
and matched controls must still pass under a named final protocol. K128 and
historical campaigns stay held. No installed environment files are modified.
