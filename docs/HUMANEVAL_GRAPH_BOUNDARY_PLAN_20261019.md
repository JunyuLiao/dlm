# Locate the remaining change at native compiled graph boundaries

The guarded native attempt007 first engine completes997.189s but fails one
observed/unobserved V2 repeat (question129,seed43; N158/159, T3583/3574). Effective
guard=True, fill=False,32MiB cuBLAS and42/42 deterministic compiler kernels are
recorded. The second engine completes897.178s and fails3 native repeats. All72
fully observed independent trajectories match; native unobserved variation
remains. The compatibility adapter succeeds; guard is not a qualified repair.
Separate inspection of completed attempts002/003/005 finds identical37 unique
compiler source/name/configuration entries across each pair. The remaining
variation therefore cannot be attributed to a recorded selected-config change.

Next trace the inputs and outputs of existing CUDAGraphWrapper calls, twice
independently:80 requests with the existing full sampler observer,72 repetitions
with asynchronous graph/logit fingerprints only, and6 pure unobserved bookends.
The last6 cover the observed V2/V31/residual failures plus earlier V3a116/42 and
V2145/42. The lightweight72 retain request order, methods and seeds; no per-step
CPU tensor transfers, RNG-state reads or full sampler-state digests are added.
The compact before-sampler logit fingerprints copy to CPU only at request finish.
Compare both independent lightweight runs and within-engine full/light/pure
contrasts. Disclose which observer has passed each check. Retain the prior
explicit deterministic model/helper policy, native graph dispatch and native
sampler callables. Return to the baseline numerical policy without the rejected
global guard/adapter/workspace candidate, using its frozen existing cache. This
is a localization diagnostic, not a cross-policy accuracy comparison.

Wrap the original native graph call exactly once. Fingerprint eligible activation
arguments before it and its returned activation tensors after it, with the already
source-tested integer128-bit locator. Record nesting order, graph-wrapper ordinal,
runtime mode and sampler index. Do not bypass/replay graphs, insert native model
forwards, alter results/arguments or read output storage before the native call.
Exclude large parameter/persistent tensors by explicit rank/size filters; disclose
this partial argument scope and never infer equal complete inputs from these
fingerprints. Parameters and native model settings remain independently bound.
CPU-only binding tests (including independent graph-only mode and pre-request
native startup passthrough) must pass and be source-bound before launch.

The earliest changed boundary at the first changed raw-logit call will distinguish a graph segment from its outside
operators. Require all traces and exact observed/unobserved outcomes, preserving
failed observation checks. Matching fingerprints are inconclusive; a localized
fault still needs actual tensor capture and same-input operator replay. No full
matrix is released. No Torch/BLAS flag is promoted and no reproducibility gate
is relaxed.

At most two1400-second native reservations charge existing phasek256 after all
attempt007 reservations close, within unchanged phase16200s/total43200s. Keep a
30s closing margin; fail closed if insufficient capacity. Preserve attempts,
source/protocol hashes, actual kernel configurations and graph/capture receipts.
Keep K128 and all historical campaigns held until reproducibility qualifies.
