# Guarded indexed-assignment compiler compatibility

Framework guard attempt006 stops during native warmup, before question records,
charging95.009s. The native sampler stores BF16 soft-embedding values in its
FP32 self_conditioning_embeds buffer. Under the pinned compiler's normal
index_put_impl_, values are converted to the destination dtype. The global
deterministic guard takes index_put_fallback before that conversion. Its
generated aten.index_put_ receives mixed dtypes and raises. The native sampler
Python function remains unchanged. This is a concrete guard-compatibility bug,
not proof of the original intermittent model-logit cause.

First freeze a bounded CUDA operator regression reproducing the same indexed
assignment with a FP32 cache, BF16 values, integer slot indices and guard=True.
Require the unpatched compiled operation to raise the dtype error. In the same
isolated process, install a narrowly scoped compiler adapter that inserts the
ordinary lowering's exact destination-dtype conversion before fallback only for
guarded CUDA, nonaccumulating, integer-index FP32<-BF16 assignments. Compare the
result to an explicit cast/copy reference; check untouched rows/source and
original Python callable identity. Disable FX graph-cache reuse for this toy
compile so an unpatched artifact cannot masquerade as patched code. No model
or sampler forward, weight change, proposal or stopping change is involved.

Do not edit installed PyTorch files or the native sampler function. The adapter
is process-local and fails if the pinned lowering structure changes. This
implements the same cast already present in the ordinary compiler path. Other
dtypes, boolean indices, accumulation and non-CUDA paths remain delegated.

Only after the regression passes, freeze a new native guarded attempt with the
adapter installed before compilation, effective-setting receipts and fresh
private compiler-cache namespaces (the runtime patch is not part of old AOT
cache keys). Preserve every failed run and its generated code. Require native
Python helper source identities, actual guard=True/fill=False, fixed BLAS
workspace and native FULL/PIECEWISE modes. No full matrix is released by this
diagnostic stage. Matching the repaired compatibility path does not by itself
repair or identify the earlier intermittent divergence.

Initial operator ceiling180s charges existing phasek256. A subsequent native
pair, if justified by the regression, gets a separately frozen bounded plan
within the unchanged phase/total caps. No silent guard whitelist or formula
change is permitted; a negative regression stops the native candidate.
