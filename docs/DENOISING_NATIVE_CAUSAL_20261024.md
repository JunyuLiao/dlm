# Native causal denoising implementation and validation

Continue the October23 diagnostic campaign on denoising_trajectory_research.
Keep nine over-limit LongBench items blocked without truncation or replacement.
All original protocols, scores, negatives and private inputs remain unchanged.

First qualify in-place full-request native continuation in
results/denoising_native_causal_20261024/attempt001/. The source-hashed protocol
preserves FULL+fix51994, the LOCAL fix, supported V2 runner, native canvas256,
max48, schedule0.8->0.4, entropy acceptance0.1, confidence0.005 and stability1,
proposal/re-noise/self-conditioning/EOS and RNG order. Fourteen focused CPU
tests pass before GPU launch. No selector intervention is allowed in this gate.

After two denoising calls on the original HumanEval/53 seed42 prompt, snapshot
all native request/scheduler/sampler state, input/UVA staging buffers, graph-owned
metadata and complete KV backing storage in memory. Restore into original tensor
storage, preserving addresses, strides and aliases. Restore Python containers
and RNG. At this first-canvas anchor no output has been emitted; reconstruct
the opaque DecodeStream with its exact original prompt and flags. Require an
unrestored reference and two restored full native continuations to have identical
logits, active sampler states, output, N, C, engine-step/model counts and RNG.
All executed branch forwards are explicit and counted; no suffix forward, forced
continuation, cached-output substitution or alternate sampler is permitted.

One idle H100, one worker, timeout900s. No preemption, no paused campaign resume.
A failed gate remains preserved and cannot qualify selector methods. Numerical
parity is an engineering gate and not evidence of accuracy or acceleration.
Only after parity passes may a separately frozen selector-only diagnostic
protocol use native common-state forks. ResponseGuard precedes MarginGuard and
ValidityReuse; no proxy or full benchmark without held-out native predictive
evidence. Original mixed response and conservative validity results are retained.

Restoration qualification attempt001 completed its dense reference and then
failed on a frozen KV-spec dataclass attribute. Preserve it. Attempt002 contains
only a pre-source CPU invocation receipt. Attempt003 uses object.__setattr__
to retain frozen object references while restoring nested mutable state, and
bounds diagnostic path labels. Fifteen CPU tests pass including frozen-object
regression. The sampler, anchor, branch sequence and parity gates are unchanged.

Attempt003 preserved its dense reference then failed because msgspec
SamplingParams cannot use object.__setattr__. Attempt004 uses the normal
attribute setter with FrozenInstanceError-only fallback, and adds a real native
SamplingParams regression. Sixteen focused CPU tests pass. Native parity gates,
anchor and sampler remain unchanged; no sparse intervention has run.

Attempt004 passes both in-place restored full-request continuations exactly,
including logits, active sampler state, output, native counters and RNG. Each
restored branch executes ten remaining native engine/model calls, emits275
tokens and commits two canvases, matching the reference. All235 storage hashes
match immediately after each restoration. The final post-continuation storage
comparison is expected to differ from the anchor and is not a restore failure.

Attempt005 freezes a bounded selector-only native diagnostic. One-item smoke
precedes four short requests and the one context-eligible LongBench request.
PIECEWISE controls expose the native consumer boundary; all-kept paged FA4
compares against the same sparse consumer. Only layer29/head0/queryblock0
may prune wholly-prefix KV64 tiles; canvas/boundary and LOCAL remain dense.
Candidates have fixed physical budget fractions0.5/0.9: mass, deterministic
near/far swap, position sham; identical supports are deduplicated. Exact FP64
Q/K/V differential scores use the preceding native displacement; guards retain
the original0.2 deleted mass,0.1 relative bias and1.25 amplification limits.
Two native denoising calls per branch measure logit motion, margin, acceptance,
stability/confidence, self-conditioning and native stopping changes. All model
forwards are explicit and counted. No suffix forward, forced continuation, old
KV, cached-output substitution, proxy fitting or benchmark follows this smoke.
This limited first-canvas oracle is not a population accuracy or acceleration
result. Predictive failure remains a no-go; no full benchmark without held-out
native prediction beyond current-error and sham controls in two families.
