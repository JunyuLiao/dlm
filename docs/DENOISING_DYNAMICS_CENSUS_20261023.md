# Native denoising census: production observer gate

Execution stage: `results/denoising_dynamics_census_20261022/attempt001/execution004/`.
Protocol SHA256: `b89d910d8d16f9fbdfe5a8217142e749fef1418c68f3e2844994a805b6b121ea`.
Original 30-item identity file SHA256:
`069a01ae660a79d87d6ba2306e52f79474522ca805d73ad21a991795f752af39`.
Original attempt protocol, manifest and receipt remain immutable.

The first arm is dense vLLM FULL with the exact #51994 mask fix and the
repository's FA4 LOCAL window fix. Pin the supported V2 model runner explicitly
(`VLLM_USE_V2_MODEL_RUNNER=1`), deterministic native helper compilation, BF16,
one H100, batch one, current model revision
`f7f5b7f5fa82ffc52addd066915886d497f5517b`. Preserve the native 256-token canvas,
48-call cap, 0.8 to 0.4 schedule, entropy-bound acceptance 0.1, joint stability
and confidence threshold 0.005, native proposal/RNG, self-conditioning,
re-noising, transitions and EOS. Do not add model forwards.

Freeze a one-item HumanEval/53 smoke before the eligible dense census.
Run warm-up, unobserved, minimal-state observer, rich observer, minimal-state
repeat and unobserved repeat at the original seed 42 and 8192 output budget.
Require identical output and prompt-token hashes, forward/call counts, final
CPU/CUDA RNG state, active per-call sampler-state hashes, zero observer RNG
changes and zero native convergence-predicate reconstruction mismatches.
Inactive initial argmax storage is excluded only before a canvas's first call.
All diagnostic timing is instrumented and cannot support speed claims.

The original frozen identity includes nine LongBench items beyond native
262144-token context. The user explicitly requires them to remain blocked:
no truncation, substitution or replacement. Preserve all 30 IDs and report
partial coverage. The remaining LongBench item has 185546 prompt tokens and
16384 output budget; its memory feasibility is a separate runtime gate.
Short requests use max model length 12288; the long item uses 202752.

Only a passing smoke permits the eligible dense census. Record per-call
entropy, margins, reconstructed acceptance, stable/confident fractions and
dwell, native stopping state, terminal attribution, N/C, output length and
actual prepare-attention physical geometry. The Q128 x KV64 rectangle
denominator is disclosed separately from actual CUDA CTA accounting.

The smoke establishes active sampler-state/repeat-output invariance, not a
full KV/layout fork-restore certificate. Phase 2 interventions remain gated
on full-state restoration. ResponseGuard, MarginGuard and ValidityReuse
remain unvalidated; no selector or benchmark expansion is authorized by a
failed gate. Preserve all negatives, original reuse provenance, and paused
campaigns. Each worker reservation is bounded at 1800 seconds.

Preparation execution001 was stopped before generation because its forward
counter covered a different runner. Preserve its source, startup log and
verified-worker SIGTERM receipt. Execution002 explicitly pinned the legacy
runner and failed before model loading: the installed vLLM rejects diffusion
models on that runner. Execution003 adapts the forward counter to the supported
V2 runner without changing the native sampler. No experimental conclusion
follows from the preceding startup attempts.

Execution003 stopped before generation because the prompt check compared a
BatchEncoding mapping with a token list. CPU inspection proves both tokenizer
and processor produce the exact frozen 91-token smoke input after extracting
input_ids. Execution004 normalizes that return format and preserves all
sampler and prompt settings. Its 15 focused CPU tests pass, including native
sampler state/RNG parity and unequal-dimension covariance finite differences.
