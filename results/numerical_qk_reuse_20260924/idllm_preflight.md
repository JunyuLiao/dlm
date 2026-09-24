# I-DLM preflight — NOT_RUN

Scope: read-only inspection of the pinned Junyu branch for a native I-DLM
baseline and a valid numerical-score/current-V reuse boundary. No model was
downloaded or loaded, and no generation or attention experiment ran.

## Repository finding

The branch has no I-DLM adapter, experiment, checkpoint reference, or runnable
configuration. `src/dllm/models/registry.py` registers fast-dLLM v1/v2,
LLaDA2.1 Mini, and DiffusionGemma only. Filename search found no I-DLM or
interleaved-diffusion model files under the repository. Text search for
`I-DLM`, `IDLM`, `i_dlm`, and “interleaved diffusion language model” found no
matches in `src`, `experiments`, `benchmarks`, `scripts`, the main README, or
`pyproject.toml`. The tracked `results` tree has no I-DLM result/config path.

| Required native fact | Preflight result |
|---|---|
| Checkpoint and tokenizer/config revision | Unknown; no reference in this branch |
| Actual block/stride schedule | Unknown; do not infer 8 or 16 |
| Proposal versus verification calls | Unknown; no native loop to inspect |
| Attention masks and legal support | Unknown |
| Cache lifetime and absolute query/KV positions | Unknown |
| Existing adapter or baseline invocation | None |

## Applicability boundary

The DiffusionGemma numerical-score/current-V implementation cannot be called
an I-DLM integration here. Reuse requires genuinely observed historical QK
scores for the **same absolute query and legal KV positions**, with compatible
layer, head, scale, mask, and cache epoch. An interleaved process may reuse a
tensor slot for a different absolute position; tensor index equality alone is
insufficient. Current V would still be projected and used for routing/output,
while historical scores would make the attention output approximate.

Proposal and verification roles must be mapped before choosing any insertion
point. If the native verifier computes probabilities or corrections from its
current attention output, stale scores can change those probabilities and the
acceptance result. No lossless-verification claim follows from the
DiffusionGemma path. A valid pilot first needs the actual local I-DLM
checkpoint/config and native source, then a native baseline, an absolute
position/cache audit, and a small same-state numerical comparison. The
DiffusionGemma canvas/stopping loop must not be copied into that model.
