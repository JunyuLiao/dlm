# Other-model feasibility (CPU/source only, v12)

No downloads, runs or guessed semantics. This is a bounded inventory of what exists locally.

| model | local weights | adapter/runner in repo | generation semantics relevant to temporal reuse | status |
|---|---|---|---|---|
| LLaDA2.1-mini (inclusionAI) | **absent**: only config + remote code under `dyh/leaseqk-a100/models/llada21-mini` (1.5 MB) | `src/dllm/models/adapters/llada21.py` (`LLaDA2MoeAttention`; `model.generate(block_length, steps, threshold, editing_threshold, max_post_steps)`) | block diffusion with **token-to-token editing**: committed positions can be revised during post-steps, so a "history" score anchor must be invalidated on edits, not only on canvas commit | blocked by a weight download (not authorized this round) |
| LLaDA-8B-Instruct (GSAI-ML) | present (15 GB HF cache, snapshot 08b83a6f) | `fast_dllm_v1.py` (`LLaDAV1Adapter`) | fixed-step block diffusion, low-confidence remasking; no native adaptive stopping comparable to DiffusionGemma | runnable in principle; would need its own qualified adapter hooks and its own AIME protocol |
| I-DLM | absent | none found in the repository | unknown; no code to read, and stride/verifier semantics must not be guessed | not feasible without new code and weights |

Any follow-up must keep the DiffusionGemma AIME failure evidence and establish per-model commit/edit semantics
before adapting M1/M3. It must not swap in an easier benchmark.
