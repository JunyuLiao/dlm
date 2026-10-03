# New-model adaptation audit (V28, 2026-10-02)

The two new models have official-stack execution smokes, not ports of the Gemma
sparse algorithms. LLaDA's toy checks passed; I-DLM's runs reached the length cap,
so its natural stopping and task quality are not qualified. Do not treat a
substring answer-presence check as benchmark accuracy. See the per-model smoke
reports under `results/v28_20261002/`.

| Model | Actual clock/identity | Current Gemma assumptions that fail |
|---|---|---|
| LLaDA2.1-mini, upstream JointThreshold | Fixed 32-row block; M2T and T2T can both update it; termination publishes final KV | Q128 DP refinement explicitly requires a complete 128-row tile; there are not two Q64 groups in a 32-row block. Removing all masks alone is not the final-KV event because T2T may continue. |
| I-DLM-8B, bundled IDLMBlockN | One causal forward serves clean/speculative verification; acceptance/rollback changes committed positions | gen_block_size=4 is not a fixed Gemma canvas. Advance can be 1, 1+accepted draft count, or a rejection-dependent amount; KV is trimmed. Row/forward indices are not persistent token identity. |

Verified code boundaries in this repository include `v27_dense_prefix.py`'s
Q128 guard, `integration.py`'s prefix+nq layout and fixed-increment carry checks,
`value_direction_hopper/query_adaptive.py`'s 48-step reset/temperature schedule,
and the DiffusionGemma-only vLLM class patch. The current official new-model stacks
use FlashInfer/128-d head attention, not the Gemma GLOBAL512 paged consumer.
The vendored `third_party/sglang` snapshot predates the installed JointThreshold;
it is not evidence for the current official loop.

Official-loop inspection: JointThreshold can force an unmask and can rewrite
already non-mask tokens; a no-change exit already has final KV and need not add a
separate commit forward. Therefore a transferred-token flag is not the complement
of a renoised flag. IDLMBlockN executes a causal forward, verifies/samples, then
trims/advances; clean sampling uses configured temperature/top-k/top-p whereas
speculative drafting uses argmax. Its existing total_forwards statistic is not
verified per request or against capture/general model calls; pure prefill is counted. Neither smoke publishes
an unverified per-request forward count.

A defensible port starts with a model-specific event clock, absolute token/KV
epoch identity, legal causal masks and an all-kept equivalence control. Temporal
sensitivity can compare only the same surviving target position. Initial I-DLM
verification/correction stays dense; carry requires a proven unchanged accepted
prefix, with changed/new/boundary tiles kept. Risk summaries, support selection,
and V-ranking ablations are reusable ideas; R6/A64/carry/q64/regroup/C-gate presets
are not already implemented for either model. Tune any new thresholds on a
separately frozen development set before evaluating benchmark quality.
