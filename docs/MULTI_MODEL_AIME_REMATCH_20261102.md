# AIME physical-sparsity correction

User scope on November 2: stop UltraLLaDA RULER, preserve its 128-cap
observations, finish LLaDA2.2 and TraDo AIME V2 recalibration/reruns, then
report all completed results. Ultra AIME was already cancelled November 1.
These cancellations supersede the original 40-panel completion requirement;
the remaining original scope is 30 panels / 4010 requests. They do not imply
successful Ultra AIME or RULER evaluation.

Original V2 realized consumer sparsity was 75.207902% for LLaDA2.2 AIME and
60.450533% for TraDo AIME, outside the frozen 48–52% target. Other completed
V2 panels were within range: LLaDA HumanEval 49.216494%, LLaDA RULER
49.190728%, Ultra HumanEval 49.448969%, TraDo HumanEval 48.838929%.

Repeating the identical LLaDA development calibration in attempt002 selected
the same threshold 0.17782794100389226 and confirmed 51.015410% on those
development prompts. It does not repair benchmark drift. The exploratory
shell coordinator also skipped existing completed shards; it is superseded
and must never define the requested matched rerun.

New supplemental campaign:
`results/multi_model_aime_v2_matched_20261102/attempt001/`.
Its immutable `rematch_protocol.json` freezes count-only calibration on all
90 AIME question/seed cells, initial thresholds 0.03 and 0.025 respectively,
at most 15 bracketed candidates, then a distinct fresh 90-request generation
pass per model. Selection sees only pooled physical kept/eligible tiles;
no scorer or gold is accessed. This is explicitly transductive calibration,
not an independent held-out confirmation. Final fresh generation must also
achieve 48–52% sparsity. A failure preserves its attempt and blocks a final
matched-sparsity claim.

The numerical workers remain pinned to 224f8f11 for LLaDA2.2 and 8c0d14b0
for TraDo. All original native sampler settings, caps, questions, seeds,
Gaussian32 projection and reuse interval four remain fixed. Source/runtime
bindings and fresh physical-work receipts accompany each rerun. Original
native, all-kept, V31 and V3a controls remain their original complete panels.
Original V2 results remain available but must be labelled target misses.

Coordinator `scripts/multi_model_aime_rematch.py` is a new orchestration
variant. Five CPU checks pass for physical pooling, duplicate rejection,
nonmonotone bracketing and existing score extraction. The existing numerical
selector and native-runtime qualifications remain applicable because those
sources are imported without modification. Final verification must check
all 180 rerun identities, source pins, physical sparsity and official scores.

## User correction: small calibration, attempt003

The user explicitly rejected90-request calibration candidates as too slow. Stop and preserve attempt002, then use six fixed questions per model, seed42 only. Sample one representative from each sextile of native dense eligible work, selecting representatives solely to approximate historical pooled tile sparsity. No gold or score is used. Import exact historical measurements for those six cells when available. At most8small candidates; damped log-threshold interpolation on count-only measurements. Full90 generation is reserved for the final V2 benchmark, whose actual48–52% target is audited. All caps/samplers/kernels stay fixed. New frozen leaf:results/multi_model_aime_v2_matched_20261102/attempt003/.

## 2026-11-02 — user accepts final target drift without another rerun

After six-question calibration, run the final 90 AIME requests once per model. Report actual sparsity even if it misses 48-52%, then proceed; no post-benchmark recalibration or repeat. Preserve attempt003 and its original stopping-gate evidence. Supplemental completion policy lives in results/multi_model_aime_v2_completion_20261102/attempt001/. Numerical execution is unchanged; only completion interpretation and next-model scheduling change. Any missed cell cannot support an exactly matched-50% claim.
