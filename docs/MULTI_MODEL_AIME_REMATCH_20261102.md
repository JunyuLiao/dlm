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
