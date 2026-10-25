# Response behavior and native stopping headroom — completed gated studies

The original Response behavior reproduces exactly on legacy questions, but its quality/work advantage does not replicate on new-to-Response questions. The native single-pulse study finds local stopping headroom, without a useful directional-score or end-to-end gain. No accuracy-preservation, speedup or validated past-motion mechanism is claimed.

## Identity and execution

Branch `denoising_trajectory_research`; starting HEAD `3a1e6e9c903b3152012e84ac105854fbb09451ee`. R generation source `5ebd3e66b4b9ac54cbba24ded2cc4154ae5d84e3`, readout commit `e2f5d47d`; H generation source `2480aa68932a6ab35744b90e3dcdd15597af8c25`. Final publication commit is the commit containing this handoff; generation source checkpoints stay distinct.

R protocol SHA256 `e1e54486d484d8a83b2870208c438a32de8b982fdaf1af17b2dec349fd580c93`; H protocol SHA256 `849443b09b3e60934710c781804d4bb52f5211bab0b464c742e0848f84c1ede6`. Exact manifest/exposure identities, 16 model and 7 runtime file verification and scored-input hashes are public. No paused campaign resumed, GPU preempted, resource ceiling imposed or push performed. Native output caps 8192(short)/16384(long), 48-call canvas cap and sampler/RNG order remain fixed.

## CPU reporting correction

The new addendum preserves A1’s negative Q-only/QKV forecast result against persistence, both frozen aggregation definitions and five individual 10% crossings. Missing Q-only component-matched controls remain explicit. Selected error/mass-matched contrasts are 19/185 and 18/179. A1 deleted no native model support and its common first canvases had zero confidence-only delays. The 299 prior receipt corrections do not change aggregate conclusions. FULL completion is recorded; the original parent wait status remains unavailable.

## Track R: complete 70 questions, two seeds, five arms panel

All 700 audit/clean pairs match output, N, C, T and final CPU/CUDA RNG; all 700 clean timings have zero new CUDA captures/JIT events. The 24-cell source-matched legacy gate passes. A supplemental 200-cell original-stratum check matches every output/count/support/lifecycle record. Question clusters are 30 AIME/40 HumanEval, with two seeds each—not 140 independent questions. The primary confirmation strata have 20 AIME/30 HE questions; strict-no-A1 AIME 18 is separate. All 70 have some known historical exposure; new-to-Response is not universal benchmark non-exposure.

|Task / stratum|Arm|Official correct / seed-cells|Total N|Total C|Total T|Mean clean W(s)|
|---|---|---:|---:|---:|---:|---:|
|aime26 / all|dense_full_fix51994|33/60|17232|1298|327823|5.282|
|aime26 / all|allkept_layer29|35/60|16899|1287|325095|5.902|
|aime26 / all|current_error_fixed4|31/60|17125|1303|328990|19.728|
|aime26 / all|sham_fixed4|31/60|17276|1303|328223|19.846|
|aime26 / all|response_guard_qdirection|36/60|17269|1310|330875|20.000|
|aime26 / new_to_response|dense_full_fix51994|21/40|11806|873|220214|5.427|
|aime26 / new_to_response|allkept_layer29|24/40|11518|873|220344|6.041|
|aime26 / new_to_response|current_error_fixed4|22/40|11195|851|214577|19.316|
|aime26 / new_to_response|sham_fixed4|21/40|12239|906|228393|21.113|
|aime26 / new_to_response|response_guard_qdirection|21/40|12491|925|234166|21.717|
|humaneval / all|dense_full_fix51994|78/80|2936|359|80386|0.732|
|humaneval / all|allkept_layer29|77/80|2922|352|78635|0.829|
|humaneval / all|current_error_fixed4|78/80|3143|371|85110|2.268|
|humaneval / all|sham_fixed4|77/80|2959|360|82734|2.109|
|humaneval / all|response_guard_qdirection|75/80|2902|357|80984|2.098|
|humaneval / new_to_response|dense_full_fix51994|58/60|2195|270|61051|0.730|
|humaneval / new_to_response|allkept_layer29|58/60|2200|261|58733|0.830|
|humaneval / new_to_response|current_error_fixed4|59/60|2424|280|64783|2.319|
|humaneval / new_to_response|sham_fixed4|58/60|2251|275|62854|2.139|
|humaneval / new_to_response|response_guard_qdirection|56/60|2206|270|61296|2.121|

Primary comparisons use 50000 question-cluster bootstraps retaining both seeds; eight comparisons use simultaneous 99.375% intervals. No noninferiority margin was declared.

|New-to-Response task|Comparator|Paired N ratio|Primary simultaneous N interval when applicable|Paired N/C|Paired T|Paired clean W|Correct→wrong / wrong→correct|
|---|---|---:|---|---:|---:|---:|---|
|aime26|dense_full_fix51994|1.1045|[0.9344662058421134, 1.3534353488566258]|1.0305|1.0816|4.0587|3 / 3|
|aime26|allkept_layer29|1.1016|[0.9690353547914224, 1.2742106139929157]|1.0268|1.0820|3.5595|6 / 3|
|aime26|current_error_fixed4|1.1150|None|1.0259|1.0951|1.1266|3 / 2|
|aime26|sham_fixed4|1.0272|None|1.0068|1.0299|1.0396|4 / 4|
|humaneval|dense_full_fix51994|0.9766|[0.889958962990883, 1.0774912155422052]|0.9865|0.9998|2.5901|4 / 2|
|humaneval|allkept_layer29|0.9712|[0.8764900102590015, 1.0799699166557568]|0.9624|1.0232|2.2691|4 / 2|
|humaneval|current_error_fixed4|0.9456|None|0.9548|0.9721|0.9604|3 / 0|
|humaneval|sham_fixed4|0.9874|None|0.9946|0.9878|1.0022|3 / 1|

Geometric question-weighted ratios and arithmetic totals differ. New AIME has more total calls/tokens under Response; new HumanEval’s small paired call reduction is uncertain and accompanies quality regressions. The legacy AIME 15/20 versus FULL 12/20 remains, but confirmation is 21/40 versus 21/40. HumanEval confirmation is 56/60 versus FULL 58/60, current-error 59/60, sham 58/60. No primary promotion gate passes.

R physical deletion is 1.579%GLOBAL/0.619%overall on AIME and 1.074%GLOBAL/0.201%overall on HumanEval; LOCAL stays dense. Discovery QK/PV, counter refresh/hold work and nested timers are separately reported in track_r/attempt001/closeout002/. Native prefill/commit remain dense and outside the denoising rectangle denominator. No speed is inferred from these counts.

The frozen text rubric and sealed descriptive prefix addendum do not establish reduced distraction/redundancy. New AIME Response outputs are longer; repeated8-grams decrease slightly while duplicate lines and revision terms increase. Retrospective extractable candidates can be wrong and are not native stopping times. Every correctness transition is programmatically inspected and input-hash checked; HumanEval losses versus FULL include 94/42(entry point), 71/43(syntax), 99/43(syntax), 156/43(test), 133/43(syntax), with 69 improving on both seeds. Correct-in-both descriptions remain post-treatment subsets.

## Track H: fresh later-canvas single-pulse diagnostic

All 8 repeated-control cells qualify full output, state, RNG, counters and active operands. Five native anchors exist; only 3 development cells have qualifying directional banks. Confirmation has three missing anchors and one failed directional contrast, so 0/4 confirmatory banks. No identity replacement, truncation, later-canvas substitution or forced continuation. Each pulse deletes exactly 5%floor prefix blocks in four units at layer29(heads0/8, qblocks0/1, Q128×KV64); protected/current-canvas, LOCAL, prefill, commit stay dense.

Ten distinct deleting requests pass40 GQA-route tests; maximum relative error versus explicit current-QKV masked attention is 0.004264(bound 0.03). Three post-pulse all-kept controls exactly reproduce. No restoration is used; old restored output assembly remains unqualified. High/low pairs match each other in current error/mass; their current-error comparisons do not. Sham aliases and unit-versus-whole bias angles are explicit.

|Development question / seed|A remaining|Current-error|Sham|Direction-high|Direction-low|
|---|---:|---:|---:|---:|---:|
|longbench_v2/66ec2755821e116aacb1b6f2 / 42|15|12|12|13|13|
|longbench_v2/66ec2755821e116aacb1b6f2 / 43|44|44|44|44|44|
|longbench_v2/6724645dbb02136c067d73ad / 43|20|17|17|17|20|

The matched directional pair produces a3-call difference in one development state, but its better member ties current-error. Neither current-only rule beats both controls across both seeds of a development question. Development fails, so no confirmation pulse is launched. No AA/SA/AS/SS factorial, hold or production policy was run.

Local stopping gain does not ensure lower whole-request work: 66ec…seed42 moves from 15 remaining calls to 12(current-error) or 13(direction), but totalN rises 767→2963/2945 and T 4546→16384. On 672464…seed43, high reduces 20→17 remaining while N rises 2905→2993;low retains 20 and 2905. Every deleting request scores wrong and reaches the output cap. All deleting reference states were already wrong; therefore zero correct-reference deleting cases can support accuracy preservation. Native confidence/reference agreement is not correctness. Different stopping counts can legitimately change later native RNG offsets; no RNG compensation is introduced.

All H timing is diagnostic instrumentation and unqualified for clean policy latency. FULL and PIECEWISE references remain distinct from the common A trajectory; consumer substitutions are systematic treatments. See H summary for every reference/method N, C, T and quality, missing cells, physical work and signed native pulse traces.

## Resource use, limitations and decision

R: 1464 complete requests, 240741 model forwards, 13560.813 allocated GPU-worker seconds. H: 56 requests, 85312 forwards, 5460.291 seconds. Combined: 1520 requests, 326053 forwards, 5.2836 GPU-worker hours across 19 serial workers, including 19 warmups, startup, qualification and cleanup. There are zero GPU-process failures. One relay-log preflight error and one derived CPU-reporting error were preserved and corrected with no GPU rerun. Parent allocation has 15s polling resolution. S/N is amortized; nested timers are not added twice.

Decisions: no promotion of unchanged Response, Q-only/QKV past-motion forecasts, or either H direction rule. Exact MarginGuard and norm-only ValidityReuse remain parked. Local headroom and different future effects are demonstrated in the tested development states, but no useful score advantage, accuracy-preserving total-work reduction, confirmed held-out headroom or clean speedup is established. Results do not reject every layer, strength or sparse-attention mechanism.

Recommended next action: independent review of the saved native pulse/commit/RNG evidence explaining local acceleration versus longer complete requests, before designing another selector. No automatic Stage B/C, production selector, kernel optimization, or additional GPU experiment is authorized by this handoff.

## Inspect first

- cpu_reporting_addendum001/summary.md and behavior_analysis001/summary.md
- track_r/attempt001/summary.md, analysis001/summary.json, closeout002/resources.json
- track_h/attempt001/summary.md, analysis001/summary.json, closeout001/native_pulse_effects.json
- closeout001/verification.json, method_decisions.json, resources.json, publication_files.json
- figures001/local_vs_total_calls.png compares local headroom with whole-request work.
- COMMANDS.md and every launch/exit receipt for exact reproduction arguments.

All new study files are task-owned. Existing AGENTS edits and pre-existing HANDOFF/DECISIONS bodies are preserved. Prompts, gold, token IDs, raw text, tensors, private paths and logs remain ignored. No push.
