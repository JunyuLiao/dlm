# Later-canvas signed-bias stopping-headroom diagnostic

Separate from unchanged Response replication and the failed A1 attention-motion forecasts. Four LongBench question clusters, two paired seeds. This is a targeted diagnostic cohort, not representative LongBench accuracy. Source/protocol and scientific source-preparation seal are public. No prompt truncation, native sampler change, restoration or resource ceiling.

|Question|Seed|FULL correct; N/C/T|PIECEWISE correct; N/C/T|Common A correct; N/C/T|FULL/PW output equal|
|---|---:|---|---|---|---|
|longbench_v2/66ec2755821e116aacb1b6f2|42|False; 2780/64/16384|False; 2780/64/16384|False; 767/18/4546|True|
|longbench_v2/66ec2755821e116aacb1b6f2|43|False; 3013/64/16384|False; 3013/64/16384|False; 2811/64/16384|True|
|longbench_v2/6724645dbb02136c067d73ad|42|False; 91/5/1186|False; 91/5/1186|False; 869/35/8909|True|
|longbench_v2/6724645dbb02136c067d73ad|43|False; 2858/64/16384|False; 2858/64/16384|False; 2905/64/16384|True|
|longbench_v2/6719dc46bb02136c067d470b|42|True; 119/7/1598|True; 119/7/1598|True; 152/8/1998|True|
|longbench_v2/6719dc46bb02136c067d470b|43|True; 134/8/1882|True; 134/8/1882|True; 266/11/2666|True|
|longbench_v2/67286036bb02136c067d9092|42|True; 114/9/2090|True; 114/9/2090|True; 87/6/1332|True|
|longbench_v2/67286036bb02136c067d9092|43|True; 111/7/1709|True; 111/7/1709|True; 84/6/1485|True|

Consumer substitutions are systematic treatments, not noise to subtract. Native remaining-call effects use the qualified common A state; FULL/PW anchors are separate trajectories.

|Question|Seed|Split|Common anchor (zero-based canvas / completed calls)|A remaining|Directional bank|Distinct selected supports|
|---|---:|---|---|---:|---|---:|
|longbench_v2/66ec2755821e116aacb1b6f2|42|development|{'after_completed_calls': 3, 'canvas': 3}|15|True|3|
|longbench_v2/66ec2755821e116aacb1b6f2|43|development|{'after_completed_calls': 4, 'canvas': 1}|44|True|4|
|longbench_v2/6724645dbb02136c067d73ad|42|development|{'after_completed_calls': 19, 'canvas': 20}|1|False|0|
|longbench_v2/6724645dbb02136c067d73ad|43|development|{'after_completed_calls': 18, 'canvas': 3}|20|True|3|
|longbench_v2/6719dc46bb02136c067d470b|42|confirmation|None|None|False|0|
|longbench_v2/6719dc46bb02136c067d470b|43|confirmation|{'after_completed_calls': 20, 'canvas': 5}|2|False|0|
|longbench_v2/67286036bb02136c067d9092|42|confirmation|None|None|False|0|
|longbench_v2/67286036bb02136c067d9092|43|confirmation|None|None|False|0|

Matched directional banks: 3/8cells; native anchors: 5/8. Missing anchors and contrasts are not replaced. Routing units and masks are not independent task samples.

|Question|Seed|Arm|Remaining calls|Gain versus A|Complete N/C/T|Correct / A correct|Alias|
|---|---:|---|---:|---:|---|---|---|
|longbench_v2/66ec2755821e116aacb1b6f2|42|current_error|12|3|2963/64/16384|False / False|None|
|longbench_v2/66ec2755821e116aacb1b6f2|42|support_sham|12|3|2963/64/16384|False / False|cell000_current_error|
|longbench_v2/66ec2755821e116aacb1b6f2|42|direction_high|13|2|2945/64/16384|False / False|None|
|longbench_v2/66ec2755821e116aacb1b6f2|42|direction_low|13|2|2945/64/16384|False / False|None|
|longbench_v2/66ec2755821e116aacb1b6f2|43|current_error|44|0|2811/64/16384|False / False|None|
|longbench_v2/66ec2755821e116aacb1b6f2|43|support_sham|44|0|2811/64/16384|False / False|None|
|longbench_v2/66ec2755821e116aacb1b6f2|43|direction_high|44|0|2811/64/16384|False / False|None|
|longbench_v2/66ec2755821e116aacb1b6f2|43|direction_low|44|0|2811/64/16384|False / False|None|
|longbench_v2/6724645dbb02136c067d73ad|43|current_error|17|3|2993/64/16384|False / False|None|
|longbench_v2/6724645dbb02136c067d73ad|43|support_sham|17|3|2993/64/16384|False / False|cell003_current_error|
|longbench_v2/6724645dbb02136c067d73ad|43|direction_high|17|3|2993/64/16384|False / False|None|
|longbench_v2/6724645dbb02136c067d73ad|43|direction_low|20|0|2905/64/16384|False / False|None|

Development decision: {'confirmation_authorized': False, 'interpretation': 'Development may reveal hindsight headroom. Only a replicated current-information rule with no observed dense-correct regression advances; no confirmation data are read here.', 'options': [{'available_cells': 3, 'eligible': False, 'mean_remaining_calls': 24.666666666666668, 'quality_regressions': [], 'replicated_development_questions': [], 'rule': 'direction_high'}, {'available_cells': 3, 'eligible': False, 'mean_remaining_calls': 25.666666666666668, 'quality_regressions': [], 'replicated_development_questions': [], 'rule': 'direction_low'}], 'selected_current_only_rule': None, 'source_sha256': 'fd2b35f1da339e93b284482efc044c2dbe1df6740a05084443633f3599f9c4ab'}

Confirmation gate passed: False. Every executed complete request is officially scored; all quality transitions versus A,FULL,PIECEWISE,current-error and sham are in analysis001/summary.json.

A wrong/capped reference cannot certify preserving a correct answer. A hindsight-best direction is oracle headroom only. Similar maximum current error/mass does not equate every position-wise perturbation. The primary direction pair varies one routing unit; the whole-composite angle and current-error matching status are recorded separately.

Repeated fresh common controls qualify full native output,N,C,T,RNG,counters and target-window fingerprints. Each executed deleting path additionally requires actual current K/V-tail equality, four GQA-route comparisons against explicit current masked attention, correct physical mask and zero protected/LOCAL deletion. Restored output assembly remains unqualified and unused.

Actual model forwards: 85312; complete native requests including warmups: 56; allocated GPU-worker seconds: 5460.291. No failed H GPU process recorded.

All H timing is audit instrumentation and is not clean policy latency. No quality-noninferiority, speedup, full-trajectory acceleration or optimal-layer claim follows. This is layer29,four routing units,one5%prefix pulse at a qualified later confidence stall; other locations/strengths and repeated policies were not tested. No Stage B/C or kernel campaign launched.
