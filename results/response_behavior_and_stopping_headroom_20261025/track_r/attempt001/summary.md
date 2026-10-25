# Unchanged Response behavior replication

Complete exploratory short-task panel:30 AIME26 and40 HumanEval questions, paired seeds42/43, five arms. Original Response selector/runtime bytes are unchanged. This is not Stage B/C.

Legacy reproduction passed24 arm/cell comparisons, including HumanEval/71 seed43. All700 audit/clean cell pairs matched output,N,C,T and final CPU/CUDA RNG.

|Task / exposure|Arm|Correct / cells|N total|C total|T total|Mean N/C|Mean clean W (s)|
|---|---|---:|---:|---:|---:|---:|---:|
|aime26 / all|dense_full_fix51994|33/60|17232|1298|327823|12.142|5.282|
|aime26 / all|allkept_layer29|35/60|16899|1287|325095|12.064|5.902|
|aime26 / all|current_error_fixed4|31/60|17125|1303|328990|12.194|19.728|
|aime26 / all|sham_fixed4|31/60|17276|1303|328223|12.191|19.846|
|aime26 / all|response_guard_qdirection|36/60|17269|1310|330875|12.180|20.000|
|aime26 / new_to_response|dense_full_fix51994|21/40|11806|873|220214|12.248|5.427|
|aime26 / new_to_response|allkept_layer29|24/40|11518|873|220344|12.161|6.041|
|aime26 / new_to_response|current_error_fixed4|22/40|11195|851|214577|12.157|19.316|
|aime26 / new_to_response|sham_fixed4|21/40|12239|906|228393|12.483|21.113|
|aime26 / new_to_response|response_guard_qdirection|21/40|12491|925|234166|12.488|21.717|
|humaneval / all|dense_full_fix51994|78/80|2936|359|80386|8.100|0.732|
|humaneval / all|allkept_layer29|77/80|2922|352|78635|8.119|0.829|
|humaneval / all|current_error_fixed4|78/80|3143|371|85110|8.166|2.268|
|humaneval / all|sham_fixed4|77/80|2959|360|82734|8.009|2.109|
|humaneval / all|response_guard_qdirection|75/80|2902|357|80984|7.882|2.098|
|humaneval / new_to_response|dense_full_fix51994|58/60|2195|270|61051|8.104|0.730|
|humaneval / new_to_response|allkept_layer29|58/60|2200|261|58733|8.299|0.830|
|humaneval / new_to_response|current_error_fixed4|59/60|2424|280|64783|8.372|2.319|
|humaneval / new_to_response|sham_fixed4|58/60|2251|275|62854|7.987|2.139|
|humaneval / new_to_response|response_guard_qdirection|56/60|2206|270|61296|7.946|2.121|

Primary confirmation retains questions as clusters, with both seeds. Eight prespecified comparisons use simultaneous99.375% bootstrap intervals;95% intervals and legacy/full/strict-no-A1 sensitivities are descriptive. No noninferiority margin was declared.

|Task|Response versus|Quality difference (simultaneous interval when primary)|Paired N ratio (simultaneous interval when primary)|Paired N/C ratio|Paired T ratio|Paired clean W ratio|
|---|---|---|---|---:|---:|---:|
|aime26|dense_full_fix51994|0.0000 [-0.1500,0.1250]|1.1045 [0.9345,1.3534]|1.0305|1.0816|4.0587|
|aime26|allkept_layer29|-0.0750 [-0.2250,0.0500]|1.1016 [0.9690,1.2742]|1.0268|1.0820|3.5595|
|aime26|current_error_fixed4|-0.0250 [-0.1750,0.1250]|1.1150 [0.9884,1.2742]|1.0259|1.0951|1.1266|
|aime26|sham_fixed4|0.0000 [-0.2000,0.1750]|1.0272 [0.9446,1.1151]|1.0068|1.0299|1.0396|
|humaneval|dense_full_fix51994|-0.0333 [-0.1167,0.0667]|0.9766 [0.8900,1.0775]|0.9865|0.9998|2.5901|
|humaneval|allkept_layer29|-0.0333 [-0.1167,0.0667]|0.9712 [0.8765,1.0800]|0.9624|1.0232|2.2691|
|humaneval|current_error_fixed4|-0.0500 [-0.1333,0.0000]|0.9456 [0.8615,1.0250]|0.9548|0.9721|0.9604|
|humaneval|sham_fixed4|-0.0333 [-0.1333,0.0500]|0.9874 [0.9436,1.0308]|0.9946|0.9878|1.0022|

Decisions:
- aime26: earlier favorable direction not reproduced; clear quality lead=False, clear total-work lead=False. Observed quality losses versus: allkept_layer29, current_error_fixed4.
- humaneval: quality/work tradeoff; clear quality lead=False, clear total-work lead=False. Observed quality losses versus: dense_full_fix51994, allkept_layer29, current_error_fixed4, sham_fixed4.

Every correctness transition is preserved in analysis001/summary.json. New-to-Response is not universally unseen: all70 identities match some known historical prompt-tokenization hash. The AIME primary20-question stratum includes two A1-exposed questions; the strict18-question sensitivity is separate. HumanEval30 new questions exclude A1.

Explicit model forwards: 240741; complete native requests: 1464; allocated GPU-worker seconds: 13560.813. No failed GPU process is recorded. Allocation includes startup, qualification, warmup and cleanup.

Quality, total calls, calls/canvas, clean latency and causal mechanism are separate findings. Audit/operator timing is not clean policy latency. Physical rectangle accounting is not CUDA CTA timing. Prefix extraction and correct-in-both subsets are retrospective descriptions, not a native stopping policy or causal redundancy explanation.

Track H is separately frozen and gated. Previous unfavorable LongBench Response evidence remains; no new LongBench Response panel was launched. No production optimization or new selector was introduced.
