# v20 Fan M1/M3 — research complete

Sole spec E:/Downloads/dllm_astra_sol_v20_fan_m1_m3_multidataset_same_day_20260927.md.
Continued e3f14520eea98e45998fb369f9027f3b2707ec4a; v18 closed and untouched.
Worktree E:/dlm/fan_m1_m3_20260927; branch research/fan-m1-m3-cost-refresh-20260927.
Preserve unrelated third_party/dinfer/assets/Wechat.JPG; never stage/reset it.
Astra owns methods/numerics/stopping/timing/interpretation; actual Sol implemented,
verified and assembled reports. Root integrated; no shared PPT/messages modified.

## Final checkpoint
700/700 main executions:50 valid blocks,350 accepted warm; separate G75 100/100:
50 successful first,50 accepted warm. No scored execution failure or missing cell.
Both H100 final stages ended by2026-09-27T16:56:14Z; read-only nvidia-smi around
17:10Z returned no compute processes on either host. No GPU/scoring work remains.
No rerun, retry, new grid, v18 relaunch or automatic budget extension.

Final Chinese report results/fan_m1_m3_multidataset_20260927/morning_or_final_brief.md.
fan_today.md and three local English slide drafts updated; first84 preserved.
adaptive_panel.json/csv/report.md retain complete main700; historical_panel.json/
csv/report.md separate G75 reference with work, quality, caps and same-GPU ratios.

M3 incremental gate NOT_PASSED for frozen GLOBAL5/P0/Triton/A8 point.
LongBench R2/R3 direct forward/native mpk.958/.950,dllm.918/.912; whole-step
also cheaper. B_A8 remains cheaper. AIME/RULER have no direct-forward headroom.
LB native/T/R2/R3 calls2360/2223/2842/2514; extra work erases savings.
LB R2/R3 E2E/native1.111/1.031; /freshT1.224/1.136; /B1.111/1.031.
LB strict correct native/T/R2/R3=6/6/6/5 of12; AIME=5/8/6/5 of12.
RULER all24/26 exact,official macro.9231. Earlyfirst84 R3gain did not persist.
No R picked from outcomes, no noninferiority or independent paper contribution.

G75L30_nativeQ128 new ALL-scope port, NOT old vLLM grouping equivalence:
RULER24/26,106calls,E2E/native1.093; AIME8/12,3367calls,.978;
LB3/12,2838calls,1.067. Same GPU question-seed pairing, later-stage drift caveat.
Old adaptive G75 evidence exists but small/different contract; held_bitmap_evidence.md.

## Immutable identities and semantics
Production CP3 00a2c4d3c93a7f685a4f75fba8ea47991fa5c383, deploy/cp3_00a2c4d.
Auxiliary CP5 05f99473406ef9b21d75b20d52d8e36b0124ddd1, deploy/cp5_05f9947.
Last runnable CPU collector CP10 33edb5b. Never overwrite deploys/first outputs.
Main GLOBAL_ONLY_NATIVE_LOCAL/P0/Triton; fiveGLOBAL and25nativeLOCAL layers.
D_native,D_matched,T_scope,M1_R1_A8_current_output,M3_R2_A8_current_output,
M3_R3_A8_current_output,B_A8_matched. G75 ALL/Triton separately named.
A8 numerical vs R1/2/3 decision clocks; R3 anchor8,nextD11. HistoricalQK/currentV
selection, currentQK/PV output; no QKV projection skipping. Common fast_t.
Native B256,max48,.8->.4temperature,.005confidence,stability1,.1entropy,
EOS/self-conditioning unchanged. RULERtaskcaps/OFF; AIME/LB8192/ON.
Binding SHA ee06c936e500fecfd219c6a9e277f99e77baed7b39d405ad96625f94e50734b5.
Protocol LF SHA99e1d287867b0906b8c5a2715d4cf962ec31e6d1f473c1fe0ce2df5891f1aa17.
Git/deploy LF; checkoutCRLF logicallyequal. Private frozen_protocol_deployed.json exact.
Main summarySHA25f5483e6650c118744fd642ecd3ec62d3178514eb8e8b2a0d730f1227ee9d5f.
HistoricalSHAae013630235ba0ba5212ea35ae627d5e4c7edf9dd1b56718b6bedb5514237107;
embedded first84/full exactly equal published main. No gold/answers committed.
Private summaries E:/dlm/v20_private/collect_remainder_002 and collect_historical_001.
Collectors/watcher terminal; all old dirs/locks preserved. No duplicates.

## Qualification, limits and honest missing evidence
GPU11/11 each, real QKV same-support envelope, physical counters, native brackets,
same-code eight-arm bridge passed. Newhost privateATen relink, kernelSHA same.
Direct full model-forward AND denoising-step separate from clean request timing.
Optional first-prefill-end CUDA span includes host gaps/latercommits; NOT puredecode
wall/TBT/GPUactive. RULER32 N16missing rows retained, naturalN4 supplement passed.
Physical skip denominators GLOBAL5 only. Detailed phase/age/state/memory tables
remain in selected001_forward/ruler_selected001_forward/selected_physical artifacts.
No full-trajectory per-state repricing or new GPUactive trace. M2/A4/earlyphase NOT_RUN.
Screen001 pre-load configfailure and two CPU missing-header buildfailures preserved.
No failed/uncertain scored request retried. NamedASRtranscript unavailable; speaker
notes read-only corroboration available. Development13/6/6questions×2seed, notheldout.

## Final resources and operations
Start2026-09-27T11:58Z; GPUstop19:28Z; final19:58Z unchanged.
844generations=44qualification+700main+100G75, cap1000.
21256.63526892662GPU-process seconds=5.904621h, cap28800seconds; max2workers.
resource_final.json outer receipts charged once, base1649.6709160804749 plus
STATE.closed_new_stage_charges; no inner-worker doublecount. All reservations zero.
Descriptors E:/dlm/v20_private/hosts.json retain pinned model/env/libs/UUID.
mpk exouser@149.165.151.254 UUID GPU-6139046a-b005-8fe5-a837-f8270472ab72,
ROOT /media/volume/dllm-1/dyh/fan_m1_m3_v20_20260927.
dllm exouser@149.165.159.64 UUID GPU-fc12ad5c-5334-5509-8fc6-465498fd3915,
ROOT /home/exouser/dyh/fan_m1_m3_v20_20260927.
Python D:/Users/30128/AppData/Local/Programs/Python/Python311/python.exe.

Existing heartbeat v18-research-stage-continuation PAUSED at completion; task not
archived. Deletion was rejected by auto-review; authorized safer pause succeeded.
User subsequently explicitly authorized code and all shareable data push to
coconight01/dlm_test branch research/fan-m1-m3-cost-refresh-20260927.
Export800 generated request records with provenance/manifest; omit source prompts,
gold and credentials. Per-canvas telemetry exists; full per-step tensors unrecorded.
Publication prepared; verify exact remote SHA after push. No GPU work implied.
Suggested next research step only: offline paired-trajectory divergence analysis
before proposing another frozen validation; do not automatically start a new round.
