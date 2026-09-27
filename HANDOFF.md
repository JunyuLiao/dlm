# v18 continuation — bounded user resource extension active

Authority: v18 plus later user “不必在乎预算 没事 gpu是自己的”, documented in
RESOURCE_AUTHORIZATION.md. Scientific scope remains frozen. Branch
research/astra-junyu-frontier-20260926; parent d2bef2189c046e9a4277ab16d751ebeab11c15d8.
Read DESIGN_DECISION.md; ignore unrelated third_party/dinfer/assets/Wechat.JPG.
Prior WSL v8, remote v15, peer branches and original immutable deployments untouched.

Completed evidence:
- Historical identity checks passed;120 old warm pairs lack strict phase evidence.
- Native stop is stable deterministic argmax AND mean entropy<.005; acceptance.1,
  B256/cap48, native temperature.8→.4 and normal EOS unchanged.
- RULER thinkingFalse/exact156 token lists; AIME thinkingTrue/exact30/8192 budget.
- Native legal all-layer scope qualified13 states; legacyLOCAL geometry differed.
- Density-only442 calibration runs: U50/T50/U60/T60 all within2pp whole/G/L.
  This is not T60 quality/work/speed success.
- Strong B8 full16 equivalent;8 strictwarm accepted; wallratio1.000187, no speedup.
- Bridge6 requests/host passes tokens/calls/stop/termination;1047 tensors identical.
- Only post-QK ball candidate stopped beforeCUDA implementation: no finite
  numerical certificate or demonstrated net execution headroom. No replacement/CVM.
- Initial RULER936 executions complete,468/host;468 strictwarm pairs, zero failures.
- Original AIME closed cleanly04:33UTC:852/1080 (mpk396,dllm456),71 whole blocks.
  Nineteen wholly unstarted blocks remain; zero partial/prefix/orphan conflicts.
  At06:34UTC AIME1049/1080 (mpk509,dllm540); quality unread.
  dllm complete; mpk c001 still running under amended finite budget.

Hosts and current AIME c001 workers (original supervisors exited rc0):
- mpk exouser@149.165.151.254, GPU-6139046a-b005-8fe5-a837-f8270472ab72.
  P=/media/volume/dllm-1/dyh/junyu_frontier_v18_20260926.
  Supervisor/PGID1671360, GPUworker1671384; originalruntime user-site ON.
- dllm exouser@149.165.159.64, GPU-fc12ad5c-5334-5509-8fc6-465498fd3915.
  N=/home/exouser/dyh/junyu_frontier_v18_20260926.
  c001 supervisor46382/worker46416 finished rc0 at1790489766; GPU now idle.
  Private copiedruntime/user-site overlay; no further AIME launch.
- Primary generation remains ROOT/deploy/bridge_json, source4d53b5c construction.
  Model/library/runtime paths are in frozen primary protocols/new CP3 config.
  Never modify active worker code/configs or restart requests to change budgets.

Frozen panels and fairness:
- ROOT/evaluation/ruler4k_primary_protocol.json:390 blocks/2808 executions.
- ROOT/evaluation/aime26_primary_protocol.json:90 blocks/1080 executions.
- Six arms D_native,D_matched,U50,U60,T50,T60; seeds101/202/303.
- Whole question-seed arm/warm groups stay on frozen GPU. No cross-host absolute
  time pooling; first output only for quality; failed/capped/unparsed remain.
- Ledgers ROOT/evaluation/ledgers/HOST_{ruler,aime}.jsonl; privateROOT/private_eval.
- Markers ROOT/evaluation/status/HOST_STAGE.{started,done}.json.

Approved finite extension (original budget file retained):
- Start2026-09-26T18:02Z;36h window. GPUcutoff2026-09-28T05:32Z, final06:02Z.
- GPU24h/host48h combined;7000 requests total3500/host; at most2 workers.
- Newfile results/junyu_frontier_v18_20260926/campaign_budget_extension_20260926.json
  SHA5a508a92ec822e428d76049902e1ebf1d199375b9c095537efa5a0de8696009e.
- Same475 preofficial attempts and baselineGPU1900s/100s charged. Sum actual
  worker_end durations; never additionally charge wrapper elapsed for same run.
- Existing original workers retain oldguard. Newcontroller waits for clean stop,
  verifies complete per-host block prefix, no partial/orphan results and quiet
  writers; then launches only missing hosts with new aime_c001 etc markers.
- Failed firstresult in a fully recorded block stays occupied and is never rerun.
  Unknown dispatch/partialblock defers continuation/finalization for review.
- No new questions/arms/seeds, at most5 densitycal policies/arm. Candidate remainsSTOP.

Active supervision: remote drivers/CP3 source6250b51; primary CPU source3bfed7d:
- Local cwd E:/dlm/v18_private/extension_6250b51; bothremote deploy/extension_6250b51.
- Archive SHAd827e54f4ed84b6222b98454b1b0e46b3a186ea7c62c195930a084a57f4527af.
- Primary controller PID51988, TTYsession21704; localcwd controller_3bfed7d; outcome
  E:/dlm/v18_private/completion_coordinator_outcome.json.
- CP3 waiting controller PID35304, TTYsession7655; outcome
  E:/dlm/v18_private/secondary_extension.outcome.json.
- Config E:/dlm/v18_private/secondary_extension_config.json,
  SHAa99c6eef0853f96c5bab260927d5fea2bbc90ae391e612df6cdd6e165cf1ad6f.
- Previous controllers49924/16656 verified/stopped; older47716/42488 exited on
  readSSHtimeouts. Their outcome files are preserved; do not use as currentstate.
- Tool session IDs may disappear across compaction: verify actual OS PID/command
  before replacement; Unknown process id is not evidence that the process exited.
-39 combined unittest cases pass; bothhost bash -n passes. CPU Torch/source/UUID
  qualification remains automatic AFTER primary offline scoring/quiet, not runyet.

Recovery only after verified exit, primary cwd controller_3bfed7d, local Python311:
python -u -m scripts.v18_coordinate --start-at aime
--continuation-deploy extension_6250b51
--budget E:/dlm/v18_private/extension_6250b51/results/junyu_frontier_v18_20260926/campaign_budget_extension_20260926.json
--after-stages-command-file E:/dlm/v18_private/finalize_command.json
--outcome E:/dlm/v18_private/completion_coordinator_outcome.json --poll-seconds 60
Persisted segment intent must be reused; never invent a new outcome to bypass it.

CP3 recovery cwd remains extension_6250b51:
python -u -m scripts.v18_secondary_pipeline --deploy extension_6250b51
--config E:/dlm/v18_private/secondary_extension_config.json
--outcome E:/dlm/v18_private/secondary_extension.outcome.json --poll-seconds 60

Flow: active aime_c001→RULERremainder→offline
primary scoring→CP3 CPUqualification→70densitycal/eval→allocationcal/eval→scoring.
Finalizer hook remains immutable scorer_f45fa83, exactgold/scorer identities; no
prior private archives/summaries existed at newcontroller installation. CP3 all10
possiblecal ledgers are budgeted; inherits exactprimary GPU. T70/T60 timing remains
cross-stage descriptive; allocationcontrols have no warm latency claim.

Heartbeat v18-research-stage-continuation updated to boundedextension through
Sep28; quietunchanged. Scripts do batches; inspect stage-level metadata only.
CP4: fetch redacted summaries, answer sevenv18 questions in morning_brief.md,
report actualdensity/work/quality/timing/missingcells and candidateSTOP honestly,
push all code/redactedresults, verify ownGPU jobsfinished, disable heartbeat.
Never upload answers/gold/credentials/largetraces or archive theuser task.

2026-09-27 recovery checkpoint:
- Original AIME worker_end GPU seconds mpk31008.756522101117/dllm31037.169668974;
  wrapper times are not added again. Snapshot2263 attempts incl475 preofficial.
- PID41020 exited on completion-marker/liveness TOCTOU; both rc0 done verified.
  Original failure saved completion_coordinator_outcome_exit_1790483717.json.
- Recovery PID49196 passed clean prefix/orphan gate and wrote c001 launch intent.
  Replacing CPU controller happened between host dispatches: dllm already started;
  mpk had no log/marker/remote process or surviving local SSH child. Manual review
  recorded c001_dispatch_recovery.json before its first dispatch. No request retry.
- Source3bfed7d rechecks matching successful done after supervisor disappears;
  unresolved absence still fails closed. Restart archives prior failure in outcome
  recovery_audit and clears stale terminal fields. Both c001 markers now adopted.
- Local controller archive SHA c0ef3d77ccfe817f986246d2d739fa23b468d78a7eafb00ce4d1505a16d92835.
  Never overlay remote immutable extension_6250b51 or bridge_json.

06:34UTC stage checkpoint: dllm AIME540/540 complete, c00184 executions,
worker_end5478.743522106997 GPU seconds (wrapper5482s is not added). mpk509/540
ongoing; combined1049/1080 and2460 campaign attempts incl475 preofficial.
Both CPU coordinators alive; primary awaits mpk before RULER remainder/scoring.
