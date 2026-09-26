# v18 continuation: CP2 frozen primary evaluation

Sole execution authority: user v18 (SHA in cp0_identity.json).
Branch research/astra-junyu-frontier-20260926; parent d2bef2189c046e9a4277ab16d751ebeab11c15d8.
Read DESIGN_DECISION.md. Prior WSL v8 and remote v15 untouched.
Ignore unrelated Windows case-collision third_party/dinfer/assets/Wechat.JPG.

Completed:
- CP0 historical identities passed; 120 historical warm pairs lack strict phase evidence.
- Native adaptive stopping preserved: stable argmax AND mean entropy<.005;
  acceptance .1, B256/cap48, temperature .8 to .4 and normal EOS unchanged.
- RULER thinking=False, exact156 calibration/eval token lists; AIME thinking=True
  with30 exact token lists and8192 output budget frozen. Initial failed capture
  remains counted/preserved (1 request,24 GPU seconds).
- Native legal all-layer scope qualified13 states. LEGACY LOCAL omitted9.97%
  of captured native-legal pairs; geometry diagnostic only.
- Density-only calibration442 requests: U50/T50/U60/T60 all within2pp whole/G/L.
  This does not establish T60 quality, calls or speed.
- Strong B8 old/new16 requests equivalent,8 strict warm accepted; new/old wall
  geomean1.000187, no speedup signal.
- Bridge6 requests per host passed full tokens/canvas calls/stop/cap/termination.
  All1047 model tensors identical; new index is declared shard repacking.
- Single post-QK ball candidate stopped before implementation at cost/numerical
  gate. Ideal GLOBAL skip coverage58.84%, LOCAL11.32%; no finite certificate
  or demonstrated net execution headroom. No alternative candidate or CVM work.

Hosts / immutable generation deployment:
- mpk exouser@149.165.151.254, GPU-6139046a-b005-8fe5-a837-f8270472ab72
  P=/media/volume/dllm-1/dyh/junyu_frontier_v18_20260926
  Python=/home/exouser/miniconda3/envs/ljy_dlm/bin/python, original user-site ON.
- dllm exouser@149.165.159.64, GPU-fc12ad5c-5334-5509-8fc6-465498fd3915
  N=/home/exouser/dyh/junyu_frontier_v18_20260926
  Python=N/bridge/runtime/miniconda3/envs/ljy_dlm/bin/python; private old user-site
  explicitly overlaid, PYTHONNOUSERSITE=1. See run_bridge.sh environment.
- Both generation cwd=ROOT/deploy/bridge_json, exact4d53b5c code construction;
  config paths/hashes bind these files. Never mutate this immutable deployment.
- Local scorer/driver additions must not cause GPU workers to import unverified
  source. New driver invokes pinned module from that cwd/PYTHONPATH.

Frozen protocols in ROOT/evaluation on both hosts:
- ruler4k_primary_protocol.json:390 blocks,2808 runs. Initial78 blocks=936 runs
  (26 preselected timing questions x3seeds x6arms xfirst/warm).
- aime26_primary_protocol.json:90 blocks,1080 runs.
- Whole question-seed block (all arms and warm) stays on assigned GPU, ordered
  round-robin mpk/dllm. IDs/digests are in RUN_MANIFEST.json.
- Execution: RULER initial -> AIME complete -> RULER remaining -> secondary70/
  allocation if budget. Generation processes have only gold-free manifests.
- Scorer-only source gold byte hashes frozen. RULER original v15 code_cp1
  results/query_adaptive_v3/configs/final_manifest.json; AIME
  /media/volume/dllm-1/dyh/numerical_qk_global_multiseed_20260925/private/manifest_all30.json.

Budget / next action:
- Start2026-09-26T18:02Z; stop GPU by2026-09-27T05:32Z, final deadline06:02Z.
- Cap7000 full requests and20 total GPU-process hours, two qualified hosts.
- Before eval475 completed requests, measured old1484s/new28s plus120s reserve;
  conservative frozen baselines old1900s/new100s in campaign_budget.json.
- Per-host quotas conservatively sum to caps; hard timeout required in addition
  to complete-block guards. Initial soft deadline21:03:20Z; AIME reserved1080.
- Initial workers launched from driver_5b14a21: mpk supervisor1614993/child1615015;
  dllm supervisor8248/child8262. Status under ROOT/evaluation/status and ledgers
  under ROOT/evaluation/ledgers. Sol-A local coordinator adopts these workers.
- Sol-B prepares secondary CPU protocol helpers, no GPU launches.
- Scores only from first outputs; partial/failed/capped/unparsed retained.
  Warm accepted only if full required evidence matches, no JIT/new SO.
- No cross-host absolute-time pooling. Host strata and within-GPU ratios.
- No raw answers/gold/credentials/large traces in Git. No peer branch edits.

Live checkpoint: initial RULER936 executed (468/host), zero first failures,
468/468 strict warm accepted. Source ledgers SHA-verified and copied between hosts.
Initial wrapper625s mpk/613s dllm; use worker_end for quota accounting, not both.
AIME supervisors: mpk1616556 PGID1616556; dllm9108 PGID9108. Both active.
One local primary coordinator: PID49924, approved TTY exec session14087,
immutable db79c828 under E:/dlm/v18_private/coordinator_db79c82. Previous local
PID39296 was verified/stopped; remote GPU workers unchanged. Earlier PIDs49208
and50660 are stopped too. Tool session handles can disappear across compaction;
verify actual OS PID/command before replacement, never infer process exit from
an Unknown process id response. Persistent controller state:
E:/dlm/v18_private/primary_coordinator_outcome.json.
Final hook E:/dlm/v18_private/finalize_command.json uses immutable local
scorer_f45fa83/scripts/v18_finalize.py and remote deploy/scorer_f45fa83.
Primary failures/partial AIME stop downstream work; coordinator waits for own
supervisors and ledger workers to exit, resyncs all four ledgers, then scores
available outputs. Unreadable remote state blocks scoring, never pretends quiet.
Recovery after confirmed controller exit: cwd coordinator_db79c82, local
Python311 -u -m scripts.v18_coordinate --start-at aime
--budget results/junyu_frontier_v18_20260926/campaign_budget.json
--after-stages-command-file E:/dlm/v18_private/finalize_command.json
--outcome E:/dlm/v18_private/primary_coordinator_outcome.json
Use approved foreground TTY; adopt existing markers, do not relaunch GPU workers.
Hourly task heartbeat v18-research-stage-continuation remains active; quiet when
unchanged, original budgets apply, disable after CP4. CP3 core f64e0b9 committed;
Windows CP3 pipeline is now deployed and waiting; no CP3 GPU work launched.

CP3 waiting controller: local PID47716, approved TTY session18255, source1c9a6e62.
Cwd E:/dlm/v18_private/cp3_1c9a6e6, both hosts deploy/cp3_1c9a6e6.
Archive SHA7204840b22b6b18078c2e89beb25524c5f88b4406faab5a857ad83c55a137aab.
Config E:/dlm/v18_private/secondary_config.json;
SHA55bc72e47887648c697393be2cc574d34300260b631c9d7584c59451aeeea43a.
Durable status E:/dlm/v18_private/secondary_config.outcome.json.
Recovery only after verified controller exit: local Python311 -u -m
scripts.v18_secondary_pipeline --deploy cp3_1c9a6e6
--config E:/dlm/v18_private/secondary_config.json --poll-seconds60
(use --poll-seconds 60 as separate CLI arguments).
Gates: exact-ID complete primary summaries + closed primary stage markers;
then actual-host CPU pytest/Torch/source/UUID checks, not yet run. All10 possible
mpk calibration-point ledgers and both secondary stages are in frozen inventory.
Sequence core70 calibration/eval -> independently calibrated allocation controls;
whole question/seed retains primary GPU. T70/T60 timing is cross-stage descriptive.
Failed/missing70 blocks prevent controls. Terminal errors preserve partial outputs;
if peer writers are still live, partial scoring is deferred in outcome. Heartbeat
should retry offline scoring after verified quiet, never duplicate GPU dispatch.
Primary and secondary scripts do batch work; inspect only stage metadata/outcomes.
Final CP4: fetch redacted summaries, answer v18 seven questions in morning_brief.md,
report actual attained densities/missing cells/budgets, push, confirm own GPU idle,
and disable heartbeat. Fast-path candidate remains stopped; no further CUDA work.
