# Active sweep, September21,2026

User requested adaptive Gaussian32/aggressiveBLASST sparsity40/50/60/70 vs
mean denoising calls per256-token canvas and accuracy, plus exact4-step dense /
Gaussian32 / aggressiveBLASST at50/70. Include dense reference, identify the
rise in steps and whether later steps within4 damage good drafts.

Active canonical worker: PID1208170. Root results/value_direction_sparsity_steps_v2
resolves to /media/volume/new-dllm/dlm-migrated-20260921/dlm/results/value_direction_sparsity_steps_v2.
Log worker_20260921T113950Z.log. Detached, lock protected. No other GPU job.
Health watcher session95815,15-minute interval, writes health.jsonl.

The disk migration exposes results/model caches via old symlinks. Disk503GiB
free. Kernel bridge provenance formerly compared raw resolved path strings;
cuda.py now resolves saved source aliases and still requires exact binary digest
and Torch version. No GPU code/algorithm changes. Tests test_sparsity_steps.py
and test_value_direction_trajectory.py:8passed. Smoke passed eight exact
instrumented/uninstrumented/archived outputs+step comparisons on2prompts and
exact4-call schedule checks. Binary873c3073e54a37b8 and bridge99ec are unchanged.

Worker reused520 compatible adaptive outputs:130each native_dense, kernel_dense,
gaussian32_s70,blasst_s70 from trajectory_v2. It calibrates40 independently
on26disjoint prompts,2pp tolerance, local/global scalar bisection, no final-score
selection, no huge QKV scratch. AggressiveBLASST40 permits above1 to independently
match local/global40; this is explicitly recorded (it is not the capped path).

New adaptive runs40/50/60/65 are on the same Hopper binary. Frozen50/60/65/70
policies reused from historical runs; historical generations with different
numerical backends are not mixed into the primary curve.65is an extra point
to identify60-65vs65-70 rise. Total adaptive12conditions x130=1560.
Fixed4 has6conditions x130=780: native_dense,kernel_dense and two sparsefamilies
at50/70. Total2340. Fixed4 suppresses native stopper and request.steps=4,
native temperatures0.8,0.7,0.6,0.5. Allprompts1canvas, official returnedbudgets30-128.
Draft tokens/scores/would-stop saved at every step. Adaptive stillmax48.
4step compresses annealing: don't claim stopping-only comparison to48.

Failed legacy40 root s40_v23 was preserved after diskfull. Its former launcher
had an improper check_sources no-op; this has been removed and main now raises
a superseded error. No old shards relabelled. v1new sweep startup failed on
migrated bridgepath, preserved too. v2 is canonical. Do not resume old launchers.

Worker source sparsity_steps.py and kernel interface hashes are frozen.
Do not edit them during running. Report is separate sparsity_steps_report.py
and can improve before import. It runs after each regime, validates all raw
counts/steps/IDs/thresholds/settings/sourcehashes and generates summary,pairedCI,
kneeintervals,plotsPNG/PDF,raw percanvas and perstep scores.

Remaining:
- Wait/monitor~15min. Investigate failures and calibrator if needed.
- After all2340 outputs, fresh report regeneration; inspect plots and independent
  audit, verify regeneration hashes. Current report newly added fields include
  adaptive/fixed4 first-draft score equality; exacttokens checked if both retained.
- All130 are previously examined; no independent validation claim.
- Report measured sparsity, accuracy, mean/median/p90,cap48counts; don't claim
  capped48 equals convergedrequiredsteps. Kneedetection is interval/descriptive.
- Report fixed4 persteptransition counts and correct->incorrect examples,
  distinguish retrospective bestdraft from implementable stoppingcriterion.
- Provide concise final results and figure/report links; don't end merely after
  launching. No kerneloptimization requested in this task.
