# Active task September22 2026,08:50UTC

User wants six intervention families preserving short denoising at high OVERALL
sparsity, plus<=2development-selected combinations and clean final runtime.
Baseline completed and must not be rerun. Full task is NOT finished.

Canonical output results/value_direction_interventions_v2 resolves to migrated
disk /media/volume/new-dllm/dlm-migrated-20260921/dlm/results/...
GPU worker1253279 detached, lockprotected, command intervention_study run.
Log worker_1790065998.log. Watcher tool session1276,15minute checks,
intervention_watch.py; last08:50alive,421calibration,65development,0final,
4/25policies frozen,0errors. Smoke worker1252975 already finished.

Files:
- intervention_policies.py:Controller + native sampler/step wrappers.
- intervention_study.py:fullworkflow/calibration/devselection/final/timing.
- intervention_study_report.py:raw-onlyaudit/report/figures; not yet exercised
  on a complete study. This report file may be edited; worker imports at end.
- INTERVENTIONS.md protocol.
- tests/test_intervention_policies.py plusprior tests:11passed total.

Frozen running source files MUST NOT be edited. Allsourcehashes archiveinroot.
Only reporting/docs/tests can be safely changed during currentexecution.

Implemented25independent configurations:
first30/40/50/60/70;late40/50/60/70;reactive60/65/70;
protect60/65/70;accept_x1/x3at60/65/70;extra_random/rankedat60/70.
Exactly same native48stepmax and temperatures/stopper; no fixed4inthisstudy.
All-retainedHopperkernel implementsdensecorrections withmatchedmask semantics.
Protection uses last-stepstable top1,p>=.999,margin>=.98, reopenschangedtop1 orp<.98;
no attentionqueryskipping. Adaptive entropy budget=base*(1+strength*acceptedfraction).
Extraacceptpool32lowestentropyrejected withp>=.5,min(4,poolsize), nativeproposal
tokenskept;independenttorchgeneratorforrandom,no nativeRNGadvance.
Reports disclose B/Cquotas can differ ondivergedrollouts; posthoc identicalstate
randomexpected vsrankeddisagreement withsamepool/k inreport.
Dense-token disagreement isNOTgroundtruthincorrectness.

Reactive thresholds derive from26densecalibration trajectories:churn95quantile,
entropyprogress5quantile,acceptprogress5quantile;requiresonecompletedtransition;
no consecutive reactive densecalls. Basic unit tests pass, smoke18cases passes
exactinstrumented/uninstrumentedoutput andcounts. Plain/dense areincluded;
archivebaselineparity not explicitly rerequired inthisstudy(no baseline rerun).

Calibration26disjointquestions,independent from13development and130final.
Commonoffset tovalidateds70local/globallogtau. Grid[-3,-2,-1,0,1,2,4] +3refinements,
wholegenerationcountweighting incldenseiterations;2pp tolerance.
Fullgrid toretainnonmonotonicresponse. Cache bypolicy+thresholdhash permits
reusingpointsacrosstargets. Missingtargetnearestpoint +maxobserved reported,
no unsupportedglobalinfeasibilityclaim. Highesttestedtau is not absolutemax;
report conditional upperbound due densecalls separately.

Allpoliciesdev-evaluate13examples beforefinal. Selectionfunctionprefers
accuracy>=dense-2pp ANDmean<=6, then highestactualsparsity;otherwise same
sort amongall. Chooses<=2late/reactiveschedule parents andbestprotect/accept
parent, creates<=2combos, recalibrates,devchecks. Timing2selected ondev only.
Allindependent+combosevaluatefull130 even ifofftarget (report limitation).
Then2repeats all130fornative_dense,kernel_dense,2selected, diagnosticoff
butpolicy-requiredsignalsincluded;warmup1prompt/condition,groupedtimingorder.
Timinggenerationparityreport must pass.

Oldintervention.py andinterventions_smoke{,_v2} are flawedpreliminary artifacts:
firstprotected2stepsdueto0basedruntime;reactivenotriggerimplemented;notresults.
Failedsmoke KeyError fixed but oldartifacts preserved. Deletionrequestwas
rejected;no deletionoccurred. Canonical controllerown1basedstep testedcorrect.

Current integration.py hasoptional policy_selector defaultNone, returnNone
mapsdense -infthreshold; kernelmathunchanged. Baselinefrozenoldhashesarchived.

User paperURLs LoSAarxiv2604.12056 andStreamingdLLM2601.17917 attempted viaweb
open/search,toolsreturnedempty. No externalpaperclaims; followuseralgorithms.

Remaining work:
1.Monitor15min,reportonlycounts/current/errors/ETA asrequested.
2.Investigatefailure/stall,never alterrunningfrozenconfig silently.
3.Afterallruns reportaudit,inspectplots,independentchecks/regeneration.
4.Add quantitative explicitcomparison answers ifreportneedsmoreinterpretation;
  same-actualsparsity andpairuncertainty required; no predeterminedwinner.
5.Answerdensefirstshift,lateefficiency,reactivevsfixed,protection,acceptance,
  highestsparsity~4steps,latency usingresults. No causalpercent attribution.
6.Checkpolicycoverage25+combos,missedtargets,reportcalibrationvsfinal drift,
  cap48meanscensoring,overfitting/exposurelimitations.
7.DoNOTfinishbymerelylaunching;finalresponseafterdata/audit handled.
