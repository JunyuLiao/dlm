# Stage A — past-motion operator prediction and layer location

Starting commit7f04e2be, branch denoising_trajectory_research. New diagnostic
study; Stage B/C and production/kernel optimization are excluded. No campaign
GPU/time/forward/cost ceiling. Native output/canvas caps remain unchanged. One
H100 worker; no paused campaign resumed. Inputs, tensors, generations and logs
stay private. Yuhan reuse, Gaussian32, C-gate, V31/V2 retain their provenance.

The CPU addendum corrects derived work and exhaustive pair interpretation while
preserving originals, the AIME quality lead and HE/LongBench negative results.

## Exact identity and pre-treatment selection

manifest_identity.json freezes12questions, each seeds42/43:4AIME,4HumanEval,
4LongBench;2development/2confirmation within each benchmark. Exposed development
short IDs18/12 andHE124/70 are retained. Confirmation uses hash-ordered source
prompts>=256tokens outside prior25: AIME17/15,HE152/72. Earlier V31/V2 exposure
exists or is not excluded; these are method confirmation, not unseen benchmarks.
This context-stratified mechanism cohort is not representative evaluation.

Long development includes the known185546token original slow Single-DocumentQA
and hash-ordered short Code Repository31197token item. Confirmation uses short
Long-dialogue27181 and medium StructuredData107856token strata. IDs, exposure
and pre-treatment exclusions are in the manifest. Original9over-limit items
stay blocked. A prospective dense profile records difficulty for every frozen
cell without selecting or replacing questions using dense or sparse outcomes.
The targeted slow long-item cohort is explicitly distinguished.

## Engineering and ordinary native execution

Use fresh requests, no restoration. Common A uses PIECEWISE and the SAME SM90
FA4 paged all-kept consumers at layers5/17/29 throughout canvas1 in every arm.
Only one designated layer may delete support in A2. Other GLOBAL, LOCAL,
prefill and commit remain native dense. After canvas1 commit return to native
PIECEWISE. Corrected FULL+fix51994 and native PIECEWISE references expose consumer
differences. All native forwards, startup, warm requests, repeats and failures
are counted. No hidden/suffix forwards or changed native sampler/RNG/stopping.

First qualify HE124 seed42. For each cell run native PIECEWISE, common all-kept
measurement-disabled, common reference and common observer requests; one full
warm request per process. Full outputs,N,C,T,final RNG,forward counts and actual
first-canvas sampler/logit/RNG fingerprints and currentQKV routes must match
common controls. Observer work is RNG-bracketed. Actual native winners are
read and checked. Assert CURRENT cache tail equals current input K/V, fixed
within-canvas prefix hashes, GQAhead0→KV0/head8→KV1, queryblocks0/1 and exact
operator agreement<=0.03worst-row relative bf16 error at every tested layer.
Fresh-request complete equality does not repair the old restored-output failure.
Audit W,S,S/N are synchronized but timing_qualified=false: no clean diagnostic
speed claim. Native/all-kept consumer effects remain systematic treatments.

## A1 anchors, bank, budgets and scores

Matched A reference length L defines boundaries after max(2,floor(L/2)) andL−2
completed calls. Deduplicate, require2<=boundary<L; never replace a canvas or
force continuation. Current call is boundary+1; future is boundary+2 ONLY if an
ordinary call occurs before the same first-canvas native commit. Missing future
is explicit. Landmarks use an oracle reference, not online trigger information.

Layers5/17/29 are independent locations. Scope per layer is heads0/8 ×
queryblocks0/1, Q128×KV64. Wholly-prefix deletion per unit is
max(1,floor(0.05B)); identical absolute count across locations at the same prefix.
No infeasible work is reallocated. B<4 and non-identifying states are reported.
At most48distinct finite supports per unit: enumerate when possible; otherwise
low-mass, near/far, positional, independent hash sets, signed value-contribution
choices inside mass pools, and half-set multi-swaps. The bank is not mostly
single exchanges around a common support. Only current/past information is used.

Guards: worst-row relative output bias<=0.1, whole deleted mass<=0.2. Protect all
current-canvas/boundary keys; mass guard implies renormalization<=1.25. Four
feasible alternatives required for selection. Ties use ordered tile tuples.
Current-error minimizes current bias. q_past minimizes full response J_Su using
actual one-step past query motion. Separately named qkv_past includes actual
past current-canvas K/V displacement, with zero verified prefix motion. Past
operands are PRIVATE diagnostic inputs, never substituted into model attention.
direction_sham rotates feature coordinates and signs deterministically, preserving
each Q/K/V Frobenius motion scale. Full J_Su, J_Au, residual(J_S−J_A)u and Q/K/V
components remain separate. Worst-row response normalization uses current dense
row-output norm floor1e-8; no direction renormalization. No target change later.

support_sham uses SHA256seed19331 among the SAME feasible candidates within5%
current-error and0.01mass of current-error's choice. No native RNG. Full-response
attribution requires equal physical deletion, current error within5%relative
(floor1e-6), mass within0.01absolute, and>=25%full-response separation with
denominator max(abs(score_a),abs(score_b),1e-6). Record position bias magnitude and
private signed directions/vectors; scalar matching is not vector matching.
Immutable bank_choices.json is saved BEFORE the next native call; future data
go to a separate file. Only diagnostic attention operators are recomputed.
Compact chosen signed vectors are kept privately; no duplicated prefix storage.

## Prespecified prediction and advancement rules

Primary prediction: mean over rows of squared vector error normalized by current
dense row norm for o_S(t+1)−o_S(t), averaged across feasible candidates within a
routing unit. Compare q_past,qkv_past,direction_sham with persistence(0) and the
bias-carry/current-error forecast J_A(past joint motion), which carries current
sparse bias while forecasting common dense motion. Secondary residual forecast:
predict e_S(t+1)−e_S(t) versus carry-current-bias(0 change). Actual FUTURE Q/K/V
motion is evaluation-only derivative oracle, separating direction forecast
failure from local-linearization failure; never enters treatment selection.

Ranking utility is selected support's actual next attention motion, normalized
by current dense row norm; compare with current-error,direction-sham,support-sham.
This is operator ranking utility; low motion may suppress necessary correction
or remain uncertain. All feasible supports, missing futures and abstentions are
reported. No masks, units or seeds count as independent questions. Compare
within each fixed question/layer/anchor/budget. Mean paired log ratios over units
and anchors within question; descriptive10000question-cluster bootstrap draws,
seed20261025, retaining both seeds. Report each task/layer/anchor/question.

A score/location advances to A2 only if engineering passes AND all4units have
>=4feasible alternatives with at least one matched25%full-response-separated pair
at the primary late anchor for>=2development questions in BOTH seeds. For q_past
use q_past full-response separation; qkv_past uses qkv_past. Additionally,
>=2confirmation questions in BOTH seeds must show>=10%lower primary forecast MSE
than BOTH bias-carry and direction-sham, AND>=10%lower selected actual motion
than BOTH current-error and support-sham. Question-mean aggregate must improve
>=10%; missing cells cannot be dropped to pass. QKV must additionally improve
Q-only forecast>=10%. These are screening rules, not population proof.
A1 confirmation is prospectively run for all12frozen questions; no tuning on its
outcomes. If identification fails, record bank failure, not failure of every
response mechanism. No kernel optimization rescues prediction failure.

## Gated A2 and stopping this assignment

If no score/location passes A1, complete the A1 diagnostic leaf, mark A2 not
launched and stop. If one passes, freeze a NEW A2 implementation/source binding
before sparse native generation without changing this scientific specification.

Development compares late SS at layers5/17/29 for passing score families. Select
at most one leading score/location by smallest question-mean remaining native
first-canvas calls versus BOTH current-error/support-sham. Ties q_past then
qkv_past, then layer5/17/29. Do not increase number of pruned layers.
Confirmation runs common AA,SA,AS,SS at the selected late location for proposed,
current-error,support-sham; middle SS is secondary; retain layer29 SS if another
location wins. Fixed support uses CURRENT Q/K/V every application, no refresh.
After designated calls use common A through same-canvas termination, then native
PIECEWISE after commit. Early stop is retained; no second-call forcing or canvas2.
Deduplicate identical schedules/supports and disclose coincidences. Require exact
pre-intervention fingerprints/RNG/counters across fresh branches.

Causal screening: at least one fewer remaining native canvas1call versus BOTH
current-error/support-sham on both seeds for>=2independent confirmation questions,
favorable question aggregate, and no official task regression on the COMPLETE
prespecified finalist set. Complete original full output caps; official-score
proposed and controls regardless of apparent improvement. Report every
correct→wrong transition, no-final/extraction errors, native48-call canvas caps
and output caps separately. Signed reference-winner margins/entropy tails/
self-conditioning/native-state changes are diagnostics, never token correctness.
No exact compiled acceptance-count claim.

No successful oracle landmark becomes an online/every-canvas policy. Before any
Stage B recommendation specify online trigger and further small prospective
qualification. Stage B/C, exhaustive MarginGuard and production kernels do not
launch. Source/model/runtime/scorer pins and every consequential choice are frozen
in protocol.json before qualification/generation. Revisions use new attempts.

