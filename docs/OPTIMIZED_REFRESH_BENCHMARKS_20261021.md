# Optimized refresh benchmark comparison, 2026-10-21

Explicit user request supersedes the previous self-imposed dense-parity gate:
compare current optimized online V2, optimized V31 and dense controls on AIME26,
MATH500, HumanEval and LongBench-v2. This is a new campaign; no old paused run
is resumed. Preserve Yuhan cross-step-reuse and Junyu Gaussian32 provenance.

Use all30 AIME26 questions at seeds42/43/44, all500 MATH500 questions at seed42,
all164 HumanEval tasks at seed42, and all503 official LongBench-v2 problems at
seed42. Thinking-on throughout. Native sampler/output limits unchanged:256
canvas,48 max calls, entropy_bound0.1, confidence0.005, stability1,temperature
0.8 to0.4. Pin deterministic Inductor policy and corrected FULL51994 and LOCAL
FA4 mask policy. Single H100, batch1, one worker at a time.

Fixed prefix token budgets: AIME26/MATH500 K1728, HumanEval K128, LongBench K8192.
GLOBAL layers5/11/17/23/29 only; LOCAL/prefill dense. Carry-first, selection at
call1 and settle progress0.15 refresh with held bonus1.386 are inherited.
V31 dense refresh uses the optimized FA4 exponentiated-score observer and fused
qblock-max topk/list. V2 uses frozen7300f103369c804d Q128 BF16x3/fastmath schedule2
true-online kernel, charges current-V projection/RMS and contiguous packing,
skips full PV online, and returns its sparse output directly. Held/carry
consumer is common inherited paged split FA4. Numerical V2 output is a new
experimental variant and inherits no previous task accuracy claim.

Arms: v2_online_optimized, v31_optimized, dense_full_fix51994, dense_piecewise,
allkept_fa4. Fresh source/runtime/data-matched controls. Reuse existing V2
thresholds as starting candidates (HumanEval0.5600716206066294, AIME/MATH0.021,
LongBench0.021); check first-canvas absolute physical GLOBAL full-PV kept tiles
under the new sparse refresh output. Select thresholds using computation only,
never accuracy or latency. On deterministic prompt-length-stratified calibration
items, evaluate a bounded grid and up to two refinements, then freeze nearest
absolute kept-work match. Target<=2% absolute first-canvas work difference and
<=0.25 percentage-point physical kept-fraction difference. If unattainable,
report mismatch explicitly; do not alter policy or fixed-K control. Full panel
first-canvas work is audited independently; calibration is transductive.

Qualification: kernel and adapter tests, native smoke and repeated-cell clean
versus audit output hash/N agreement, explicit selector/threshold/kernel/library
receipt, all5 GLOBAL/25 LOCAL calls per denoise and zero LOCAL skips, fixed V31
budget and output finite checks, zero capture during clean timing. Warm one
full native request per worker. Separate audit and synchronized request/prefill
boundary clean pass. Preserve all failed attempts; use new attempt directories
for relaunches. Per-refresh events are audit-only, never clean latency.

Report official accuracy, paired/item-clustered95% CIs, GLOBAL/LOCAL/overall
physical accounting, first-canvas and total absolute kept tiles, N,N/C,T,S/N,S,W,
refresh latency and counts, graph captures and source identity. V2 QK/PZ repeats
four times across output groups; full-PV kept-work matching does not equal total
FLOPs, and duplicated routing/projection is fully charged in runtime.

Private manifests/gold/raw generations remain under the new campaign private
root. Public sanitized receipts and aggregates only. Final claims require all
coverage, source/protocol receipts, clean timings, official scores and completion
markers. No inferred speedup from sparsity or fewer forwards.
