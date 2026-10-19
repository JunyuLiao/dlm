# V31 failure map and conditional support-repair study

Authorized 2026-10-19 (Hong Kong). Starting repository checkpoint f4d3f349.
Yuhan's inherited attention-mass selector and cross-step reuse are the reference.
This is a new bounded diagnostic campaign, not a resume of the paused LongBench
full503 or cached-refresh studies. Canonical root:
`results/v31_failure_map_20261019/attempt001/`.

## Three-stage research plan

1. **Establish failures.** Map V31's accuracy and native convergence versus its
   prefix budget using corrected FULL dense, matched PIECEWISE dense, all-kept
   FA4, and fixed16 refresh controls. Screen a metadata-selected cohort, then
   independently repeat candidate failures with new seeds and model processes.
   Preserve all outcomes and distinguish a candidate, a confirmed case, and a
   population claim. This campaign can establish cases, not general superiority.
2. **Diagnose same-budget repair.** Only confirmed failures advance. At the first
   genuinely held sparse call after initial selection, independently replace the
   current attention output (dense versus the inherited sparse output) and future
   support (fresh same-K mass map versus held map). All four alternatives compute
   the same expensive diagnostic observation; it is never a cheap algorithm.
   Common-prefix checks, untouched V31 bookends, a sham intervention, and repeated
   native outcomes determine whether future support can repair the failure.
3. **Develop an algorithm.** Conditional future work: use the observed repair
   mechanism to design causal, cheap admission/lifetime prediction or selection.
   Keep the native sampler unchanged; validate on new questions and measure
   complete selection plus generation cost. No new production selector or
   training campaign is authorized by this diagnostic execution plan. If stage 1
   finds no confirmed cases, stop rather than weakening gates or forcing stage 2.

## Fixed model and comparison contract

DiffusionGemma-26B-A4B, existing pinned vLLM/SM90 FA4, one H100, batch one.
Corrected FULL causal mask (#51994) and LOCAL-window fix are mandatory. Native
canvas256, maximum48 calls/canvas, confidence0.005, stability1, entropy-bound0.1,
temperature0.8 to0.4, thinking on, and each manifest's original output budget are
unchanged. Existing official LongBench rendering/truncation is preserved and
reported; this is not an untruncated-context claim. No output truncation to meet
the runtime cap, hidden native forwards, fixed-step stopping, or extra RNG draws.

Only eligible GLOBAL prefix budgets change: 8192/4096/2048 tokens, respectively
128/64/32 KV64 blocks per head/Q128 unit. Five GLOBAL layers, dense LOCAL, prompt
prefill, boundary/canvas protection, and commits. Keep step1 selection, first-call
carry, settledness threshold0.15 and sticky1.386. Fixed16 changes only refresh
calls to zero-based16/32. Every budget is named separately; absolute physical
work includes initialization/carry/refresh and may differ with the trajectory.

## Cohort and stages

Select four LongBench questions per rendered-token band [16K,32K), [32K,64K),
[64K,96K), [96K,120K+256), by SHA256 of the frozen salt, dataset and index. No
generation result participates. These inputs have been examined historically;
this is development screening, not held-out confirmation. Reserve a separate
metadata-selected short and long warmup question outside the 16 questions.

- Qualification: all three budgets and all three dense controls, one short and
  one long qualification input, separate instrumented receipts; prove fixed-K,
  GLOBAL/LOCAL scope, source/seed/config identity and no measured CUDA captures.
- Screen: all16 questions, seeds1/42, clean execution. Each K has V31 before/after
  bookends. K8192 additionally has fixed16. FULL/PIECEWISE/all-kept each cover the
  same32 cells. Total320 measured requests, excluding warmups/qualification.
  Rotate question/seed order and family order; all arms run on the same H100.
- Confirmation: choose at most one K and eight candidate questions using the
  rule below. Seeds43/44, two independent engine repetitions, three dense
  controls and V31 bookends plus fixed16. At most192 measured requests.
  This is independent replication on selected questions, not held-out inference.
- Repair: at most four confirmed questions at that K, seeds43/44, two independent
  processes, four factorial alternatives plus V31 before/after (at most96
  measured requests). Audit-only: never report its latency as deployable speed.

## Failure and promotion gates, fixed before generation

Screen population flags: V31 minus FULL accuracy <=-3 percentage points, or
pooled calls/canvas >=1.10 times FULL. Report question-cluster intervals, not
just flags. Controls and length/cap effects must accompany every aggregate.

Candidate **case** requires the same failure type on both screen seeds:

- Accuracy: both V31 bookends wrong, corrected FULL and all-kept correct.
- Convergence: both V31 bookends and FULL/all-kept correct; the smaller V31
  calls/canvas exceeds both reference ratios by >=10% and >=1 call/canvas;
  neither V31 nor those references hits the output cap. V31's excess must be
  >2 times its own bookend difference. This is descriptive outcome stratification
  used for case selection, never an unbiased population benefit estimate.

Choose the **largest K** with candidate cases (8192, then4096, then2048), avoiding
an unnecessary extreme-budget focus. Select up to8 by the same metadata hash,
not severity. Frozen confirmation uses both new seeds and both engine repeats.
A confirmed case must satisfy its original failure type in all four cells,
now against FULL, PIECEWISE and all-kept, with no V31 score drift. Missing,
timeout, ambiguous control evidence or a type switch does not pass. Confirmation
continues to report all selected cases, including reversals. A population flag
without reproducible cases cannot launch repairs under this bounded plan.

Repair qualification requires matching recorded native prefixes before the
first treated GLOBAL layer, correct same-K/tail masks in all5 layers, original
sampler/RNG and no added model forward, all alternatives exposed, and exact
V31/sham output-N-C-T parity. Nonmatching groups remain reported as unresolved.
Fresh-support rescue is only a mechanistic lead if the same direction appears
in both processes and seeds without worsening official score. Report every
case and uncertainty; one early intervention cannot rule out a later repair.
Attention error, retained mass, fewer calls, or nonsignificance cannot by
themselves establish answer quality or noninferiority.

## Budget and stopping

Hard H100 reservation cap **14,400 seconds (4 hours)**, including interpreter/model
startup, compilation, warmups, failures, partial requests, and device tests.

| Phase | Maximum reservation |
|---|---:|
| Qualification |900s (15min)|
| Screen |5400s (90min)|
| Confirmation |3600s (60min)|
| Repair |3600s (60min)|
| Recovery |900s (15min)|

No automatic borrowing or cap expansion. Check idleness and acquire one shared
GPU lock; never stop another user's worker. Each child gets a deadline bounded
by phase/global remaining time and its own process group. Stop only that group
at the deadline; retain partial records but do not count incomplete comparisons.
No new engine starts with <180 seconds available. Recovery requires an explicit
frozen correction and uses only the reserved900 seconds. Old attempts remain
immutable; resuming a failed leaf requires a new leaf. Every reservation is
closed or charged its full bound if orphaned. Write terminal state and GPU-idle
receipt. No waiting for another GPU worker is charged as active GPU reservation.

## Metrics and evidence

Clean official accuracy, pass+finished, N, C, pooled N/C, T, output caps,
median/P90/P95/maxN, W, S, amortized S/N, paired differences and question-cluster
bootstrap intervals (20,000 draws, seed1729); same-arm bookend drift and
cross-engine/seed sensitivity. N/C is aggregate and does not causally align
different canvases. Report both total calls and output length.

Audit separately reports actual GLOBAL eligible/kept, LOCAL eligible/skipped,
their denoising-only total denominator, initialization/refresh/held/carry work,
and all extra observer work. Audit results never substitute for clean outputs;
if their trajectories differ, say so explicitly. Clean masking is checked by
live settings, native counters and unchanged source, not invented audit fields.
Official scoring binds raw completions privately to manifests, seeds and
budgets. Public files contain numeric outcomes/hashes only. Source/spec commit
precedes frozen protocol, snapshot, qualification and generation. Completion
markers distinguish completed phases from budget-limited or gated-out stages.

## Novelty boundary

Mask selection/reuse is inherited and already represented by MAGE/SparseD;
periodic refresh is a control. The prospective algorithmic contribution is
preventing harmful native-decision changes at useful sparsity, with measurable
net benefit. The prior refresh-cost no-go concerned deleting refresh expense
on fixed trajectories; it does not bound savings from fewer whole-model calls.
