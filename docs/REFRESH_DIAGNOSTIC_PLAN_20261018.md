# Bounded V31 refresh diagnosis — preregistered 2026-10-18

This development diagnostic asks whether useful optimization headroom remains,
and whether a refresh helps through its current dense output, its future support,
or both. It does not promote a new method or resume the paused full503 campaign.
The inherited attention-mass/cross-step-reuse implementation is Yuhan's work.
The intervention and profiling instrumentation are separate study variants.

## Scope and immutable execution

Commit this specification and the diagnostic implementation before freezing the
execution snapshot and generating answers. Use a new attempt under
`results/v31_refresh_diagnostic_20261018/`; record the source, runtime, model,
input and schedule hashes. Never overwrite a failed attempt. The local idle H100
80 GB is the only GPU, with a process lock and one worker at a time. The pinned
environment and model are read-only. Existing campaigns remain paused.

Preserve native proposal, acceptance, self-conditioning, renoising, stopping,
temperature, thinking and original generation budgets. Canvas=256, maximum
denoising steps=48, entropy bound=0.1, confidence=0.005, stability=1, temperature
0.8 to 0.4. Preserve dense prefill/commit and LOCAL attention. V31 uses Q128/KV64,
8192 eligible prefix tokens per selection unit, mandatory canvas/boundary tiles,
all five GLOBAL layers (5,11,17,23,29), initial selection at call1, carry and
sticky=1.386. The inherited progress threshold is 0.15. This is not the Slack
stickiness-off comparison. Fixed16 changes only the progress clock to zero-based
canvas calls16 and32; call1 and carry remain.

## Budget and cohort

Hard cap: 7200 seconds of GPU reservation, including startup, warmup, profiling,
failed attempts and any successor attempts. Allocate qualification900s,
headroom1500s, factorial3900s, reserve900s. Per-phase limits and the campaign
deadline are enforced. Do not start an engine without enough remaining reserved
time. If a deadline interrupts a request, preserve it as incomplete; never
shorten native output or iteration limits to make a run fit. No automatic
full503 expansion is authorized by this plan.

Choose eight LongBench-v2 questions, two from each prompt-length bin <32K,
32–64K,64–96K,>=96K, using a deterministic hash of dataset/index metadata,
independent of previous scores or denoising calls. Choose four RULER128K items
from distinct task families using the same rule. Seeds1 and42; two independent
engine repetitions per suite. The protocol publishes the chosen indices before
generation. These inputs were previously examined; all findings are exploratory.

## Qualification and 2x2 diagnostic

One first genuine mid-canvas progress refresh per request is eligible: an old
same-canvas sparse map exists, the native clock fires after initial selection,
and the eligible prefix is larger than K. Do not substitute an initialization or
commit if no such event occurs. No exposure is an outcome, especially on RULER.
At the first eligible event, treat all five GLOBAL layers. Deeper-layer input
changes caused by an earlier treated layer are part of the intervention.

| Arm | Current attention output | Map used subsequently |
|---|---|---|
| dense_fresh | Exact dense | Exact newly selected map |
| dense_hold | Exact dense | Pre-refresh map |
| sparse_fresh | Native sparse on pre-refresh map | Exact newly selected map |
| sparse_hold | Native sparse on pre-refresh map | Pre-refresh map |

Each factorial arm computes both dense and sparse alternatives at this single
episode, using current Q/K/V; no full model forward or sampler call is added.
This keeps the current sparse operator common to the two sparse treatments.
The original progress clock advances in all arms. Hold restores both the list
and bitmap, with identical fixed prefix budgets and mandatory tiles. Subsequent
native scheduling is unchanged. Dense observation and extra sparse oracle work
are counted separately. Factorial timing is instrumented and never a speed claim.

For each question/seed/engine run original V31 before and after the four arms;
rotate treatment order and reverse it in the second engine. Warm each treatment
before measured groups. Maximum measured factorial requests=12x2x2x6=288,
excluding explicit warmups. Qualification first runs the six-arm design on one
predeclared medium-length LongBench item. CPU lifecycle/budget tests must pass;
native receipts must prove selected layers, exactK, unchanged LOCAL work,
unchanged RNG by the diagnostic, no extra native calls and no new graph capture.

Hash the pre-intervention compact sampler history, current sampler state,
self-conditioning, RNG, support maps and first-treated-layer current Q/K/V.
Inputs are not saved in public outputs. Exact matching of all four arms' prefix
signatures is required for a common-state causal contrast. This is a repeat-run
diagnostic with measured matching, not an assumed literal cloned full engine.
Report all mismatches and outcomes; never retry or filter to favor a method.
Dense_fresh is the no-op intervention control. A/A bookends and engine repeats
measure runtime/trajectory variation. A no-op discrepancy is investigated before
attributing downstream differences to the treatment.

## Separate runtime headroom pass

At batch1 choose two fixed LongBench inputs around32–48K and96–120K from metadata.
Run two clean repetitions per input for inherited V31, fixed16, corrected native
FULL dense, matched PIECEWISE dense, and all-kept sparse consumer. Warm separately,
use native graph modes and actual capture counters. Clean timing has only native
prefill/request synchronization boundaries and no diagnostic/profile hooks.

Use separate audited runs on the same inputs for physical GLOBAL/LOCAL work,
initial observation, progress refresh, sparse reuse/carry, dense commit, prefill,
KV gathering, selection and consumer costs. CUDA-event spans are inclusive and
must not be added to their children. A single bounded PyTorch/CUDA profile per
engine estimates disjoint activity categories and the union of busy intervals.
Keep unclassified/graph-replay activity explicit; GPU gaps are not automatically
host idle. Do not present activity shares as critical-path speedup bounds.
FULL, PIECEWISE and all-kept controls expose substrate cost and numerical effects.
The current native adapter and harness are single-request; B2/B4/B8 serving is
outside this bounded study. No claim about larger batch follows from batch1.

Report accuracy, output tokens, finish reason, N,C,N/C, decode S, request W,
amortized S/N, peak memory, graph captures and physical eligible/kept tiles by
scope. Record observation frequency/cost and amortization by useful reuse steps.
Use request/phase costs to estimate optimistic removable refresh overhead, then
subtract candidate cost before proposing a realizable latency improvement.

## Decision rules (research triage, not confirmation)

1. Structural failure (wrong scope/budget, added native calls, changed sampler,
   RNG consumption, source drift, missing mandatory receipt or timed graph
   capture) stops promotion and preserves the failed attempt. A fixed successor
   needs a new source snapshot and attempt within the same total GPU budget.
2. Fewer than six distinct matched/exposed LongBench questions, missing seed or
   engine coverage, or no-op/A/A discrepancies comparable to treatment effects:
   causal result is unresolved. Report exposure on RULER separately; zero
   exposure means this intervention does not address its initialization/commit
   bottleneck. It does not imply refresh is unnecessary.
3. A deployable refresh optimization is worth further work only if conservative
   accounting identifies a plausible >=10% incremental request-latency benefit
   over optimized V31 in an explicitly named useful regime. If even an optimistic
   zero-cost refresh bound is below10%, stop investing in refresh for that regime.
   Unknown overlap/attribution yields an unresolved bound, not a positive result.
4. Treat current dense output as a priority if its removal consistently changes
   next-step state and causes >=5% mean N increase beyond A/A variation, or at
   least two reproducible adverse answer flips across repeats. Apply the same
   screen to future support. Require stable signs across seeds/repeats and inspect
   leave-one-question-out/tail sensitivity. These are mechanism-screen criteria,
   not statistical accuracy/noninferiority guarantees.
5. Output-dominant: investigate selective exact execution/correction; support-
   dominant: investigate cheap support construction; interaction-dominant: design
   a coupled mechanism; neither decisive: retain fixed scheduling and investigate
   initialization/commit/cross-canvas amortization only if headroom supports it.
   If runtime headroom fails, stop selector variants on this configuration and
   first qualify a different model/workload/serving regime.

Report question-clustered intervals preserving seeds/repeats, simple factorial
contrasts, interaction, A/A differences, all failures, missing cells and tail
dependence. No nonsignificant contrast establishes noninferiority. No top-tier
novelty or serving-speed claim follows from this diagnostic alone.
