# Replicated native evaluation under observed execution variation

The unchanged V31 control is not exactly reproducible in this compiled runtime.
Two control-history engines each ran six identical requests for three AIME
inputs. Engine 1 repeated all outputs exactly; engine 2 changed the final long
input repetition from 395 to 393 calls and changed output, despite unchanged
input and RNG seed. There was no value intervention. Earlier diagnostics also
show substantial variation across engines. Neither source nor sampler is being
changed to force a repeatable result.

The exact-parity process-matched campaign remains held, as does the older
independent-process campaign. The new study has a separate protocol. It retains
all structurally valid outcomes, including control drift, rather than selecting
only deterministic cases. This is an explicit change in the comparison design,
not a reinterpretation of a failed old gate. It cannot diagnose the numerical
root cause or guarantee absence of method-dependent execution effects.

## Generation design

Complete the predeclared warmup factor first. Its result is execution evidence,
not method efficacy. All new common-engine comparisons use an identical
all-method warmup to initialize the applicable computations. This policy is
selected before examining warmup-factor results; it does not assert that
warmup removes variability. For AIME use the long factor input; other workloads
use both already qualified smoke inputs. Require sparse anchor exposure and
held correction where applicable; retain warmup receipts separately.

Keep the existing development cohorts and four promoted candidates frozen:
AIME 30 problems x 3 seeds, LongBench-v2 64, MATH 100, RULER128K 104. Every
cell runs in two fresh engines. The existing equal-sized shards define pairs
of engines: 10 AIME, 8 LongBench, 10 MATH, 13 RULER. No method or input is
selected using the held accuracy results.

In each engine and for each cell, run V31, the four candidates, then V31.
The first engine uses the previously defined Latin rotation of candidates;
the second reverses that candidate order. All six requests use identical native
input and seed. Each candidate therefore has two trials per cell; control
has four. This is repeated evaluation, not answer ensembling or best-of-two
scoring. Use both control bookends, never the more favorable one.

Native proposal, acceptance, stopping, re-noising, self-conditioning and output
limits stay unchanged. Value changes remain at GLOBAL layer 29. Every head
and Q128 tile has exactly K eligible prefix KV64 blocks: K=27 on AIME/MATH,
K=128 on LongBench/RULER; common mandatory/carry additions stay separate.
LOCAL remains dense. Clean timing stays paused. Source, runtime, official
scorers and inputs are frozen; one worker uses the H100 at a time.

Fresh dense FULL fix51994, PIECEWISE and all-kept references run once per cell
under the same source and runtime. They provide substrate/accounting context;
unresolved cross-process variation limits fine-grained numerical comparisons
against them. Incremental method conclusions use the common-engine V31 pair.

## Gates and failure handling

Before accuracy, run a two-engine smoke for every workload and all named
reference paths. Validate source/input bindings, live request arms, exact K,
layer scope, native settings, consumer path, sparse activation, capture counts
and official scorer coverage. The previous same-state and H100 method gates
remain required because method code is unchanged.

Output drift is an audited outcome. Publish bookend differences in hash,
N/C/T and official correctness, plus inactive-candidate mismatches. They do
not automatically discard a case. Structural failures still stop the queue
and preserve the failed attempt. Never relaunch a structurally valid engine
because its outputs or score are unfavorable. Completed shards resume only
after hash verification. A relaunch after an interrupted worker gets a new
attempt directory.

## Analysis

Score each repeat and each control position independently with the official
scorer and full planned coverage. For cell i and engine replicate r, compute
the candidate-minus-mean-of-two-controls difference. Average over the two
engines, then over the fixed cells. AIME averages seeds within problem;
RULER reports the official task-average score. Count every repeated generation
equally within its method; never turn repeated correctness into pass@2.

Report each engine replicate separately, then the aggregate. Report accuracy,
N/problem-trial, N/C, C, T, p95 N, cap rate, GLOBAL and LOCAL physical work,
overall denominator, and extra discovery/correction work. Absolute tile and
call totals must be normalized for the unequal number of control/candidate
trials. Audit wall times are not clean latency or speedup.

Use 20,000 fixed-seed bootstrap draws for problem-cluster intervals (task
clusters on RULER), and separately resample the paired engine-shard blocks.
The latter jointly retains both engine replicates and all cells in each shard.
These are complementary sensitivity intervals, not a joint confidence region;
there are only 8-13 engine pairs per workload. Also report control-before vs
control-after score and trajectory differences, between-engine variation and
inactive-candidate drift. A small candidate gain within control variability
is not evidence of a reliable improvement.

The prior exploratory promotion screen is unchanged: no observed accuracy
loss and no larger mean N on any workload, at least one strict accuracy gain,
and no greater than 5% p95-N or cap-rate regression. State uncertainty and
replicate consistency even if this screen passes. Full503 requires a separate
frozen protocol; no positive research claim follows from a development screen.
All earlier failed and held campaigns retain their original interpretation.
