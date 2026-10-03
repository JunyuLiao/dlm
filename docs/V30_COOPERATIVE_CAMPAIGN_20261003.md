# V30 cooperative sparsity and reproducibility campaign

Branch: `research/cooperative-sparsity-20261003`, from V29 `035b70c3f`.
User authorization: 2026-10-03, US Central (UTC-5). Store attributed peer
documentation, execute value-only evaluation, investigate the dense/PIECEWISE
trajectory discrepancy, and explore separately named Junyu/Haowei variants.
Use multiple questions and eight seeds for performance/accuracy confirmation.
The root agent owns critical reasoning, protocol decisions and source review;
delegation is limited to checked mechanical work and preparation.

## Correction to earlier interpretation

The profiled singleton32K observations do not establish that sparse main is
slower at64K/96K. Recomputed closed accounting is in
[the length audit](../results/v30_20261003/closed_length_accounting001/README.md).
V28 preview main/native has S/N ratios0.902307 at64K and0.864398 at96K; W is
1.084097 and0.849983, respectively. N is1.197263 and0.955624. These are small
two-question-per-bin preview point estimates, not a new significance claim.
V18b controls have different warm/request inventories from main/dense; retain
that confound. Four repeats in V18b are two seeds times two sequential repeats.
Do not equate S/N with a fixed-input pure forward cost.

The inspected request receipts do not retain measured physical tile density.
High sparsity from a different HF/token audit cannot be substituted into these
vLLM requests. The execution path is proven separately from its actual density.

## What PIECEWISE does and what remains unresolved

PIECEWISE replays captured regions of a forward while dynamic attention calls
execute outside those regions. It does not divide/shorten the prompt. Current
sparse routing needs that dynamic execution; FULL graph replay could otherwise
freeze Python routing at capture. Keep strongest default dense and matched
PIECEWISE references, including no-hook and native-hook controls.

The root has read the actual installed sampler, runner, graph manager and
adapter. Native delegates original FA4 and sampler arguments/returns. The
sampler consumes global CUDA random draws, and the runner records request seed
as unapplied. This is not proof of the cause of the observed374-vs99 dense
trajectory discrepancy. Source passthrough, CPU tests and initialization graph
mode do not prove GPU equality. First-warm and timed graph/RNG/state evidence
must precede causal conclusions. The separately frozen12-engine diagnostic
checks reproducibility, not speed or large-sample quality.

## Peer references and execution boundaries

Junyu docs are snapshotted atb890ff494 under
[peer_references/junyu/b890ff494](peer_references/junyu/b890ff494/README.md),
with attribution, pinned links and private-path redaction. The source branch
remains unchanged. Interpretation belongs to the pinned review, not to the
quoted peer claims. Haowei's latest reviewed source is6f7279c16.

Original Junyu uniform/gated CPU formula tests were executed in a new own
isolated directory:31passed, CUDA not initialized. These do not qualify the
CUDA kernel, sampler lifecycle, accuracy or E2E timing. The identified ABI4
debug-score guard must be corrected in an explicitly named isolated patch
before affected GPU qualification; preserve the original peer source.

Value-only must be explicitly separated into the peer current-QK/retained-state
Gaussian32 router (sensitivity1, noT/C_gate) and any future cached-DP value-only
ablation of our system. They are not the same method. An HF/SDPA reproduction
is a substrate-specific result; only a matched strong FA4/official serving
comparison supports our headline speed claim. All-kept peer-kernel and native
controls are required to expose bridge/kernel cost. Do not use placeholder
zero thresholds or unverified historical binaries.

## Execution and reporting contract

Current V29 GPU workers/bindings remain immutable. A local status-query SSH
timeout stopped the old monitor, not those GPU experiments. Read-only checks
confirmed all three GPUs occupied and no completed-worker failures. Restored
monitor completion002 retries only identified read-only transport errors;
original formal sources, worker directories and score protocols remain fixed.
Component003, dense003 and engineering003 wait on that strict completion.
Old failed monitor/queue records are preserved and never rewritten as successes.

Every new variant needs unit tests, source/config freeze, independent numerical
qualification, path/density receipts and zero timed JIT/capture. Qualification
samples cannot establish speed or accuracy. Confirmation uses paired questions
and eight predeclared seeds, reports W including prefill, S, S/N, actual N,
commits, output length and correctness with question-cluster95% intervals.
Sampler/native stopping is fixed. Retain short AIME/HumanEval quality checks.
No new score-driven tuning on the confirmation set.

Regroup candidates must count support construction, forward movement, inverse
restoration and lifetime costs. Compare against the same optimized natural-order
consumer. Peer register-fed statistics are a cooperation candidate only when
our dense-prefix risk semantics are preserved. No new GPU benefit is claimed.
