# v18 design decision — frozen before new evaluation

Authority: user-requested `dllm_astra_sol_master_v18_20260926.md` only.
Parent: d2bef2189c046e9a4277ab16d751ebeab11c15d8. Branch:
research/astra-junyu-frontier-20260926. Coordinator: Astra; bounded execution
agents actually dispatched as gpt-6-sol. No v16/v17 completion is assumed.

## Native contract and scope

Inspected mpk import of Transformers 5.11.0 generation source SHA256
b814a6fc41492794c1f50ba71c750206684e25dfdd53932e0842675871fef0de.
`_denoising_step` passes deterministic processed-logit argmax to
`StableAndConfidentStoppingCriteria`; it returns stability AND mean entropy
strictly below confidence_threshold. Acceptance separately sorts entropy and
uses cumulative entropy minus the current maximum <= entropy_bound. It is
reversible; renoising consumes RNG even when positions are accepted. Neither
the current noisy canvas nor sampled proposal is the stability history.
Keep B256/cap48, temperature 0.8->0.4, confidence .005, stability 1,
entropy_bound .1, EOS, self-conditioning, and complete-answer budgets.

Native decoder DynamicCache has already cropped LOCAL prefix. Its SDPA route
uses the supplied mask and noncausal full canvas, without another query-relative
window. Junyu integration currently applies that extra window. Formal scope
must be a separately named `native_legal_all_layers`: preserve attention kind
for threshold choice but remove the extra window from geometry/validity only.
Forbid causal decoder calls and unsupported additive masks; qualify explicit
bool masks, cache/window boundaries, and full current canvas. Legacy remains
available solely as LEGACY_REPRO. Dense numerical agreement does not mean
bitwise native equality; keep D_matched and D_native separate.

## Candidate cards (one selected investigation)

1. **Post-QK ball screen with unchanged parent fallback — selected for bounded
opportunity measurement, not yet authorized for kernel integration.** Build
center/radius from actual current projected V; immutable prefix metadata may
reuse the existing producer lease. QK remains fully computed. Real arithmetic
triangle bound can bypass PZ only when both Q64 halves certify all valid rows
of one Q128 decision. Prior failures: stale final weights and uncentered proxy
are not this mechanism. Added build/read/scan and cluster vote costs may erase
the saving. Falsify on 12–24 predetermined real states and direct call cost;
if ideal bound already fails, stop without kernel work. No stop modification.

2. **Risk algebra rearrangement — not selected.** Parent already has fixed
squared-risk specialization and the v14 T-only optimization. Replacing logs
or norm reduction could change near-threshold decisions and is not a new
independent contribution without measured headroom. Do not duplicate it.

3. **Mass-only radius bound — not selected.** Bounding with alpha times global
maximum Z norm removes a distance calculation but is weaker than the ball.
No evidence of enough separation; do not implement a second candidate.

## Numerical and synchronization gate

The implemented P row is FP32 exp, row_sum, reciprocal and multiplication,
not an exactly normalized real probability. If W=sum(P), a sound real bound
for that row is ||W*c-o|| + W*r (or ||c-o|| + W*r + |W-1|*||c||), not
automatically ||c-o||+r. Parent PZ uses high/residual TF32x3 terms, omits
low*low, and accumulates in a particular MMA order. Any deployed certificate
must additionally bound that product/accumulation error, metadata rounding,
alpha multiplication, squared-norm reduction and the parent's comparison.
Empirical epsilon or an ideal triangle proof cannot certify deployed equality.
Exceptional values, ties, first-support and unsupported precision fall back.

Parent projects/registers before `decide`, shares Z and K/P scratch, runs two
Q64 CTAs per physical decision, and overlaps PV producer/consumer phases.
Both halves and all participants must execute identical collective/barrier
sequences. A quick certificate vote would be an additional cluster collective;
the original decide and consumer phase must still be executed even on bypass.
No speculative per-half early return. No CUDA edits until opportunity and a
finite-precision certificate justify the cost. Strict qualification includes
support/eligibility/output, execution counters, sanitizers and full native
parent/child token/calls/stop/RNG checks before 144 AIME runs.

## Execution order and resource reservation

CP0: existing evidence audit + strong B8 export removal + stop/scope tests.
CP1: scope qualification, independent 26-row balanced calibration (<=5 pairs
per new method-point), frozen 130-row RULER protocol, bridge if qualified.
Then RULER 50/60 primary, AIME 30 x seeds101/202/303 x six arms x first/warm,
then 70/allocation secondaries. T60 success and quality equivalence are unknown.
Use RULER threshold transfer to AIME; no AIME26 tuning. Candidate fallback
parent is T50 if pre-evaluation calibration cannot qualify T60, never by eval.

Preserve >=1080 AIME executions plus scoring time in the 7000 total cap.
Wall <=12h; GPU process <=20h only with two qualified hosts, otherwise <=10h;
reserve final 30min for scoring. Model load/compile/diagnostics count. Complete
question-seed-arm-repeat blocks stay on one GPU; timing invalidation never
replaces the first answer. Shard hosts before outputs. No cross-host absolute
latency pooling, no peer branch changes, no CVM_latest expansion.

## Initial machine finding

mpk .254: H10080 GPU-6139046a-b005-8fe5-a837-f8270472ab72, no model job
at inspection; root ~51GB free, private volume ~967GB free. Current research
tree clean at parent SHA. dllm .64: H10080
GPU-fc12ad5c-5334-5509-8fc6-465498fd3915, no model job; root ~1.4TB free.
Existing .64 runtime (Torch2.13/cu130, Triton3.7.1, Transformers5.17) differs
from .254 (Torch2.6/cu124, Triton3.2, Transformers5.11) including actual source
hashes. .64 has an older text-export model, not yet verified against the pinned
snapshot. It is NOT bridge-qualified. Do not allocate official shards to it
until private environment/model qualification and two-prompt bridge pass.

## CP1 prospective opportunity-state refinement (before any ball result)

All four U/T50/60 points reached overall/LOCAL/GLOBAL density tolerance using
at most five density-only policy pairs. This is calibration success, not quality,
step-count, or latency success. Keep the predeclared T60 opportunity parent.
The first12 T50 states remain the scope reference. Before seeing any ball
opportunity output, predeclare twelve additional states at the same two prompts,
seed101, layers0/5/29 and steps1/3 under the final calibrated T60 policy. Total24
real states stays within v18. Analyze these T60 states for the selected parent;
no evaluation output or answer informs this choice. The waiting T50 ball job was
stopped before GPU execution. No CUDA implementation is authorized by this step.
