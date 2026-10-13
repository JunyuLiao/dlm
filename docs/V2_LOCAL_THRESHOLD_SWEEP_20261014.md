# Accuracy-first V2 local threshold sweep

Authorized scope: official 503-problem LongBench-v2, all 30 AIME26 problems
across seeds 42/43/44, and the fixed sampled 100-problem MATH500 subset.
This is exploratory full-panel tuning; no held-out confirmation is claimed.
Clean timing stays paused. Existing campaigns and paused queues are untouched.

The unchanged selector is `value_v2_online_preserve_mass`, with one uniform
GLOBAL threshold, rank32/seed1729 projected V used only for routing, native FA4
output, no mass anchor, and no fixed quota. The inherited Yuhan V31 control
remains `qblock_max`, with select step 1, first-call carry, settledness trigger
0.15, sticky bonus 1.386. LOCAL, prefill, commit, and native sampler are fixed.
Exact greedy, stable refresh, cached refresh, sensitivity weighting, force-ref,
and canvas reseeding are disabled in this study.

Stage 1 tests 0.018, 0.0195, 0.021, 0.0225, 0.024. Rank shared thresholds by
mean workload accuracy (each of the three workloads has equal weight; AIME is
averaged over its three seeds), then normalized calls/problem, then absolute
kept-GLOBAL work ratios, with the latter two breaking exact accuracy ties.
After every stage-1 arm is complete and scored, test the two adjacent midpoints
(best +/- 0.00075), yielding seven tested shared thresholds. Report each
workload's best threshold separately as well as the shared winner. Claim only
the best tested point; retain the non-monotonic curve and all negative arms.

All workloads have fresh, matched V31, FULL fix51994 dense, PIECEWISE dense,
and all-kept FA4 controls. LongBench uses its existing manifest and seed 1,
block32/chunk16384/MEM0.85/maxlen141312, V31 K8192. AIME uses its existing
8192-token thinking manifest, block64/chunk4096/MEM0.80/maxlen9216, V31 K1728.
MATH500 preserves the existing deterministic sampling seed 20261011, generation
seed 1, 4096-token thinking manifest, block64/chunk4096/MEM0.80/maxlen16384,
V31 K1728. The short-workload K choice applies only to V31, never as a V2 quota.

Freeze production code into a private execution snapshot before qualification.
Record all execution/input hashes, model inventory, runtime source identities,
GPU/host, schedule seeds and protocol hash. Qualify all five method paths on two
cells of each workload, then run rotated immutable resumable shards. A partial
attempt is retained and rerun in a new attempt directory; complete attempts are
validated and skipped. One worker per H100, resource-idle wait, and active
progress/error checks about every 30 minutes.

Use the existing official LongBench and AIME scorers, enforcing prompt/manifest/
seed/budget/runtime bindings. MATH500 uses bound math-verify symbolic equivalence
with the same final-response parser; unclosed thinking is wrong. The existing
quick MATH500 rowinfo has malformed compact-token hashes and missing fields, so
rebuild scoring metadata only, preserving generation manifest bytes exactly.
Do not use its old string-normalization quick score as the primary metric.

Report accuracy, per-seed AIME scores, LongBench strata, paired problem-clustered
bootstrap intervals (30 clusters for AIME), exploratory cell McNemar, output
hash/length/call mismatches, parser/cap coverage, N/C/T distributions, phase
work, physical eligible/kept/skipped GLOBAL/overall tiles, selector discovery/
evaluation/timing, and instrumented audit wall time separately. Rank observed
adapter canvas IDs into actual ordinals for early-canvas accounting; never fill
invented zero-valued early canvases or infer physical work from selector counts.

Campaign: `results/v31_v2_local_threshold_sweep_20261014_attempt002/`.
Entrypoint: `scripts/v31_v2_local_threshold_sweep.py`.

Preparation attempt001 was retained before any GPU work: legacy LongBench
schedules store token counts, while quick schedules store token lists. The
validator now accepts both only after checking against the same manifest.
