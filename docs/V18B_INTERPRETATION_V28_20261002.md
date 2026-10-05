# V18b read-only interpretation for V28

Reviewed 2026-10-02 21:02 (US Central, UTC-5). Source is parent branch
`research/humaneval-v27-20261001`, commit
`e1d2f89c29b6a713ffd3419cf7149d1a34abbd50`,
[published summary](https://github.com/coconight01/dlm_test/blob/e1d2f89c29b6a713ffd3419cf7149d1a34abbd50/results/m1_m2_m3_frontier_v27_20260929/vllm_panel_v18b/summary.md)
and its adjacent receipts. No parent code or result directory was merged into V28.

## What the completed panel establishes

All 568 timed requests and eight workers closed on one host and one frozen
generation deployment, `8704cd072`. Main/dense cover 59 questions (24/24/11
at 32K/64K/96K), four sequential-repeat labels per question, with two distinct
engine-seed blocks (7001/7002). The repeat label is not a per-request sampling
seed; equal labels do not establish an identical random trajectory across arms.
Native-hook PIECEWISE and adapter all-kept controls cover four questions per bin.
Formal-panel reserved GPU time is 10654.113 s, or 12809.068 s including qualification.

All ratios below are candidate/reference paired geometric means. W includes
prefill and adapter request boundaries but excludes HTTP; S is decode span.
N counts actual denoising forwards, including one unused speculative forward
per request, and excludes encoder commit forwards. S/N is amortized cost.
Intervals are question-cluster 95% bootstrap intervals retaining repetitions.

| Length | Main/dense W [95% CI] | S | S/N [95% CI] | N | Strict correct main/dense |
|---|---|---|---|---|---|
| 32K | .7814 [.6902,.8833] | .7600 | 1.0302 [1.0198,1.0413] | .7377 | 63/48 out of 96 each |
| 64K | .7439 [.6719,.8233] | .6840 | .9351 [.9287,.9412] | .7315 | 47/47 out of 96 each |
| 96K | .8070 [.6453,.9857] | .7605 | .8821 [.8653,.8955] | .8622 | 16/16 out of 44 each |

The completed panel supports lower request time against default native dense
on these measured questions. It does not make all of that gain a sparse-kernel
contribution. At 32K, S/N is slower, while N falls substantially. At 64K/96K,
S/N improves, but changed trajectories and forward counts also affect W.
Main/dense actual N totals are 50844/72322. Strict correctness totals are
126/111; task correctness is 126/112 because one dense answer was correct but capped.

Strict accuracy differences are +15.63 percentage points [6.25,27.08] at 32K,
0 [-8.33,8.33] at 64K, and 0 [-11.36,9.09] at 96K. The latter two intervals
allow meaningful losses: equal observed counts are not proof of noninferiority.
The source does not publish an overall question-cluster CI; bin CIs cannot be
combined to invent one. The older HF counts remain valid observations, while
their speed ratios are conditional on the weaker HF dense substrate.

## Increment above matched references is not established generally

These controls contain only four independent questions per bin (16 requests).

| Length | Main/native W [95% CI] | Main/all-kept W [95% CI] | Main/native S/N | Main/native N |
|---|---|---|---|---|
| 32K | 1.1377 [1.0040,1.2883] | .9716 [.9300,1.0050] | .9862 | 1.1778 |
| 64K | 1.0065 [.8750,1.1648] | .9255 [.8969,.9668] | .9137 | 1.0982 |
| 96K | 1.0446 [.8347,1.3149] | 1.0315 [.8349,1.2745] | .8504 | 1.2810 |

Main is not consistently faster than the matched native control. The clear
64K time reduction versus all-kept is a useful lead, but main answers 7/16
correct there versus 8/16 for each matched control. Across this control subset,
strict counts are main/native/all-kept = 26/28/28 out of 48 each.

Native itself has lower W point estimates than default dense in the subset, largely alongside
lower N: its S/N ratios versus dense are 1.0504/1.0209/1.0180, all above one.
Thus its lower W must not be casually attributed to faster PIECEWISE engineering.
Small control coverage, RNG consumption and numerical changes remain plausible
contributors; the panel does not identify their individual causal shares.
Keep default dense, native-hook and all-kept controls in subsequent comparisons.

Tracked timed CUDA capture, backend and Inductor compilation counters are zero.
V18 predates the dedicated Triton/CuTe monitor; warning logs alone do not prove
universal JIT absence. Allocator costs were retained. Adapter D2H routing and
commit costs remain inside the appropriate measured spans. This is local
single-request latency evidence, not an HTTP or concurrent-throughput result.

## Consequences for the active V28 preview

The frozen six-arm preview remains unchanged: four distinct engine seeds,
two sequential repeats, six distinct questions (two per length bin), 48 timed
requests per variant. More engine seeds address one source of step variability;
they do not turn six questions into 48 independent questions. Report by-seed N,
W, S, S/N and correctness, alongside question-cluster CIs. Do not select the
best seed or treat repeated responses as independent accuracy observations.

Use this preview to screen Q64 and the matched canvas-memory-release option.
Kernel/component positives alone cannot qualify them. Regroup held gather plus
consumer plus scatter is currently negative; the tested Triton permutation
backend is also negative. Preserve those results without expanding them into
claims against every regroup implementation. Haowei overlap remains a cooperation
candidate, not incorporated peer code.

After scoring, any confirmation study must freeze a larger, distinct question
set and more independent seed blocks, expand matched controls beyond the small
V18 subset, and state the noninferiority margin and analysis before generation.
If implementing per-request RNG reset, qualify that as a separately named
protocol change first; it is not implied by an engine seed or repeat label.
Do not retrofit extra comparisons or change the active generation schedule.

New models remain at official dense smoke/stop qualification and CPU event
prototypes. No sparse-port accuracy or speed evidence exists for LLaDA2.1-mini
or I-DLM-8B yet.
