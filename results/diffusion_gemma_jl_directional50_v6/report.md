# 50%-only directional routing: AIME26 + LongBench v2

Status: COMPLETE; 640/640 audited final outputs. Three new routers at50% only; dense + original/aggressive BLASST + mass-only cached baselines.

## Setup and selection

All30 AIME26 problems,2048 output tokens;50 LongBench v2 questions selected deterministically from the completed newer100 NeMo study,4096 output tokens and32K context cap. Rows are copied verbatim, preserving tokenized prompts, seed42 and official scoring. NeMo is the pinned actual LongBench prompt/MCQ scorer, not a bespoke imitation; AIME retains the existing scorer and prompts. The old50-ID cohort is superseded.

| domain | count |
| --- | --- |
| Code Repository Understanding | 5 |
| Long In-context Learning | 8 |
| Long Structured Data Understanding | 3 |
| Long-dialogue History Understanding | 4 |
| Multi-Document QA | 13 |
| Single-Document QA | 17 |

Selection: Hamilton domain then subtask quotas, SHA256(seed20260915,ID) ordering; no scores or predictions used. All final samples have previous exposure. AIME full30 overlaps the6 calibration problems; noncalibration24 is separate, not fresh held-out confirmation. Follow-on LongBench calibration profile: final50; current overlap is 6/50. Reference/baseline calibration remains unchanged. Headline scores include all30 AIME and all50 LongBench problems, including calibration members. See selection.json, dataset_audit.json and the verbatim manifests.

Pinned BF16 DiffusionGemma revision f7f5b7f5fa82ffc52addd066915886d497f5517b;128-query×64-KV physical tiles; prefix+canvas eligible, native masks and GQA. Canvas256/up to48 denoising steps, thinkingFalse, unchanged native0.4–0.8 schedule (temperature0 sentinel is not greedy). The same output budgets/settings/seeds are used for every method. Dense outputs are imported once, never rerun for final comparison.

## Algorithms and calibration

Three prespecified methods, in order: centered Gaussian32, centered random-sign32, and the existing Gaussian32 cancellation guard. Primary seed1729; independent guard seed2718. The guard uses max(primary,second) centered risk only where projected kappa<0.25 and online alpha>=0.1. It stores two32-dimensional sketches (64 coordinates total). Full-dimensional reference results are reused. All retain strict worst-query128x64 tile gating, ties/first support, native masks/GQA, prefix+canvas eligibility, FP32 routing and ordinary original-V renormalization. No algorithm is changed or chosen by final accuracy.

The unchanged BLASST implementation uses exact token QK and per-row KV-block maxima; a physical tile skips only when all valid rows vote skip. Original BLASST caps lambda at1; aggressive BLASST allows lambda>1 using the preceding-seen-maximum rule. Source inverse-valid-KV-length thresholds are lambda=exp(log_scale)/L with the recorded cap; scalar thresholds are exp(log_threshold). Verified unattainable original types use constant lambda1. Local/global policies remain distinct and shared across layers/heads/steps.

All400 dense/BLASST/mass/reference50 outputs are reused from the independently audited reference bundle. The three follow-on methods are calibrated on six fixed questions/benchmark at full2048/4096-token budgets. Existing empirical-rank proposals and scalar fallback allow at most12 joint points. Deployment requires measured physical overall/global/local calibration sparsity between48% and52%, with no inherited75% exception. Historical thresholds are warm starts only. Gaussian/sign32 reuse historical calibration-state CDFs; the guard32 CDF is recomputed with the exact requested operator on those saved states, while a historical rank16 guard scalar is only the first verified trial. No final accuracy is used for threshold selection.

Historical540 early512-token dense calibration states remain the common diagnostic/proposal population; they are not substitutes for full-budget verification and do not exhaustively cover later generation. Calibration membership differs from historical baselines/reference where documented; headline denominators never drop calibration questions. Achieved final sparsity can still transfer imperfectly and is reported without retuning from final scores. Cached mass-only is the existing max-based candidate mass bound, not mass_exact, so its comparison does not isolate direction from mass estimation alone.

## Main results

aime26: dense 17/30. 

cancel_guard_gaussian_r32_s50: 13/30, actual sparsity 50.7% (global 51.3%, local 50.6%); dense accuracy delta -13.3 pp, paired 95% CI [-26.7, -3.3] pp.

full_centered_s50: 15/30, actual sparsity 51.3% (global 51.4%, local 51.3%); dense accuracy delta -6.7 pp, paired 95% CI [-23.3, +13.3] pp.

jl_gaussian_r32_s50: 15/30, actual sparsity 50.7% (global 50.7%, local 50.7%); dense accuracy delta -6.7 pp, paired 95% CI [-26.7, +10.0] pp.

jl_sign_r32_s50: 14/30, actual sparsity 49.7% (global 48.2%, local 50.0%); dense accuracy delta -10.0 pp, paired 95% CI [-26.7, +6.7] pp.

6 nearest-point comparisons are not within3 percentage points in all overall/global/local sparsity measures; do not infer a matched-budget winner from those comparisons.

longbench_v2: dense 18/50. 

cancel_guard_gaussian_r32_s50: 14/50, actual sparsity 47.7% (global 47.1%, local 51.0%); dense accuracy delta -8.0 pp, paired 95% CI [-20.0, +2.0] pp.

full_centered_s50: 21/50, actual sparsity 49.6% (global 49.6%, local 49.8%); dense accuracy delta +6.0 pp, paired 95% CI [-6.0, +18.0] pp.

jl_gaussian_r32_s50: 18/50, actual sparsity 47.0% (global 46.3%, local 50.3%); dense accuracy delta +0.0 pp, paired 95% CI [-14.0, +14.0] pp.

jl_sign_r32_s50: 17/50, actual sparsity 47.2% (global 46.7%, local 50.0%); dense accuracy delta -2.0 pp, paired 95% CI [-10.0, +6.0] pp.

5 nearest-point comparisons are not within3 percentage points in all overall/global/local sparsity measures; do not infer a matched-budget winner from those comparisons.

### aime26: full

| condition | target | threshold | score | accuracy | delta_pp | whole | global_s | local_s | mass | agreement | trajectory_local_error | unparsed | length_limited |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blasst_aggressive_s50 | 50.0000 | local: exp(8.09133)/L; global: exp(7.77928)/L | 11/30 | 36.6667 | -20.0000 | 51.3614 | 49.6095 | 51.7731 | 69.9423 | 2.4106 | 0.5001 | 0 | 6 |
| blasst_original_s50 | 50.0000 | local: 1 (unattainable); global: 1 (unattainable) | 16/30 | 53.3333 | -3.3333 | 28.8187 | 35.4672 | 27.2322 | 93.4179 | 5.7355 | 0.1468 | 0 | 2 |
| cancel_guard_gaussian_r32_s50 | 50.0000 | local: exp(-0.774403); global: exp(-1.22591) | 13/30 | 43.3333 | -13.3333 | 50.7086 | 51.2861 | 50.5663 | 72.9178 | 4.4709 | 0.2545 | 0 | 6 |
| dense | 0.0000 | none / ranking budget | 17/30 | 56.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0 | 6 |
| full_centered_s50 | 50.0000 | local: exp(-0.740297); global: exp(-1.21186) | 15/30 | 50.0000 | -6.6667 | 51.3059 | 51.4252 | 51.2764 | 72.6830 | 4.8750 | 0.2574 | 0 | 7 |
| jl_gaussian_r32_s50 | 50.0000 | local: exp(-0.759528); global: exp(-1.23614) | 15/30 | 50.0000 | -6.6667 | 50.6621 | 50.6697 | 50.6602 | 73.3096 | 5.1972 | 0.2527 | 0 | 5 |
| jl_sign_r32_s50 | 50.0000 | local: exp(-0.776932); global: exp(-1.28216) | 14/30 | 46.6667 | -10.0000 | 49.6721 | 48.2015 | 50.0299 | 74.3005 | 4.7805 | 0.2476 | 0 | 6 |
| mass_s50 | 50.0000 | local: exp(-0.0184422); global: exp(-0.04564) | 14/30 | 46.6667 | -10.0000 | 51.3914 | 52.9351 | 51.0090 | 75.0530 | 4.2693 | 0.3014 | 0 | 8 |

### aime26: noncalibration24

| condition | target | threshold | score | accuracy | delta_pp | whole | global_s | local_s | mass | agreement | trajectory_local_error | unparsed | length_limited |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blasst_aggressive_s50 | 50.0000 | local: exp(8.09133)/L; global: exp(7.77928)/L | 9/24 | 37.5000 | -20.8333 | 51.3468 | 49.6216 | 51.7528 | 70.3458 | 2.5868 | 0.4932 | 0 | 5 |
| blasst_original_s50 | 50.0000 | local: 1 (unattainable); global: 1 (unattainable) | 13/24 | 54.1667 | -4.1667 | 28.9761 | 35.2079 | 27.4916 | 93.4145 | 6.0140 | 0.1471 | 0 | 2 |
| cancel_guard_gaussian_r32_s50 | 50.0000 | local: exp(-0.774403); global: exp(-1.22591) | 11/24 | 45.8333 | -12.5000 | 50.7631 | 51.1681 | 50.6636 | 72.9458 | 4.4120 | 0.2528 | 0 | 5 |
| dense | 0.0000 | none / ranking budget | 14/24 | 58.3333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 0 | 5 |
| full_centered_s50 | 50.0000 | local: exp(-0.740297); global: exp(-1.21186) | 12/24 | 50.0000 | -8.3333 | 51.4690 | 51.8490 | 51.3750 | 72.5994 | 5.0550 | 0.2566 | 0 | 5 |
| jl_gaussian_r32_s50 | 50.0000 | local: exp(-0.759528); global: exp(-1.23614) | 12/24 | 50.0000 | -8.3333 | 50.7085 | 51.0653 | 50.6203 | 73.4097 | 4.7615 | 0.2503 | 0 | 4 |
| jl_sign_r32_s50 | 50.0000 | local: exp(-0.776932); global: exp(-1.28216) | 12/24 | 50.0000 | -8.3333 | 49.7190 | 48.0401 | 50.1258 | 74.2753 | 4.9511 | 0.2458 | 0 | 4 |
| mass_s50 | 50.0000 | local: exp(-0.0184422); global: exp(-0.04564) | 11/24 | 45.8333 | -12.5000 | 51.6113 | 53.1945 | 51.2174 | 74.9433 | 3.8060 | 0.3024 | 0 | 6 |

### longbench_v2: full

| condition | target | threshold | score | accuracy | delta_pp | whole | global_s | local_s | mass | agreement | trajectory_local_error | unparsed | length_limited |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blasst_aggressive_s50 | 50.0000 | local: exp(8.37258)/L; global: exp(8.36106)/L | 23/50 | 46.0000 | 10.0000 | 46.9802 | 46.7117 | 48.3711 | 75.8637 | 5.2580 | 0.3402 | 3 | 0 |
| blasst_original_s50 | 50.0000 | local: exp(0); global: min(1, exp(8.36106)/L) | 22/50 | 44.0000 | 8.0000 | 44.0676 | 48.1195 | 23.1271 | 93.8722 | 6.9345 | 0.1303 | 0 | 0 |
| cancel_guard_gaussian_r32_s50 | 50.0000 | local: exp(-0.753086); global: exp(-5.162) | 14/50 | 28.0000 | -8.0000 | 47.6958 | 47.0643 | 51.0299 | 72.0714 | 4.3534 | 0.2680 | 5 | 1 |
| dense | 0.0000 | none / ranking budget | 18/50 | 36.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0000 | 100.0000 | 0.0000 | 3 | 0 |
| full_centered_s50 | 50.0000 | local: exp(-0.731019); global: exp(-5.01796) | 21/50 | 42.0000 | 6.0000 | 49.6155 | 49.5805 | 49.7930 | 72.3568 | 5.1859 | 0.2777 | 1 | 0 |
| jl_gaussian_r32_s50 | 50.0000 | local: exp(-0.72214); global: exp(-5.16173) | 18/50 | 36.0000 | 0.0000 | 46.9580 | 46.3041 | 50.3413 | 71.9082 | 5.1171 | 0.2827 | 1 | 0 |
| jl_sign_r32_s50 | 50.0000 | local: exp(-0.724918); global: exp(-5.15948) | 17/50 | 34.0000 | -2.0000 | 47.2288 | 46.6876 | 50.0216 | 72.6259 | 5.3078 | 0.2748 | 3 | 0 |
| mass_s50 | 50.0000 | local: exp(-0.0152636); global: exp(-1.3682) | 20/50 | 40.0000 | 4.0000 | 48.7401 | 48.3491 | 50.7455 | 74.4101 | 4.3753 | 0.3151 | 1 | 0 |

### LongBench v2 domain results

| domain | condition | score | accuracy |
| --- | --- | --- | --- |
| Code Repository Understanding | blasst_aggressive_s50 | 3/5 | 60.0000 |
| Code Repository Understanding | blasst_original_s50 | 2/5 | 40.0000 |
| Code Repository Understanding | cancel_guard_gaussian_r32_s50 | 1/5 | 20.0000 |
| Code Repository Understanding | dense | 3/5 | 60.0000 |
| Code Repository Understanding | full_centered_s50 | 1/5 | 20.0000 |
| Code Repository Understanding | jl_gaussian_r32_s50 | 2/5 | 40.0000 |
| Code Repository Understanding | jl_sign_r32_s50 | 1/5 | 20.0000 |
| Code Repository Understanding | mass_s50 | 2/5 | 40.0000 |
| Long In-context Learning | blasst_aggressive_s50 | 4/8 | 50.0000 |
| Long In-context Learning | blasst_original_s50 | 3/8 | 37.5000 |
| Long In-context Learning | cancel_guard_gaussian_r32_s50 | 3/8 | 37.5000 |
| Long In-context Learning | dense | 2/8 | 25.0000 |
| Long In-context Learning | full_centered_s50 | 5/8 | 62.5000 |
| Long In-context Learning | jl_gaussian_r32_s50 | 1/8 | 12.5000 |
| Long In-context Learning | jl_sign_r32_s50 | 4/8 | 50.0000 |
| Long In-context Learning | mass_s50 | 3/8 | 37.5000 |
| Long Structured Data Understanding | blasst_aggressive_s50 | 0/3 | 0.0000 |
| Long Structured Data Understanding | blasst_original_s50 | 0/3 | 0.0000 |
| Long Structured Data Understanding | cancel_guard_gaussian_r32_s50 | 0/3 | 0.0000 |
| Long Structured Data Understanding | dense | 0/3 | 0.0000 |
| Long Structured Data Understanding | full_centered_s50 | 0/3 | 0.0000 |
| Long Structured Data Understanding | jl_gaussian_r32_s50 | 0/3 | 0.0000 |
| Long Structured Data Understanding | jl_sign_r32_s50 | 0/3 | 0.0000 |
| Long Structured Data Understanding | mass_s50 | 0/3 | 0.0000 |
| Long-dialogue History Understanding | blasst_aggressive_s50 | 0/4 | 0.0000 |
| Long-dialogue History Understanding | blasst_original_s50 | 0/4 | 0.0000 |
| Long-dialogue History Understanding | cancel_guard_gaussian_r32_s50 | 0/4 | 0.0000 |
| Long-dialogue History Understanding | dense | 0/4 | 0.0000 |
| Long-dialogue History Understanding | full_centered_s50 | 0/4 | 0.0000 |
| Long-dialogue History Understanding | jl_gaussian_r32_s50 | 0/4 | 0.0000 |
| Long-dialogue History Understanding | jl_sign_r32_s50 | 0/4 | 0.0000 |
| Long-dialogue History Understanding | mass_s50 | 0/4 | 0.0000 |
| Multi-Document QA | blasst_aggressive_s50 | 6/13 | 46.1538 |
| Multi-Document QA | blasst_original_s50 | 7/13 | 53.8462 |
| Multi-Document QA | cancel_guard_gaussian_r32_s50 | 3/13 | 23.0769 |
| Multi-Document QA | dense | 5/13 | 38.4615 |
| Multi-Document QA | full_centered_s50 | 6/13 | 46.1538 |
| Multi-Document QA | jl_gaussian_r32_s50 | 4/13 | 30.7692 |
| Multi-Document QA | jl_sign_r32_s50 | 4/13 | 30.7692 |
| Multi-Document QA | mass_s50 | 6/13 | 46.1538 |
| Single-Document QA | blasst_aggressive_s50 | 10/17 | 58.8235 |
| Single-Document QA | blasst_original_s50 | 10/17 | 58.8235 |
| Single-Document QA | cancel_guard_gaussian_r32_s50 | 7/17 | 41.1765 |
| Single-Document QA | dense | 8/17 | 47.0588 |
| Single-Document QA | full_centered_s50 | 9/17 | 52.9412 |
| Single-Document QA | jl_gaussian_r32_s50 | 11/17 | 64.7059 |
| Single-Document QA | jl_sign_r32_s50 | 8/17 | 47.0588 |
| Single-Document QA | mass_s50 | 9/17 | 52.9412 |

## Measurements, uncertainty, and limitations

Physical sparsity is SUM(skipped eligible tiles)/SUM(eligible tiles), not an average of sample/layer percentages. Decoder denoising attention only; dense prefill is excluded. Region counts separate prefix, canvas and boundary tiles. Main retained mass and local error compare dense/sparse attention on each run’s corresponding QKV; they are not a replay of cached dense-generation states after token divergence. Shared-QKV diagnostics separately compare every method on identical saved dense calibration states, with error=sqrt(sum error²/sum dense-output²). Sampling is one query head/query tile per selected layer/step state, not exhaustive; shared_operator.csv records coverage, full-dimensional risk underestimation and overlapping type statistics by layer/head/step/attention type/region. Full PV is diagnostic-only for projected routing.

Token agreement compares every generated position, including after the first divergence; missing/extra positions disagree and EOS is included. Accuracy is micro, with domain macro also exported. Paired prompt-bootstrap95% intervals are exploratory, not multiplicity-corrected. Comparisons select the nearest observed actual-sparsity baseline and flag unmatched global/local budgets; no target-based interpolation or unsupported winner. Correlations across dependent configurations are descriptive, not causal. Gaussian versus random-sign is tested at rank32 only. No rank8/16 comparison or end-to-end seed-sensitivity conclusion is claimed.

projection_seed_sensitivity.json/csv reuse the prior Gaussian/sign32 calibration-only shared-state checks for the fixed seeds1729/2718/31415 at the historical512-token thresholds, not the new final thresholds. They compare worst-row projected/full-dimensional risks with all previous blocks retained; this is a support-controlled distortion check, not final-generation sparsity or accuracy evidence. No seed is selected from these results.

QK, block softmax and projected PV remain computed. Sketch caching refreshes changed canvas/boundary values; work_accounting.json records projected multiply-adds, reuse/refresh and storage. The retained native BF16 PV is still a dense-shaped masked matmul in this reference evaluation. Physical deletion is not proof of hardware-compute avoidance; runtime engineering is not a FlashAttention/system-speedup claim. Full-dimensional error may correlate differently with downstream accuracy; report both.

## Failures and reproducibility

Missing entries: 0; violations: 0. See audit.json and preserved failures.jsonl. Superseded jobs were canceled by explicit user approval, not excluded for scientific failure; their artifacts remain unchanged. No method is eliminated based on a tiny accuracy screen.

Regenerate without inference: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_directional50 report`; independently repeat with `verify`. Final tables use completed raw final shards only; incomplete conditions never appear as complete scores.

## Inference accounting

The240 slots are three methods times80 final questions, in addition to400 cached reference/baseline slots. Calibration trials and20 initial short validation calls are separately accounted in inference_accounting.json. Exact selected calibration outputs also serve as final outputs without repeated inference. Failed attempts remain in failures.jsonl. All75% and uncentered-contribution follow-ons are superseded, not scientific exclusions.
