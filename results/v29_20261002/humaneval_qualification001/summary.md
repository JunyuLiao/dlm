# HumanEval pipeline qualification: one question, four arms

dataset/index/repeat; repeat is not a sampling seed; seed_applied=false

95% item-cluster percentile bootstrap retaining all repeats

W=request boundary wall; S=decode span excluding initial prefill; P=initial prefill; N=actual denoising forwards including speculative unused execution and excluding encoder commits; SN=S/N amortized cost

Observed differences and question-cluster CIs only; noninferiority margin unset. No noninferiority decision. McNemar is exploratory; repeats share questions.

| dataset | arm | cells | items | strict correct | task correct | cap | unparsed |
|---|---|---:|---:|---:|---:|---:|---:|
| humaneval | dense | 1 | 1 | 1 | 1 | 0 | 0 |
| humaneval | method | 1 | 1 | 1 | 1 | 0 | 0 |
| humaneval | native | 1 | 1 | 1 | 1 | 0 | 0 |
| humaneval | allkept | 1 | 1 | 1 | 1 | 0 | 0 |

Per-arm means/medians use all timed requests; decode_s_per_denoise_forward_geomean is the geometric mean of request S/N; commit mean uses only its explicitly reported observed cells

| dataset | arm | W mean / median (s) | S mean / median (s) | prefill mean (s) | denoise N mean / median / sum | unused speculative denoise mean | scheduler denoise sum | S/N geometric mean (s) | commit forwards mean (observed cells) |
|---|---|---|---|---:|---|---:|---:|---:|---|
| humaneval | dense | 2.8688 / 2.8688 | 2.8411 / 2.8411 | 0.0278 | 146.0000 / 146.0000 / 146 | 1.0000 | 145 | 0.019459 | 14.0000 (1) |
| humaneval | method | 3.0252 / 3.0252 | 2.9860 / 2.9860 | 0.0392 | 135.0000 / 135.0000 / 135 | 1.0000 | 134 | 0.022119 | 12.0000 (1) |
| humaneval | native | 7.0364 / 7.0364 | 7.0025 / 7.0025 | 0.0339 | 334.0000 / 334.0000 / 334 | 1.0000 | 333 | 0.020966 | 26.0000 (1) |
| humaneval | allkept | 2.5518 / 2.5518 | 2.5158 / 2.5158 | 0.0360 | 118.0000 / 118.0000 / 118 | 1.0000 | 117 | 0.021321 | 11.0000 (1) |

Ratios are candidate/reference; values below one mean faster or fewer forwards.

| dataset | comparison | cells | items | correct (base) | W [95% CI] | S [95% CI] | prefill P [95% CI] | S/N [95% CI] | N [95% CI] | accuracy difference [95% CI] | exploratory McNemar p |
|---|---|---:|---:|---|---|---|---|---|---|---|---:|
| humaneval | method/dense | 1 | 1 | 1 (1) | 1.0545 CI unavailable (one item) | 1.0510 CI unavailable (one item) | 1.4120 CI unavailable (one item) | 1.1367 CI unavailable (one item) | 0.9247 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| humaneval | native/dense | 1 | 1 | 1 (1) | 2.4527 CI unavailable (one item) | 2.4648 CI unavailable (one item) | 1.2222 CI unavailable (one item) | 1.0774 CI unavailable (one item) | 2.2877 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| humaneval | allkept/dense | 1 | 1 | 1 (1) | 0.8895 CI unavailable (one item) | 0.8855 CI unavailable (one item) | 1.2970 CI unavailable (one item) | 1.0956 CI unavailable (one item) | 0.8082 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| humaneval | method/native | 1 | 1 | 1 (1) | 0.4299 CI unavailable (one item) | 0.4264 CI unavailable (one item) | 1.1553 CI unavailable (one item) | 1.0550 CI unavailable (one item) | 0.4042 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| humaneval | method/allkept | 1 | 1 | 1 (1) | 1.1855 CI unavailable (one item) | 1.1869 CI unavailable (one item) | 1.0886 CI unavailable (one item) | 1.0374 CI unavailable (one item) | 1.1441 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
