# V29 LongBench generation/scorer qualification

Qualification only: 12/12 pipeline workers completed. These singleton records are not accuracy or speed evidence; all wrong, capped and unparsed outputs remain scored failures. Ratios and scorer-generated statistics below are retained as diagnostic details only.

dataset/index/repeat; repeat is not a sampling seed; seed_applied=false

95% item-cluster percentile bootstrap retaining all repeats

W=request boundary wall; S=decode span excluding initial prefill; P=initial prefill; N=actual denoising forwards including speculative unused execution and excluding encoder commits; SN=S/N amortized cost

Observed differences and question-cluster CIs only; noninferiority margin unset. No noninferiority decision. McNemar is exploratory; repeats share questions.

| dataset | arm | cells | items | strict correct | task correct | cap | unparsed |
|---|---|---:|---:|---:|---:|---:|---:|
| longbench_v2_32k | dense | 1 | 1 | 1 | 1 | 0 | 0 |
| longbench_v2_32k | method | 1 | 1 | 1 | 1 | 0 | 0 |
| longbench_v2_32k | native | 1 | 1 | 1 | 1 | 0 | 0 |
| longbench_v2_32k | allkept | 1 | 1 | 1 | 1 | 0 | 0 |
| longbench_v2_64k | dense | 1 | 1 | 0 | 0 | 1 | 1 |
| longbench_v2_64k | method | 1 | 1 | 1 | 1 | 0 | 0 |
| longbench_v2_64k | native | 1 | 1 | 1 | 1 | 0 | 0 |
| longbench_v2_64k | allkept | 1 | 1 | 1 | 1 | 0 | 0 |
| longbench_v2_96k | dense | 1 | 1 | 0 | 0 | 0 | 0 |
| longbench_v2_96k | method | 1 | 1 | 0 | 0 | 0 | 0 |
| longbench_v2_96k | native | 1 | 1 | 0 | 0 | 0 | 0 |
| longbench_v2_96k | allkept | 1 | 1 | 0 | 0 | 0 | 0 |

Per-arm means/medians use all timed requests; decode_s_per_denoise_forward_geomean is the geometric mean of request S/N; commit mean uses only its explicitly reported observed cells

| dataset | arm | W mean / median (s) | S mean / median (s) | prefill mean (s) | denoise N mean / median / sum | unused speculative denoise mean | scheduler denoise sum | S/N geometric mean (s) | commit forwards mean (observed cells) |
|---|---|---|---|---:|---|---:|---:|---:|---|
| longbench_v2_32k | dense | 5.3394 / 5.3394 | 4.3002 / 4.3002 | 1.0392 | 136.0000 / 136.0000 / 136 | 1.0000 | 135 | 0.031619 | 16.0000 (1) |
| longbench_v2_32k | method | 3.7191 / 3.7191 | 2.6753 / 2.6753 | 1.0438 | 82.0000 / 82.0000 / 82 | 1.0000 | 81 | 0.032626 | 11.0000 (1) |
| longbench_v2_32k | native | 7.1335 / 7.1335 | 6.0949 / 6.0949 | 1.0386 | 183.0000 / 183.0000 / 183 | 1.0000 | 182 | 0.033305 | 23.0000 (1) |
| longbench_v2_32k | allkept | 4.0043 / 4.0043 | 2.9608 / 2.9608 | 1.0435 | 87.0000 / 87.0000 / 87 | 1.0000 | 86 | 0.034033 | 13.0000 (1) |
| longbench_v2_64k | dense | 49.8609 / 49.8609 | 47.2812 / 47.2812 | 2.5798 | 1131.0000 / 1131.0000 / 1131 | 1.0000 | 1130 | 0.041805 | 32.0000 (1) |
| longbench_v2_64k | method | 18.2512 / 18.2512 | 15.6506 / 15.6506 | 2.6006 | 422.0000 / 422.0000 / 422 | 1.0000 | 421 | 0.037087 | 16.0000 (1) |
| longbench_v2_64k | native | 24.1257 / 24.1257 | 21.5244 / 21.5244 | 2.6013 | 497.0000 / 497.0000 / 497 | 1.0000 | 496 | 0.043309 | 24.0000 (1) |
| longbench_v2_64k | allkept | 22.3833 / 22.3833 | 19.7851 / 19.7851 | 2.5982 | 469.0000 / 469.0000 / 469 | 1.0000 | 468 | 0.042186 | 18.0000 (1) |
| longbench_v2_96k | dense | 22.1867 / 22.1867 | 18.4822 / 18.4822 | 3.7045 | 381.0000 / 381.0000 / 381 | 1.0000 | 380 | 0.048510 | 17.0000 (1) |
| longbench_v2_96k | method | 14.5220 / 14.5220 | 10.7812 / 10.7812 | 3.7408 | 245.0000 / 245.0000 / 245 | 1.0000 | 244 | 0.044005 | 16.0000 (1) |
| longbench_v2_96k | native | 23.1573 / 23.1573 | 19.4437 / 19.4437 | 3.7136 | 389.0000 / 389.0000 / 389 | 1.0000 | 388 | 0.049984 | 23.0000 (1) |
| longbench_v2_96k | allkept | 11.7442 / 11.7442 | 8.0059 / 8.0059 | 3.7382 | 157.0000 / 157.0000 / 157 | 1.0000 | 156 | 0.050993 | 13.0000 (1) |

Ratios are candidate/reference; values below one mean faster or fewer forwards.

| dataset | comparison | cells | items | correct (base) | W [95% CI] | S [95% CI] | prefill P [95% CI] | S/N [95% CI] | N [95% CI] | accuracy difference [95% CI] | exploratory McNemar p |
|---|---|---:|---:|---|---|---|---|---|---|---|---:|
| longbench_v2_32k | method/dense | 1 | 1 | 1 (1) | 0.6965 CI unavailable (one item) | 0.6221 CI unavailable (one item) | 1.0044 CI unavailable (one item) | 1.0318 CI unavailable (one item) | 0.6029 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_32k | native/dense | 1 | 1 | 1 (1) | 1.3360 CI unavailable (one item) | 1.4174 CI unavailable (one item) | 0.9995 CI unavailable (one item) | 1.0533 CI unavailable (one item) | 1.3456 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_32k | allkept/dense | 1 | 1 | 1 (1) | 0.7500 CI unavailable (one item) | 0.6885 CI unavailable (one item) | 1.0042 CI unavailable (one item) | 1.0763 CI unavailable (one item) | 0.6397 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_32k | method/native | 1 | 1 | 1 (1) | 0.5214 CI unavailable (one item) | 0.4389 CI unavailable (one item) | 1.0049 CI unavailable (one item) | 0.9796 CI unavailable (one item) | 0.4481 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_32k | method/allkept | 1 | 1 | 1 (1) | 0.9288 CI unavailable (one item) | 0.9036 CI unavailable (one item) | 1.0003 CI unavailable (one item) | 0.9587 CI unavailable (one item) | 0.9425 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_64k | method/dense | 1 | 1 | 1 (0) | 0.3660 CI unavailable (one item) | 0.3310 CI unavailable (one item) | 1.0081 CI unavailable (one item) | 0.8871 CI unavailable (one item) | 0.3731 CI unavailable (one item) | 1.0000 CI unavailable (one item) | 1 |
| longbench_v2_64k | native/dense | 1 | 1 | 1 (0) | 0.4839 CI unavailable (one item) | 0.4552 CI unavailable (one item) | 1.0083 CI unavailable (one item) | 1.0360 CI unavailable (one item) | 0.4394 CI unavailable (one item) | 1.0000 CI unavailable (one item) | 1 |
| longbench_v2_64k | allkept/dense | 1 | 1 | 1 (0) | 0.4489 CI unavailable (one item) | 0.4185 CI unavailable (one item) | 1.0072 CI unavailable (one item) | 1.0091 CI unavailable (one item) | 0.4147 CI unavailable (one item) | 1.0000 CI unavailable (one item) | 1 |
| longbench_v2_64k | method/native | 1 | 1 | 1 (1) | 0.7565 CI unavailable (one item) | 0.7271 CI unavailable (one item) | 0.9997 CI unavailable (one item) | 0.8563 CI unavailable (one item) | 0.8491 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_64k | method/allkept | 1 | 1 | 1 (1) | 0.8154 CI unavailable (one item) | 0.7910 CI unavailable (one item) | 1.0009 CI unavailable (one item) | 0.8791 CI unavailable (one item) | 0.8998 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_96k | method/dense | 1 | 1 | 0 (0) | 0.6545 CI unavailable (one item) | 0.5833 CI unavailable (one item) | 1.0098 CI unavailable (one item) | 0.9071 CI unavailable (one item) | 0.6430 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_96k | native/dense | 1 | 1 | 0 (0) | 1.0437 CI unavailable (one item) | 1.0520 CI unavailable (one item) | 1.0025 CI unavailable (one item) | 1.0304 CI unavailable (one item) | 1.0210 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_96k | allkept/dense | 1 | 1 | 0 (0) | 0.5293 CI unavailable (one item) | 0.4332 CI unavailable (one item) | 1.0091 CI unavailable (one item) | 1.0512 CI unavailable (one item) | 0.4121 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_96k | method/native | 1 | 1 | 0 (0) | 0.6271 CI unavailable (one item) | 0.5545 CI unavailable (one item) | 1.0073 CI unavailable (one item) | 0.8804 CI unavailable (one item) | 0.6298 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
| longbench_v2_96k | method/allkept | 1 | 1 | 0 (0) | 1.2365 CI unavailable (one item) | 1.3467 CI unavailable (one item) | 1.0007 CI unavailable (one item) | 0.8630 CI unavailable (one item) | 1.5605 CI unavailable (one item) | 0.0000 CI unavailable (one item) | 1 |
