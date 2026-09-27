# v21 trajectory audit

Verified 800 published requests (400 first, 400 accepted warm), manifest SHA-256 `c4358f2f658b532dc07f5c9f1652088f95f308146f035ff47ac53cf7288efbd1`.
Qualified first-cell score status: **complete**. The score source is the frozen v20 `score_firsts` export; no new scoring rule is used.
The export manifest binds CRLF protocol bytes `addd616b84bb1c8428e9d4e7286a6fcb65a9e476a96d69b9a743bfc9e6a1b1bb`; the scorer binds deployed/Git LF bytes `99e1d287867b0906b8c5a2715d4cf962ec31e6d1f473c1fe0ce2df5891f1aa17`. Replacing each LF with CRLF gives the manifest hash exactly, with the same parsed JSON.

The CSV has one row per first cell, with its accepted warm timing. Stop categories are disjoint per canvas. Calls equal the sum of canvas calls; `calls_per_canvas` distributions and pooled group distributions are in JSON.

| Dataset | Host | Arm | Cells | Calls mean | Canvases mean | Warm wall mean (s) | Strict correct | Cap-only canvases | Native-stop-only canvases |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| aime26 | 149.165.151.254 | B_A8_matched | 6 | 367.0 | 27.5 | 55.56 | 3 | 0 | 165 |
| aime26 | 149.165.151.254 | D_matched | 6 | 351.3 | 26.0 | 52.32 | 2 | 0 | 156 |
| aime26 | 149.165.151.254 | D_native | 6 | 365.5 | 26.3 | 54.42 | 1 | 0 | 158 |
| aime26 | 149.165.151.254 | G75L30_nativeQ128 | 6 | 335.8 | 24.8 | 51.39 | 3 | 0 | 149 |
| aime26 | 149.165.151.254 | M1_R1_A8_current_output | 6 | 340.8 | 24.0 | 53.31 | 3 | 0 | 144 |
| aime26 | 149.165.151.254 | M3_R2_A8_current_output | 6 | 412.3 | 28.5 | 63.17 | 2 | 1 | 170 |
| aime26 | 149.165.151.254 | M3_R3_A8_current_output | 6 | 366.7 | 26.7 | 56.03 | 3 | 0 | 160 |
| aime26 | 149.165.151.254 | T_scope | 6 | 353.7 | 23.5 | 53.24 | 3 | 0 | 141 |
| aime26 | 149.165.159.64 | B_A8_matched | 6 | 245.7 | 21.0 | 35.17 | 4 | 0 | 126 |
| aime26 | 149.165.159.64 | D_matched | 6 | 194.3 | 17.8 | 27.65 | 5 | 0 | 107 |
| aime26 | 149.165.159.64 | D_native | 6 | 234.5 | 20.8 | 33.59 | 4 | 0 | 125 |
| aime26 | 149.165.159.64 | G75L30_nativeQ128 | 6 | 225.3 | 18.8 | 34.56 | 5 | 0 | 113 |
| aime26 | 149.165.159.64 | M1_R1_A8_current_output | 6 | 258.3 | 20.8 | 37.81 | 4 | 0 | 125 |
| aime26 | 149.165.159.64 | M3_R2_A8_current_output | 6 | 234.8 | 20.5 | 33.69 | 4 | 0 | 123 |
| aime26 | 149.165.159.64 | M3_R3_A8_current_output | 6 | 277.7 | 23.3 | 40.12 | 2 | 0 | 140 |
| aime26 | 149.165.159.64 | T_scope | 6 | 260.7 | 21.7 | 37.29 | 5 | 1 | 129 |
| longbench_v2 | 149.165.151.254 | B_A8_matched | 6 | 204.7 | 12.5 | 36.22 | 3 | 0 | 75 |
| longbench_v2 | 149.165.151.254 | D_matched | 6 | 159.3 | 10.8 | 30.44 | 3 | 1 | 64 |
| longbench_v2 | 149.165.151.254 | D_native | 6 | 155.0 | 11.0 | 30.10 | 4 | 0 | 66 |
| longbench_v2 | 149.165.151.254 | G75L30_nativeQ128 | 6 | 165.5 | 11.2 | 29.63 | 3 | 0 | 67 |
| longbench_v2 | 149.165.151.254 | M1_R1_A8_current_output | 6 | 173.3 | 11.7 | 32.75 | 4 | 0 | 70 |
| longbench_v2 | 149.165.151.254 | M3_R2_A8_current_output | 6 | 214.7 | 13.8 | 39.04 | 4 | 0 | 83 |
| longbench_v2 | 149.165.151.254 | M3_R3_A8_current_output | 6 | 177.7 | 12.2 | 32.45 | 4 | 0 | 73 |
| longbench_v2 | 149.165.151.254 | T_scope | 6 | 114.8 | 8.8 | 22.82 | 4 | 0 | 53 |
| longbench_v2 | 149.165.159.64 | B_A8_matched | 6 | 258.5 | 13.0 | 43.46 | 1 | 0 | 78 |
| longbench_v2 | 149.165.159.64 | D_matched | 6 | 342.2 | 13.8 | 56.03 | 1 | 10 | 72 |
| longbench_v2 | 149.165.159.64 | D_native | 6 | 238.3 | 12.3 | 43.37 | 2 | 0 | 74 |
| longbench_v2 | 149.165.159.64 | G75L30_nativeQ128 | 6 | 307.5 | 14.0 | 51.97 | 0 | 1 | 83 |
| longbench_v2 | 149.165.159.64 | M1_R1_A8_current_output | 6 | 263.8 | 13.2 | 47.41 | 2 | 0 | 79 |
| longbench_v2 | 149.165.159.64 | M3_R2_A8_current_output | 6 | 259.0 | 12.7 | 44.51 | 2 | 0 | 76 |
| longbench_v2 | 149.165.159.64 | M3_R3_A8_current_output | 6 | 241.3 | 12.3 | 41.61 | 1 | 0 | 74 |
| longbench_v2 | 149.165.159.64 | T_scope | 6 | 255.7 | 12.5 | 45.45 | 2 | 0 | 75 |
| ruler4k | 149.165.151.254 | B_A8_matched | 13 | 5.5 | 1.0 | 1.22 | 12 | 0 | 13 |
| ruler4k | 149.165.151.254 | D_matched | 13 | 4.2 | 1.0 | 1.00 | 12 | 0 | 13 |
| ruler4k | 149.165.151.254 | D_native | 13 | 4.2 | 1.0 | 0.99 | 12 | 0 | 13 |
| ruler4k | 149.165.151.254 | G75L30_nativeQ128 | 13 | 4.5 | 1.0 | 1.07 | 12 | 0 | 13 |
| ruler4k | 149.165.151.254 | M1_R1_A8_current_output | 13 | 4.8 | 1.0 | 1.13 | 12 | 0 | 13 |
| ruler4k | 149.165.151.254 | M3_R2_A8_current_output | 13 | 5.5 | 1.0 | 1.24 | 12 | 0 | 13 |
| ruler4k | 149.165.151.254 | M3_R3_A8_current_output | 13 | 5.2 | 1.0 | 1.16 | 12 | 0 | 13 |
| ruler4k | 149.165.151.254 | T_scope | 13 | 4.1 | 1.0 | 0.99 | 12 | 0 | 13 |
| ruler4k | 149.165.159.64 | B_A8_matched | 13 | 3.5 | 1.0 | 0.85 | 12 | 0 | 13 |
| ruler4k | 149.165.159.64 | D_matched | 13 | 3.8 | 1.0 | 0.88 | 12 | 0 | 13 |
| ruler4k | 149.165.159.64 | D_native | 13 | 3.5 | 1.0 | 0.84 | 12 | 0 | 13 |
| ruler4k | 149.165.159.64 | G75L30_nativeQ128 | 13 | 3.6 | 1.0 | 0.95 | 12 | 0 | 13 |
| ruler4k | 149.165.159.64 | M1_R1_A8_current_output | 13 | 3.3 | 1.0 | 0.84 | 12 | 0 | 13 |
| ruler4k | 149.165.159.64 | M3_R2_A8_current_output | 13 | 3.2 | 1.0 | 0.83 | 12 | 0 | 13 |
| ruler4k | 149.165.159.64 | M3_R3_A8_current_output | 13 | 3.3 | 1.0 | 0.83 | 12 | 0 | 13 |
| ruler4k | 149.165.159.64 | T_scope | 13 | 3.5 | 1.0 | 0.87 | 12 | 0 | 13 |

Same-GPU cell ratios to D_native, T_scope, and B_A8_matched are in JSON and CSV; a ratio below 1 means less warm request wall time or fewer first decoder calls. Absolute times are shown separately by host.

| Dataset | Host | Arm / reference | Paired cells | Warm wall ratio of totals | First call ratio of totals |
|---|---|---|---:|---:|---:|
| aime26 | 149.165.151.254 | D_matched / B_A8_matched | 6 | 0.942 | 0.957 |
| aime26 | 149.165.151.254 | D_matched / D_native | 6 | 0.961 | 0.961 |
| aime26 | 149.165.151.254 | D_matched / T_scope | 6 | 0.983 | 0.993 |
| aime26 | 149.165.151.254 | M3_R2_A8_current_output / B_A8_matched | 6 | 1.137 | 1.124 |
| aime26 | 149.165.151.254 | M3_R2_A8_current_output / D_native | 6 | 1.161 | 1.128 |
| aime26 | 149.165.151.254 | M3_R2_A8_current_output / T_scope | 6 | 1.186 | 1.166 |
| aime26 | 149.165.151.254 | M3_R3_A8_current_output / B_A8_matched | 6 | 1.008 | 0.999 |
| aime26 | 149.165.151.254 | M3_R3_A8_current_output / D_native | 6 | 1.030 | 1.003 |
| aime26 | 149.165.151.254 | M3_R3_A8_current_output / T_scope | 6 | 1.052 | 1.037 |
| aime26 | 149.165.159.64 | D_matched / B_A8_matched | 6 | 0.786 | 0.791 |
| aime26 | 149.165.159.64 | D_matched / D_native | 6 | 0.823 | 0.829 |
| aime26 | 149.165.159.64 | D_matched / T_scope | 6 | 0.742 | 0.746 |
| aime26 | 149.165.159.64 | M3_R2_A8_current_output / B_A8_matched | 6 | 0.958 | 0.956 |
| aime26 | 149.165.159.64 | M3_R2_A8_current_output / D_native | 6 | 1.003 | 1.001 |
| aime26 | 149.165.159.64 | M3_R2_A8_current_output / T_scope | 6 | 0.903 | 0.901 |
| aime26 | 149.165.159.64 | M3_R3_A8_current_output / B_A8_matched | 6 | 1.141 | 1.130 |
| aime26 | 149.165.159.64 | M3_R3_A8_current_output / D_native | 6 | 1.194 | 1.184 |
| aime26 | 149.165.159.64 | M3_R3_A8_current_output / T_scope | 6 | 1.076 | 1.065 |
| longbench_v2 | 149.165.151.254 | D_matched / B_A8_matched | 6 | 0.840 | 0.779 |
| longbench_v2 | 149.165.151.254 | D_matched / D_native | 6 | 1.011 | 1.028 |
| longbench_v2 | 149.165.151.254 | D_matched / T_scope | 6 | 1.334 | 1.388 |
| longbench_v2 | 149.165.151.254 | M3_R2_A8_current_output / B_A8_matched | 6 | 1.078 | 1.049 |
| longbench_v2 | 149.165.151.254 | M3_R2_A8_current_output / D_native | 6 | 1.297 | 1.385 |
| longbench_v2 | 149.165.151.254 | M3_R2_A8_current_output / T_scope | 6 | 1.710 | 1.869 |
| longbench_v2 | 149.165.151.254 | M3_R3_A8_current_output / B_A8_matched | 6 | 0.896 | 0.868 |
| longbench_v2 | 149.165.151.254 | M3_R3_A8_current_output / D_native | 6 | 1.078 | 1.146 |
| longbench_v2 | 149.165.151.254 | M3_R3_A8_current_output / T_scope | 6 | 1.422 | 1.547 |
| longbench_v2 | 149.165.159.64 | D_matched / B_A8_matched | 6 | 1.289 | 1.324 |
| longbench_v2 | 149.165.159.64 | D_matched / D_native | 6 | 1.292 | 1.436 |
| longbench_v2 | 149.165.159.64 | D_matched / T_scope | 6 | 1.233 | 1.338 |
| longbench_v2 | 149.165.159.64 | M3_R2_A8_current_output / B_A8_matched | 6 | 1.024 | 1.002 |
| longbench_v2 | 149.165.159.64 | M3_R2_A8_current_output / D_native | 6 | 1.026 | 1.087 |
| longbench_v2 | 149.165.159.64 | M3_R2_A8_current_output / T_scope | 6 | 0.979 | 1.013 |
| longbench_v2 | 149.165.159.64 | M3_R3_A8_current_output / B_A8_matched | 6 | 0.957 | 0.934 |
| longbench_v2 | 149.165.159.64 | M3_R3_A8_current_output / D_native | 6 | 0.959 | 1.013 |
| longbench_v2 | 149.165.159.64 | M3_R3_A8_current_output / T_scope | 6 | 0.915 | 0.944 |
| ruler4k | 149.165.151.254 | D_matched / B_A8_matched | 13 | 0.825 | 0.764 |
| ruler4k | 149.165.151.254 | D_matched / D_native | 13 | 1.009 | 1.019 |
| ruler4k | 149.165.151.254 | D_matched / T_scope | 13 | 1.009 | 1.038 |
| ruler4k | 149.165.151.254 | M3_R2_A8_current_output / B_A8_matched | 13 | 1.017 | 1.000 |
| ruler4k | 149.165.151.254 | M3_R2_A8_current_output / D_native | 13 | 1.245 | 1.333 |
| ruler4k | 149.165.151.254 | M3_R2_A8_current_output / T_scope | 13 | 1.244 | 1.358 |
| ruler4k | 149.165.151.254 | M3_R3_A8_current_output / B_A8_matched | 13 | 0.956 | 0.931 |
| ruler4k | 149.165.151.254 | M3_R3_A8_current_output / D_native | 13 | 1.170 | 1.241 |
| ruler4k | 149.165.151.254 | M3_R3_A8_current_output / T_scope | 13 | 1.169 | 1.264 |
| ruler4k | 149.165.159.64 | D_matched / B_A8_matched | 13 | 1.032 | 1.089 |
| ruler4k | 149.165.159.64 | D_matched / D_native | 13 | 1.043 | 1.089 |
| ruler4k | 149.165.159.64 | D_matched / T_scope | 13 | 1.017 | 1.089 |
| ruler4k | 149.165.159.64 | M3_R2_A8_current_output / B_A8_matched | 13 | 0.969 | 0.933 |
| ruler4k | 149.165.159.64 | M3_R2_A8_current_output / D_native | 13 | 0.979 | 0.933 |
| ruler4k | 149.165.159.64 | M3_R2_A8_current_output / T_scope | 13 | 0.954 | 0.933 |
| ruler4k | 149.165.159.64 | M3_R3_A8_current_output / B_A8_matched | 13 | 0.978 | 0.956 |
| ruler4k | 149.165.159.64 | M3_R3_A8_current_output / D_native | 13 | 0.989 | 0.956 |
| ruler4k | 149.165.159.64 | M3_R3_A8_current_output / T_scope | 13 | 0.964 | 0.956 |

## Predeclared text-inspection cells

Before text access: for LB select largest positive/negative M3R2-minus-native call delta and two largest absolute Dmatched-minus-native discordances without duplicate q/seed; for AIME select positive/negative M3R2 extremes, tie by id then seed.

- longbench_v2 `longbench_v2/66f2aac2821e116aacb2a9de` seed 101: largest_positive_M3R2_call_delta; M3R2−native +143 calls, Dmatched−native -24 calls.
- longbench_v2 `longbench_v2/66ed2c87821e116aacb1f149` seed 101: largest_negative_M3R2_call_delta; M3R2−native -52 calls, Dmatched−native -55 calls.
- longbench_v2 `longbench_v2/66f120a9821e116aacb26d02` seed 202: largest_Dmatched_discordance; M3R2−native +27 calls, Dmatched−native +499 calls.
- longbench_v2 `longbench_v2/66f9625fbb02136c067c5456` seed 101: second_Dmatched_discordance; M3R2−native +98 calls, Dmatched−native +161 calls.
- aime26 `aime26/23` seed 202: largest_positive_M3R2_call_delta; M3R2−native +262 calls, Dmatched−native +13 calls.
- aime26 `aime26/2` seed 101: largest_negative_M3R2_call_delta; M3R2−native -119 calls, Dmatched−native -100 calls.

| Selected arm | First completions | Mean repeated 16-token window fraction | No final-channel delimiter | Length-capped | No delimiter and capped |
|---|---:|---:|---:|---:|---:|
| D_native | 6 | 0.069 | 0 | 0 | 0 |
| D_matched | 6 | 0.163 | 1 | 0 | 0 |
| T_scope | 6 | 0.086 | 0 | 0 | 0 |
| B_A8_matched | 6 | 0.098 | 1 | 1 | 1 |
| M1_R1_A8_current_output | 6 | 0.090 | 0 | 0 | 0 |
| M3_R2_A8_current_output | 6 | 0.100 | 0 | 0 | 0 |
| M3_R3_A8_current_output | 6 | 0.080 | 1 | 2 | 1 |
| G75L30_nativeQ128 | 6 | 0.074 | 0 | 0 | 0 |

Repetition is the fraction of 16-token sliding windows previously seen within that same first completion; `final_channel_delimiter_present` only checks the literal `<channel|>` delimiter used by the existing AIME parser and is not a new correctness rule. Selected-cell numbers are in JSON.

## Limits

- No full per-step QKV/attention tensors or per-decision age distribution were recorded.
- The export manifest protocol SHA is exactly the CRLF expansion of the deployed/Git LF protocol bytes; parsed JSON and all cell identities are unchanged. The qualified scorer binds the LF bytes.
- Token-prefix and aligned-chunk identity is descriptive; after divergence positional tokens need not represent the same semantic step.
- The device timeline is first encoder-forward end to final CUDA event, including host gaps and later work; it is not synchronized decode wall time.
- Historical G75L30 follows core execution and may have temporal drift.
