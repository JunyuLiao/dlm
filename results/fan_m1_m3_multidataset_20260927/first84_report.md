# Adaptive panel: first84 checkpoint

All 84 planned executions have ledger rows; all six question-seed blocks have seven successful first outputs and seven accepted warm pairs. This is two questions per dataset, seed 101, and is exploratory. The RULER two-question subset is not the official 13-task macro.

Quality uses first outputs only. Time is accepted warm whole-request wall (model load excluded). CUDA event span, when present, starts at the first actual encoder-forward end and ends after generate; it includes host launch gaps and later encoder/commit work, and is not synchronized prefill-excluded generation wall. Absolute means are reported by host only. Paired method/reference ratio <1 is faster; request wall per call is amortized, not direct forward time.

Protocol `v20_fan_4a8f6a589714623e`; panel SHA-256 `99e1d287867b0906b8c5a2715d4cf962ec31e6d1f473c1fe0ce2df5891f1aa17`; binding SHA-256 `ee06c936e500fecfd219c6a9e277f99e77baed7b39d405ad96625f94e50734b5`. No gold or raw completions appear here.

## ruler4k

| Arm | Score mean | Correct/2 | Strict EOS | Task at cap | Parsed/2 | EOS wrong | Output cap | Canvas cap | Native stop | Decoder calls | Calls/canvas pooled | Calls/canvas med/p90 | Output tokens | Tokens/canvas pooled | A/D/H layer calls | First fail/missing | Warm accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B_A8_matched | 0.500 | 1/2 | 1 | 0 | 2/2 | 0 | 1 | 0 | 2 | 8 | 4.00 | 4.00/4.00 | 81 | 40.50 | 10/0/30 | 0/0 | 2/2 |
| D_matched | 0.500 | 1/2 | 1 | 0 | 2/2 | 0 | 1 | 0 | 2 | 10 | 5.00 | 5.00/5.80 | 87 | 43.50 | 50/0/0 | 0/0 | 2/2 |
| D_native | 0.500 | 1/2 | 1 | 0 | 2/2 | 0 | 1 | 0 | 2 | 8 | 4.00 | 4.00/4.00 | 81 | 40.50 | N/A/N/A/N/A | 0/0 | 2/2 |
| M1_R1_A8_current_output | 0.500 | 1/2 | 1 | 0 | 2/2 | 0 | 1 | 0 | 2 | 8 | 4.00 | 4.00/4.00 | 81 | 40.50 | 10/30/0 | 0/0 | 2/2 |
| M3_R2_A8_current_output | 0.500 | 1/2 | 1 | 0 | 2/2 | 0 | 1 | 0 | 2 | 8 | 4.00 | 4.00/4.00 | 81 | 40.50 | 10/10/20 | 0/0 | 2/2 |
| M3_R3_A8_current_output | 0.500 | 1/2 | 1 | 0 | 2/2 | 0 | 1 | 0 | 2 | 8 | 4.00 | 4.00/4.00 | 81 | 40.50 | 10/10/20 | 0/0 | 2/2 |
| T_scope | 0.500 | 1/2 | 1 | 0 | 2/2 | 0 | 1 | 0 | 2 | 8 | 4.00 | 4.00/4.00 | 81 | 40.50 | N/A/N/A/N/A | 0/0 | 2/2 |

Absolute accepted-warm timing by host:

| Arm | Host | Warm whole request mean (s) | CUDA first-encoder-end→finish span mean (s) | Accepted warm requests |
|---|---|---:|---:|---:|
| B_A8_matched | 149.165.151.254 | 1.003 | 0.581 | 1 |
| B_A8_matched | 149.165.159.64 | 0.950 | 0.547 | 1 |
| D_matched | 149.165.151.254 | 1.289 | 0.867 | 1 |
| D_matched | 149.165.159.64 | 0.932 | 0.529 | 1 |
| D_native | 149.165.151.254 | 0.998 | 0.578 | 1 |
| D_native | 149.165.159.64 | 0.939 | 0.535 | 1 |
| M1_R1_A8_current_output | 149.165.151.254 | 1.011 | 0.591 | 1 |
| M1_R1_A8_current_output | 149.165.159.64 | 0.967 | 0.562 | 1 |
| M3_R2_A8_current_output | 149.165.151.254 | 1.007 | 0.585 | 1 |
| M3_R2_A8_current_output | 149.165.159.64 | 0.956 | 0.552 | 1 |
| M3_R3_A8_current_output | 149.165.151.254 | 1.007 | 0.586 | 1 |
| M3_R3_A8_current_output | 149.165.159.64 | 0.956 | 0.553 | 1 |
| T_scope | 149.165.151.254 | 1.003 | 0.583 | 1 |
| T_scope | 149.165.159.64 | 0.949 | 0.545 | 1 |

Same-question, same-GPU paired timing and exact per-host decomposition:

| Method/reference | Paired cells | Geometric request ratio | Host .254 ratio; calls ratio; amortized wall/call | Host .64 ratio; calls ratio; amortized wall/call | Paired score delta |
|---|---:|---:|---|---|---:|
| B_A8_matched/D_native | 2 | 1.008 | n=1; 1.005; 1.000; 1.005 | n=1; 1.011; 1.000; 1.011 | 0.000 |
| B_A8_matched/T_scope | 2 | 1.001 | n=1; 1.000; 1.000; 1.000 | n=1; 1.001; 1.000; 1.001 | 0.000 |
| D_matched/D_native | 2 | 1.131 | n=1; 1.291; 1.500; 0.861 | n=1; 0.992; 1.000; 0.992 | 0.000 |
| D_matched/T_scope | 2 | 1.124 | n=1; 1.285; 1.500; 0.857 | n=1; 0.982; 1.000; 0.982 | 0.000 |
| D_matched/B_A8_matched | 2 | 1.123 | n=1; 1.285; 1.500; 0.857 | n=1; 0.981; 1.000; 0.981 | 0.000 |
| M1_R1_A8_current_output/D_native | 2 | 1.021 | n=1; 1.013; 1.000; 1.013 | n=1; 1.029; 1.000; 1.029 | 0.000 |
| M1_R1_A8_current_output/T_scope | 2 | 1.014 | n=1; 1.008; 1.000; 1.008 | n=1; 1.019; 1.000; 1.019 | 0.000 |
| M1_R1_A8_current_output/B_A8_matched | 2 | 1.013 | n=1; 1.008; 1.000; 1.008 | n=1; 1.018; 1.000; 1.018 | 0.000 |
| M3_R2_A8_current_output/D_native | 2 | 1.013 | n=1; 1.009; 1.000; 1.009 | n=1; 1.017; 1.000; 1.017 | 0.000 |
| M3_R2_A8_current_output/T_scope | 2 | 1.006 | n=1; 1.005; 1.000; 1.005 | n=1; 1.008; 1.000; 1.008 | 0.000 |
| M3_R2_A8_current_output/B_A8_matched | 2 | 1.006 | n=1; 1.005; 1.000; 1.005 | n=1; 1.006; 1.000; 1.006 | 0.000 |
| M3_R3_A8_current_output/D_native | 2 | 1.013 | n=1; 1.009; 1.000; 1.009 | n=1; 1.017; 1.000; 1.017 | 0.000 |
| M3_R3_A8_current_output/T_scope | 2 | 1.006 | n=1; 1.005; 1.000; 1.005 | n=1; 1.008; 1.000; 1.008 | 0.000 |
| M3_R3_A8_current_output/B_A8_matched | 2 | 1.005 | n=1; 1.004; 1.000; 1.004 | n=1; 1.006; 1.000; 1.006 | 0.000 |
| T_scope/D_native | 2 | 1.007 | n=1; 1.004; 1.000; 1.004 | n=1; 1.010; 1.000; 1.010 | 0.000 |
| T_scope/B_A8_matched | 2 | 0.999 | n=1; 1.000; 1.000; 1.000 | n=1; 0.999; 1.000; 0.999 | 0.000 |

Termination counts, per-canvas call histogram, call positions 0/1/2+ and output-token/canvas quantiles for every arm are in the CSV and copied redacted JSON. RULER 13-task macro: N/A (subset).

## aime26

| Arm | Score mean | Correct/2 | Strict EOS | Task at cap | Parsed/2 | EOS wrong | Output cap | Canvas cap | Native stop | Decoder calls | Calls/canvas pooled | Calls/canvas med/p90 | Output tokens | Tokens/canvas pooled | A/D/H layer calls | First fail/missing | Warm accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B_A8_matched | 0.500 | 1/2 | 1 | 0 | 1/2 | 0 | 1 | 0 | 44 | 621 | 14.11 | 12.23/15.55 | 11045 | 251.02 | 480/0/2625 | 0/0 | 2/2 |
| D_matched | 0.500 | 1/2 | 1 | 0 | 1/2 | 0 | 1 | 0 | 48 | 652 | 13.58 | 12.34/15.32 | 12223 | 254.65 | 3260/0/0 | 0/0 | 2/2 |
| D_native | 0.500 | 1/2 | 1 | 0 | 1/2 | 0 | 1 | 0 | 50 | 668 | 13.36 | 12.16/15.58 | 12733 | 254.66 | N/A/N/A/N/A | 0/0 | 2/2 |
| M1_R1_A8_current_output | 0.500 | 1/2 | 1 | 0 | 1/2 | 0 | 1 | 0 | 47 | 577 | 12.28 | 11.51/13.20 | 11782 | 250.68 | 470/2415/0 | 0/0 | 2/2 |
| M3_R2_A8_current_output | 0.500 | 1/2 | 1 | 0 | 1/2 | 0 | 1 | 0 | 47 | 651 | 13.85 | 12.51/15.48 | 11827 | 251.64 | 530/1165/1560 | 0/0 | 2/2 |
| M3_R3_A8_current_output | 0.500 | 1/2 | 1 | 0 | 1/2 | 0 | 1 | 0 | 48 | 718 | 14.96 | 13.33/17.24 | 12167 | 253.48 | 550/850/2190 | 0/0 | 2/2 |
| T_scope | 0.500 | 1/2 | 1 | 0 | 1/2 | 0 | 1 | 0 | 46 | 689 | 14.98 | 13.54/16.48 | 11586 | 251.87 | N/A/N/A/N/A | 0/0 | 2/2 |

Absolute accepted-warm timing by host:

| Arm | Host | Warm whole request mean (s) | CUDA first-encoder-end→finish span mean (s) | Accepted warm requests |
|---|---|---:|---:|---:|
| B_A8_matched | 149.165.151.254 | 78.305 | 78.157 | 1 |
| B_A8_matched | 149.165.159.64 | 14.251 | 14.116 | 1 |
| D_matched | 149.165.151.254 | 76.274 | 76.122 | 1 |
| D_matched | 149.165.159.64 | 19.957 | 19.822 | 1 |
| D_native | 149.165.151.254 | 77.242 | 77.091 | 1 |
| D_native | 149.165.159.64 | 20.599 | 20.464 | 1 |
| M1_R1_A8_current_output | 149.165.151.254 | 68.598 | 68.448 | 1 |
| M1_R1_A8_current_output | 149.165.159.64 | 20.577 | 20.442 | 1 |
| M3_R2_A8_current_output | 149.165.151.254 | 79.280 | 79.129 | 1 |
| M3_R2_A8_current_output | 149.165.159.64 | 19.263 | 19.128 | 1 |
| M3_R3_A8_current_output | 149.165.151.254 | 87.602 | 87.451 | 1 |
| M3_R3_A8_current_output | 149.165.159.64 | 19.833 | 19.698 | 1 |
| T_scope | 149.165.151.254 | 81.982 | 81.832 | 1 |
| T_scope | 149.165.159.64 | 19.883 | 19.748 | 1 |

Same-question, same-GPU paired timing and exact per-host decomposition:

| Method/reference | Paired cells | Geometric request ratio | Host .254 ratio; calls ratio; amortized wall/call | Host .64 ratio; calls ratio; amortized wall/call | Paired score delta |
|---|---:|---:|---|---|---:|
| B_A8_matched/D_native | 2 | 0.837 | n=1; 1.014; 0.996; 1.018 | n=1; 0.692; 0.683; 1.013 | 0.000 |
| B_A8_matched/T_scope | 2 | 0.827 | n=1; 0.955; 0.951; 1.004 | n=1; 0.717; 0.703; 1.020 | 0.000 |
| D_matched/D_native | 2 | 0.978 | n=1; 0.987; 0.977; 1.011 | n=1; 0.969; 0.972; 0.997 | 0.000 |
| D_matched/T_scope | 2 | 0.966 | n=1; 0.930; 0.933; 0.997 | n=1; 1.004; 1.000; 1.004 | 0.000 |
| D_matched/B_A8_matched | 2 | 1.168 | n=1; 0.974; 0.981; 0.993 | n=1; 1.400; 1.423; 0.984 | 0.000 |
| M1_R1_A8_current_output/D_native | 2 | 0.942 | n=1; 0.888; 0.829; 1.071 | n=1; 0.999; 0.993; 1.006 | 0.000 |
| M1_R1_A8_current_output/T_scope | 2 | 0.931 | n=1; 0.837; 0.791; 1.057 | n=1; 1.035; 1.022; 1.013 | 0.000 |
| M1_R1_A8_current_output/B_A8_matched | 2 | 1.125 | n=1; 0.876; 0.832; 1.053 | n=1; 1.444; 1.454; 0.993 | 0.000 |
| M3_R2_A8_current_output/D_native | 2 | 0.980 | n=1; 1.026; 0.987; 1.040 | n=1; 0.935; 0.930; 1.006 | 0.000 |
| M3_R2_A8_current_output/T_scope | 2 | 0.968 | n=1; 0.967; 0.942; 1.027 | n=1; 0.969; 0.957; 1.013 | 0.000 |
| M3_R2_A8_current_output/B_A8_matched | 2 | 1.170 | n=1; 1.012; 0.990; 1.022 | n=1; 1.352; 1.361; 0.993 | 0.000 |
| M3_R3_A8_current_output/D_native | 2 | 1.045 | n=1; 1.134; 1.108; 1.023 | n=1; 0.963; 0.951; 1.013 | 0.000 |
| M3_R3_A8_current_output/T_scope | 2 | 1.032 | n=1; 1.069; 1.058; 1.010 | n=1; 0.997; 0.978; 1.020 | 0.000 |
| M3_R3_A8_current_output/B_A8_matched | 2 | 1.248 | n=1; 1.119; 1.113; 1.006 | n=1; 1.392; 1.392; 1.000 | 0.000 |
| T_scope/D_native | 2 | 1.012 | n=1; 1.061; 1.048; 1.013 | n=1; 0.965; 0.972; 0.993 | 0.000 |
| T_scope/B_A8_matched | 2 | 1.209 | n=1; 1.047; 1.052; 0.996 | n=1; 1.395; 1.423; 0.981 | 0.000 |

Termination counts, per-canvas call histogram, call positions 0/1/2+ and output-token/canvas quantiles for every arm are in the CSV and copied redacted JSON. RULER 13-task macro: not applicable.

## longbench_v2

| Arm | Score mean | Correct/2 | Strict EOS | Task at cap | Parsed/2 | EOS wrong | Output cap | Canvas cap | Native stop | Decoder calls | Calls/canvas pooled | Calls/canvas med/p90 | Output tokens | Tokens/canvas pooled | A/D/H layer calls | First fail/missing | Warm accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B_A8_matched | 0.500 | 1/2 | 1 | 0 | 2/2 | 1 | 0 | 0 | 38 | 881 | 23.18 | 23.16/23.23 | 9339 | 245.76 | 635/0/3770 | 0/0 | 2/2 |
| D_matched | 0.500 | 1/2 | 1 | 0 | 2/2 | 1 | 0 | 1 | 29 | 706 | 23.53 | 23.56/23.91 | 7417 | 247.23 | 3530/0/0 | 0/0 | 2/2 |
| D_native | 1.000 | 2/2 | 2 | 0 | 2/2 | 0 | 0 | 0 | 32 | 707 | 22.09 | 22.12/22.45 | 7951 | 248.47 | N/A/N/A/N/A | 0/0 | 2/2 |
| M1_R1_A8_current_output | 1.000 | 2/2 | 2 | 0 | 2/2 | 0 | 0 | 0 | 34 | 700 | 20.59 | 20.60/20.71 | 8445 | 248.38 | 500/3000/0 | 0/0 | 2/2 |
| M3_R2_A8_current_output | 1.000 | 2/2 | 2 | 0 | 2/2 | 0 | 0 | 0 | 39 | 836 | 21.44 | 21.74/23.10 | 9736 | 249.64 | 610/1525/2045 | 0/0 | 2/2 |
| M3_R3_A8_current_output | 1.000 | 2/2 | 2 | 0 | 2/2 | 0 | 0 | 0 | 24 | 483 | 20.12 | 19.89/21.01 | 5827 | 242.79 | 350/585/1480 | 0/0 | 2/2 |
| T_scope | 1.000 | 2/2 | 2 | 0 | 2/2 | 0 | 0 | 0 | 22 | 474 | 21.55 | 20.65/23.27 | 5415 | 246.14 | N/A/N/A/N/A | 0/0 | 2/2 |

Absolute accepted-warm timing by host:

| Arm | Host | Warm whole request mean (s) | CUDA first-encoder-end→finish span mean (s) | Accepted warm requests |
|---|---|---:|---:|---:|
| B_A8_matched | 149.165.151.254 | 93.299 | 91.308 | 1 |
| B_A8_matched | 149.165.159.64 | 52.611 | 49.533 | 1 |
| D_matched | 149.165.151.254 | 65.699 | 63.463 | 1 |
| D_matched | 149.165.159.64 | 58.999 | 55.922 | 1 |
| D_native | 149.165.151.254 | 66.603 | 64.611 | 1 |
| D_native | 149.165.159.64 | 61.107 | 58.031 | 1 |
| M1_R1_A8_current_output | 149.165.151.254 | 69.595 | 67.606 | 1 |
| M1_R1_A8_current_output | 149.165.159.64 | 54.129 | 51.055 | 1 |
| M3_R2_A8_current_output | 149.165.151.254 | 79.742 | 77.753 | 1 |
| M3_R2_A8_current_output | 149.165.159.64 | 62.425 | 59.349 | 1 |
| M3_R3_A8_current_output | 149.165.151.254 | 32.840 | 30.851 | 1 |
| M3_R3_A8_current_output | 149.165.159.64 | 50.195 | 46.892 | 1 |
| T_scope | 149.165.151.254 | 26.122 | 24.132 | 1 |
| T_scope | 149.165.159.64 | 59.492 | 56.190 | 1 |

Same-question, same-GPU paired timing and exact per-host decomposition:

| Method/reference | Paired cells | Geometric request ratio | Host .254 ratio; calls ratio; amortized wall/call | Host .64 ratio; calls ratio; amortized wall/call | Paired score delta |
|---|---:|---:|---|---|---:|
| B_A8_matched/D_native | 2 | 1.098 | n=1; 1.401; 1.512; 0.926 | n=1; 0.861; 0.956; 0.901 | -0.500 |
| B_A8_matched/T_scope | 2 | 1.777 | n=1; 3.572; 4.014; 0.890 | n=1; 0.884; 0.964; 0.917 | -0.500 |
| D_matched/D_native | 2 | 0.976 | n=1; 0.986; 1.003; 0.984 | n=1; 0.966; 0.994; 0.971 | -0.500 |
| D_matched/T_scope | 2 | 1.579 | n=1; 2.515; 2.662; 0.945 | n=1; 0.992; 1.003; 0.989 | -0.500 |
| D_matched/B_A8_matched | 2 | 0.889 | n=1; 0.704; 0.663; 1.062 | n=1; 1.121; 1.040; 1.078 | 0.000 |
| M1_R1_A8_current_output/D_native | 2 | 0.962 | n=1; 1.045; 1.054; 0.991 | n=1; 0.886; 0.920; 0.963 | 0.000 |
| M1_R1_A8_current_output/T_scope | 2 | 1.557 | n=1; 2.664; 2.799; 0.952 | n=1; 0.910; 0.928; 0.980 | 0.000 |
| M1_R1_A8_current_output/B_A8_matched | 2 | 0.876 | n=1; 0.746; 0.697; 1.070 | n=1; 1.029; 0.963; 1.069 | 0.500 |
| M3_R2_A8_current_output/D_native | 2 | 1.106 | n=1; 1.197; 1.249; 0.958 | n=1; 1.022; 1.109; 0.921 | 0.000 |
| M3_R2_A8_current_output/T_scope | 2 | 1.790 | n=1; 3.053; 3.317; 0.920 | n=1; 1.049; 1.119; 0.937 | 0.000 |
| M3_R2_A8_current_output/B_A8_matched | 2 | 1.007 | n=1; 0.855; 0.826; 1.035 | n=1; 1.187; 1.161; 1.022 | 0.500 |
| M3_R3_A8_current_output/D_native | 2 | 0.636 | n=1; 0.493; 0.501; 0.983 | n=1; 0.821; 0.882; 0.932 | 0.000 |
| M3_R3_A8_current_output/T_scope | 2 | 1.030 | n=1; 1.257; 1.331; 0.945 | n=1; 0.844; 0.890; 0.948 | 0.000 |
| M3_R3_A8_current_output/B_A8_matched | 2 | 0.579 | n=1; 0.352; 0.332; 1.062 | n=1; 0.954; 0.923; 1.034 | 0.500 |
| T_scope/D_native | 2 | 0.618 | n=1; 0.392; 0.377; 1.041 | n=1; 0.974; 0.991; 0.982 | 0.000 |
| T_scope/B_A8_matched | 2 | 0.563 | n=1; 0.280; 0.249; 1.124 | n=1; 1.131; 1.037; 1.090 | 0.500 |

Termination counts, per-canvas call histogram, call positions 0/1/2+ and output-token/canvas quantiles for every arm are in the CSV and copied redacted JSON. RULER 13-task macro: not applicable.

The two-question exploratory checkpoint cannot establish an accuracy margin, speedup across datasets, or the result of the planned full panel.
