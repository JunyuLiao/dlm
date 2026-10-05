# Historical G75L30 native-Q128 extension (separate 100 executions)

The optional historical extension has 100/100 execution rows: 50 successful first outputs and 50 accepted warm runs. It uses ALL_NATIVE_LEGAL scope even though the core M1/M3 panel uses a different selected scope. The extension is outside both core700 and first84 counts. Its first outputs are scored with the same frozen task gold/scorers as core; no answers, gold, prompts or raw completions are published here.

Comparisons use the same frozen question, seed and GPU as core native/T/B. Historical runs happened later; all time contrasts are descriptive and may include temporal drift. Absolute accepted-warm whole-request wall is shown by host only. The CUDA first-encoder-forward-end to finish span includes host gaps and later encoder/commit work; it is not synchronized prefill-excluded generation wall. Ratio historical/reference <1 is faster. Exact per-host total-time decomposition is decoder-call ratio × amortized request-wall-per-call ratio; the latter is not direct model-forward time.

Redacted source SHA-256 ae013630235ba0ba5212ea35ae627d5e4c7edf9dd1b56718b6bedb5514237107; panel SHA-256 99e1d287867b0906b8c5a2715d4cf962ec31e6d1f473c1fe0ce2df5891f1aa17; binding SHA-256 ee06c936e500fecfd219c6a9e277f99e77baed7b39d405ad96625f94e50734b5. Core first84/full sections are identical to adaptive_panel.json.

## ruler4k (13 questions × 2 seeds)

| Historical task score | Strict EOS correct | Task at cap | EOS wrong | Unparsed | Output/iteration caps | Calls; canvases; pooled calls/canvas | Output tokens | A/D/H attention-layer calls | Terminations |
|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 0.923 (official 13-task macro 0.923) | 24/26 | 0 | 0 | 0 | 2/0 | 106; 26; 4.08 | 1043 | 1560/0/1620 | {"eos":24,"length":2} |

| Reference | Reference task score | Paired score delta | Overall geometric request ratio | Host | Paired cells | Historical warm whole-request mean (s) | CUDA first-encoder-end span mean (s) | Host geometric ratio | Host total-time ratio | Host calls ratio | Host amortized wall/call ratio |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| D_native | 0.923 | 0.000 | 1.093 | 149.165.151.254 | 13 | 1.070 | 0.679 | 1.059 | 1.077 | 1.093 | 0.986 |
| D_native | 0.923 | 0.000 | 1.093 | 149.165.159.64 | 13 | 0.949 | 0.528 | 1.128 | 1.124 | 1.044 | 1.077 |
| T_scope | 0.923 | 0.000 | 1.088 | 149.165.151.254 | 13 | 1.070 | 0.679 | 1.075 | 1.076 | 1.113 | 0.967 |
| T_scope | 0.923 | 0.000 | 1.088 | 149.165.159.64 | 13 | 0.949 | 0.528 | 1.102 | 1.096 | 1.044 | 1.049 |
| B_A8_matched | 0.923 | 0.000 | 1.023 | 149.165.151.254 | 13 | 1.070 | 0.679 | 0.933 | 0.880 | 0.819 | 1.074 |
| B_A8_matched | 0.923 | 0.000 | 1.023 | 149.165.159.64 | 13 | 0.949 | 0.528 | 1.122 | 1.113 | 1.044 | 1.065 |

The historical/T and historical/B ratios are derived algebraically from same-cell historical/native and reference/native ratios with the identical native warm denominator on each host. No joint historical/reference bootstrap interval is available in the redacted summaries, so none is inferred. The full calls/canvas histogram is in the CSV and JSON.

## aime26 (6 questions × 2 seeds)

| Historical task score | Strict EOS correct | Task at cap | EOS wrong | Unparsed | Output/iteration caps | Calls; canvases; pooled calls/canvas | Output tokens | A/D/H attention-layer calls | Terminations |
|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 0.667 | 8/12 | 0 | 0 | 4 | 4/0 | 3367; 262; 12.85 | 65932 | 15720/0/85290 | {"eos":8,"length":4} |

| Reference | Reference task score | Paired score delta | Overall geometric request ratio | Host | Paired cells | Historical warm whole-request mean (s) | CUDA first-encoder-end span mean (s) | Host geometric ratio | Host total-time ratio | Host calls ratio | Host amortized wall/call ratio |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| D_native | 0.417 | 0.250 | 0.978 | 149.165.151.254 | 6 | 51.386 | 51.240 | 0.948 | 0.944 | 0.919 | 1.028 |
| D_native | 0.417 | 0.250 | 0.978 | 149.165.159.64 | 6 | 34.561 | 34.419 | 1.008 | 1.029 | 0.961 | 1.071 |
| T_scope | 0.667 | 0.000 | 0.967 | 149.165.151.254 | 6 | 51.386 | 51.240 | 1.041 | 0.965 | 0.950 | 1.016 |
| T_scope | 0.667 | 0.000 | 0.967 | 149.165.159.64 | 6 | 34.561 | 34.419 | 0.898 | 0.927 | 0.864 | 1.072 |
| B_A8_matched | 0.583 | 0.083 | 0.925 | 149.165.151.254 | 6 | 51.386 | 51.240 | 0.862 | 0.925 | 0.915 | 1.011 |
| B_A8_matched | 0.583 | 0.083 | 0.925 | 149.165.159.64 | 6 | 34.561 | 34.419 | 0.992 | 0.983 | 0.917 | 1.071 |

The historical/T and historical/B ratios are derived algebraically from same-cell historical/native and reference/native ratios with the identical native warm denominator on each host. No joint historical/reference bootstrap interval is available in the redacted summaries, so none is inferred. The full calls/canvas histogram is in the CSV and JSON.

## longbench_v2 (6 questions × 2 seeds)

| Historical task score | Strict EOS correct | Task at cap | EOS wrong | Unparsed | Output/iteration caps | Calls; canvases; pooled calls/canvas | Output tokens | A/D/H attention-layer calls | Terminations |
|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 0.250 | 3/12 | 0 | 9 | 0 | 0/1 | 2838; 151; 18.79 | 37472 | 9060/0/76080 | {"eos":12} |

| Reference | Reference task score | Paired score delta | Overall geometric request ratio | Host | Paired cells | Historical warm whole-request mean (s) | CUDA first-encoder-end span mean (s) | Host geometric ratio | Host total-time ratio | Host calls ratio | Host amortized wall/call ratio |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| D_native | 0.500 | -0.250 | 1.067 | 149.165.151.254 | 6 | 29.632 | 27.310 | 0.980 | 0.985 | 1.068 | 0.922 |
| D_native | 0.500 | -0.250 | 1.067 | 149.165.159.64 | 6 | 51.967 | 49.382 | 1.163 | 1.198 | 1.290 | 0.929 |
| T_scope | 0.500 | -0.250 | 1.176 | 149.165.151.254 | 6 | 29.632 | 27.310 | 1.190 | 1.298 | 1.441 | 0.901 |
| T_scope | 0.500 | -0.250 | 1.176 | 149.165.159.64 | 6 | 51.967 | 49.382 | 1.161 | 1.143 | 1.203 | 0.951 |
| B_A8_matched | 0.333 | -0.083 | 1.067 | 149.165.151.254 | 6 | 29.632 | 27.310 | 0.924 | 0.818 | 0.809 | 1.012 |
| B_A8_matched | 0.333 | -0.083 | 1.067 | 149.165.159.64 | 6 | 51.967 | 49.382 | 1.232 | 1.196 | 1.190 | 1.005 |

The historical/T and historical/B ratios are derived algebraically from same-cell historical/native and reference/native ratios with the identical native warm denominator on each host. No joint historical/reference bootstrap interval is available in the redacted summaries, so none is inferred. The full calls/canvas histogram is in the CSV and JSON.

The historical label denotes a native-Q128/K64 mass-selector transplant. It is not numerically identical to the older vLLM G75/L30 physical grouping. No policy selection or cross-dataset pooled accuracy is inferred here.
