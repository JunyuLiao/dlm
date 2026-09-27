# v21 numerical diagnostic summary

Diagnostic only. No precision promotion decision is made here.

Reported states: 18/18; unreported: 0; stage errors: 0; failure receipts: 0.

| Dataset | ID | Canvas | Call | Host | Status | Operator new/old | Frozen M3 new/old | D logits new/old | M3 logits new/old |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| aime26 | aime26/2 | 0 | 0 | mpk | diagnosed | 0.5305 | 0.532 | 1.101 | 1.123 |
| aime26 | aime26/2 | 0 | 1 | mpk | diagnosed | 0.5948 | 0.5965 | 1.027 | 0.9949 |
| aime26 | aime26/2 | 0 | 3 | mpk | diagnosed | 0.4765 | 0.4773 | 0.9312 | 1.013 |
| aime26 | aime26/8 | 1 | 0 | dllm | diagnosed | 0.4965 | 0.5013 | 0.8612 | 0.9573 |
| aime26 | aime26/8 | 1 | 3 | dllm | diagnosed | 0.4512 | 0.4496 | 0.8209 | 0.8782 |
| aime26 | aime26/8 | 1 | 6 | dllm | diagnosed | 0.4154 | 0.4164 | 0.7471 | 0.949 |
| longbench_v2 | longbench_v2/66f9625fbb02136c067c5456 | 0 | 0 | mpk | diagnosed | 0.4714 | 0.4701 | 0.9067 | 1.026 |
| longbench_v2 | longbench_v2/66f9625fbb02136c067c5456 | 0 | 1 | mpk | diagnosed | 0.5103 | 0.518 | 1.048 | 0.9753 |
| longbench_v2 | longbench_v2/66f9625fbb02136c067c5456 | 0 | 3 | mpk | diagnosed | 0.4282 | 0.4371 | 1.162 | 1.084 |
| longbench_v2 | longbench_v2/66f2aac2821e116aacb2a9de | 1 | 0 | dllm | diagnosed | 0.5069 | 0.464 | 0.9754 | 1.093 |
| longbench_v2 | longbench_v2/66f2aac2821e116aacb2a9de | 1 | 3 | dllm | diagnosed | 0.4296 | 0.4405 | 0.8092 | 1.034 |
| longbench_v2 | longbench_v2/66f2aac2821e116aacb2a9de | 1 | 6 | dllm | diagnosed | 0.3869 | 0.3906 | 0.8364 | 1.056 |
| ruler4k | ruler4k/ruler_4096_cwe_p0046 | 0 | 0 | mpk | diagnosed | 0.4118 | 0.4239 | 1.018 | 0.9552 |
| ruler4k | ruler4k/ruler_4096_cwe_p0046 | 0 | 1 | mpk | diagnosed | 0.419 | 0.4345 | 0.7571 | 0.9973 |
| ruler4k | ruler4k/ruler_4096_cwe_p0046 | 0 | 3 | mpk | diagnosed | 0.4028 | 0.4177 | 1.342 | 1.269 |
| ruler4k | ruler4k/ruler_4096_fwe_p0096 | 0 | 0 | dllm | diagnosed | 0.3868 | 0.3887 | 0.94 | 0.9748 |
| ruler4k | ruler4k/ruler_4096_fwe_p0096 | 0 | 1 | dllm | diagnosed | 0.3976 | 0.3993 | 1.034 | 1.008 |
| ruler4k | ruler4k/ruler_4096_fwe_p0096 | 0 | 3 | dllm | missing | N/A | N/A | N/A | N/A |

Aggregate directions (new relative-L2 versus old):

- Operator full FP32: 17 improved, 0 worse, 0 equal, 1 unavailable; median ratio 0.4295606947829661.
- Frozen M3 support: 17 improved, 0 worse, 0 equal, 1 unavailable; median ratio 0.4405304459577428.

Per-state metrics, failures, layout parity, and source identities are in `numeric_diagnostic_summary.json`.
