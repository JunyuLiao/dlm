# RULER128K seed42: V31 versus V2 tau=0.021

Same104 prompts, 8 per13 tasks. Seed42 is the native panel seed; base31 and Gaussian32 projection seed1729 stay fixed. Official RULER gives partial credit.

| Method | Official score /100 | Full-credit items | N/problem | Median N | C/problem | N/C | Tokens/problem | GLOBAL kept tiles | GLOBAL sparsity | Overall sparsity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current_v31_control | 86.250 | 81/104 | 5.240 | 4.0 | 1.00 | 5.240 | 25.96 | 96,039,360 | 46.00% | 43.85% |
| v2_tau_0p021 | 85.769 | 81/104 | 5.635 | 4.0 | 1.00 | 5.635 | 25.90 | 95,030,781 | 50.33% | 47.98% |

V2 minus V31: -0.481 score points; 13-task-cluster bootstrap95% interval [-1.442, +0.000]. Full-credit binary McNemar p=1.0000.
Native call ratio 1.07523; absolute retained GLOBAL work ratio 0.98950.

| Task (8 items each) | V31 score | V2 score | Difference |
|---|---:|---:|---:|
| cwe | 70.000 | 63.750 | -6.250 |
| fwe | 100.000 | 100.000 | +0.000 |
| niah_multikey_1 | 100.000 | 100.000 | +0.000 |
| niah_multikey_2 | 100.000 | 100.000 | +0.000 |
| niah_multikey_3 | 87.500 | 87.500 | +0.000 |
| niah_multiquery | 100.000 | 100.000 | +0.000 |
| niah_multivalue | 93.750 | 93.750 | +0.000 |
| niah_single_1 | 100.000 | 100.000 | +0.000 |
| niah_single_2 | 100.000 | 100.000 | +0.000 |
| niah_single_3 | 100.000 | 100.000 | +0.000 |
| qa_1 | 50.000 | 50.000 | +0.000 |
| qa_2 | 25.000 | 25.000 | +0.000 |
| vt | 95.000 | 95.000 | +0.000 |

Both arms share identical source/protocol/runtime/model/input pins and use fresh generation. LOCAL stays dense; native FA4 executes attention; projected V is selector-only.
GLOBAL/overall work counts audited denoising attention tiles, not literal CUDA kernel launches. Additional selector discovery/selection and instrumented wall time are separate; clean timing stays paused.
These previously examined104 prompts are not a fresh held-out confirmation. Other campaigns remain paused.
