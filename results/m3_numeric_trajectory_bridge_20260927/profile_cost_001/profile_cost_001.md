# v21 direct profile cost summary

CPU summary of direct complete-call timing. No promotion decision.

Requested states: {'reached': 17, 'missing': 1}; unique measured cells: 22. Each canvas/boundary/length appears once.

| Dataset | ID | Canvas | Host | Boundary | Length | Reached | Native ms | Legacy/native | Layout/legacy | Numeric/legacy | Combined/legacy |
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| aime26 | aime26/2 | 0 | mpk | model_forward | N4 | 4 | 501.8 | 1.036 | 0.9979 | 1.001 | 1.001 |
| aime26 | aime26/2 | 0 | mpk | model_forward | N16 | 11 | 1880 | 1.025 | 0.995 | 0.9986 | 0.9971 |
| aime26 | aime26/2 | 0 | mpk | denoising_step | N4 | 4 | 665.5 | 1.032 | 1.001 | 1.001 | 1.001 |
| aime26 | aime26/2 | 0 | mpk | denoising_step | N16 | 11 | 1956 | 1.029 | 0.9898 | 0.9999 | 0.9905 |
| longbench_v2 | longbench_v2/66f9625fbb02136c067c5456 | 0 | mpk | model_forward | N4 | 4 | 744.3 | 1.016 | 0.998 | 0.9965 | 0.9938 |
| longbench_v2 | longbench_v2/66f9625fbb02136c067c5456 | 0 | mpk | model_forward | N16 | 14 | 2837 | 0.9853 | 0.9994 | 0.9987 | 0.9994 |
| longbench_v2 | longbench_v2/66f9625fbb02136c067c5456 | 0 | mpk | denoising_step | N4 | 4 | 765.8 | 1.018 | 0.9978 | 0.994 | 0.9936 |
| longbench_v2 | longbench_v2/66f9625fbb02136c067c5456 | 0 | mpk | denoising_step | N16 | 14 | 2920 | 0.988 | 0.9995 | 0.9986 | 0.9979 |
| ruler4k | ruler4k/ruler_4096_cwe_p0046 | 0 | mpk | model_forward | N4 | 4 | 685.3 | 1.03 | 0.9989 | 1.002 | 1.001 |
| ruler4k | ruler4k/ruler_4096_cwe_p0046 | 0 | mpk | model_forward | N16 | 7 | 1217 | 1.024 | 0.9998 | 1 | 0.9998 |
| ruler4k | ruler4k/ruler_4096_cwe_p0046 | 0 | mpk | denoising_step | N4 | 4 | 708.2 | 1.029 | 1 | 1.003 | 1.003 |
| ruler4k | ruler4k/ruler_4096_cwe_p0046 | 0 | mpk | denoising_step | N16 | 7 | 1256 | 1.025 | 0.9986 | 0.9993 | 0.9994 |
| aime26 | aime26/8 | 1 | dllm | model_forward | N4 | 4 | 464 | 1.032 | 1.002 | 0.9993 | 0.9997 |
| aime26 | aime26/8 | 1 | dllm | model_forward | N16 | 12 | 1911 | 1.019 | 0.9998 | 1 | 0.9998 |
| aime26 | aime26/8 | 1 | dllm | denoising_step | N4 | 4 | 630.1 | 1.026 | 1.001 | 0.9983 | 0.998 |
| aime26 | aime26/8 | 1 | dllm | denoising_step | N16 | 12 | 1950 | 1.023 | 1 | 0.9987 | 0.9987 |
| longbench_v2 | longbench_v2/66f2aac2821e116aacb2a9de | 1 | dllm | model_forward | N4 | 4 | 708 | 0.9462 | 1.002 | 1.005 | 1.003 |
| longbench_v2 | longbench_v2/66f2aac2821e116aacb2a9de | 1 | dllm | model_forward | N16 | 15 | 2927 | 0.9456 | 0.9983 | 1.001 | 1 |
| longbench_v2 | longbench_v2/66f2aac2821e116aacb2a9de | 1 | dllm | denoising_step | N4 | 4 | 731.5 | 0.9547 | 1.001 | 1.001 | 1.001 |
| longbench_v2 | longbench_v2/66f2aac2821e116aacb2a9de | 1 | dllm | denoising_step | N16 | 15 | 3015 | 0.9499 | 1 | 1 | 0.9994 |
| ruler4k | ruler4k/ruler_4096_fwe_p0096 | 0 | dllm | model_forward | N4 | 3 | 445.9 | 1.013 | 1 | 1.004 | 1.002 |
| ruler4k | ruler4k/ruler_4096_fwe_p0096 | 0 | dllm | denoising_step | N4 | 3 | 461.8 | 1.017 | 0.9994 | 1.004 | 1.002 |

Direct epoch spans, wall spans, bracket drift, physical pair denominators, copy traces, accepted peak memory, digest parity, every requested state, and source identities are in `profile_cost_001.json`. The CSV has one row per unique cell and arm.
