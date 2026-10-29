# HumanEval technical reproducibility — diagnostic, unqualified

23 completed engines, 520 officially scored/rescored records. Cause unresolved.
No full K256/K128 rerun or reliable method comparison is released.

| Type | Left | Right | Unique questions / pairs | Output differences | N differences | Score differences | Max abs N delta |
|---|---|---|---:|---:|---:|---:|---:|
| same_method_technical_repeat | dense_on_audit_plain001 | dense_on_audit_plain002 | 6 / 12 | 12 | 10 | 0 | 78 |
| audit_clean_equivalence | dense_on_audit_plain001 | dense_on_clean_plain001 | 6 / 12 | 12 | 10 | 0 | 78 |
| audit_clean_equivalence | dense_on_audit_plain001 | dense_on_clean_plain002 | 6 / 12 | 12 | 12 | 0 | 124 |
| audit_clean_equivalence | dense_on_audit_plain002 | dense_on_clean_plain001 | 6 / 12 | 0 | 0 | 0 | 0 |
| audit_clean_equivalence | dense_on_audit_plain002 | dense_on_clean_plain002 | 6 / 12 | 12 | 12 | 0 | 90 |
| audit_clean_equivalence | dense_on_audit_trace001 | dense_on_clean_trace001 | 6 / 12 | 12 | 12 | 2 | 234 |
| same_method_technical_repeat | dense_on_clean_plain001 | dense_on_clean_plain002 | 6 / 12 | 12 | 12 | 0 | 90 |
| same_method_technical_repeat | dense_on_cublas_audit001 | dense_on_cublas_audit002 | 6 / 12 | 12 | 12 | 2 | 196 |
| audit_clean_equivalence | dense_on_cublas_audit001 | dense_on_cublas_clean001 | 6 / 12 | 12 | 10 | 0 | 163 |
| audit_clean_equivalence | dense_on_cublas_audit001 | dense_on_cublas_clean002 | 6 / 12 | 12 | 12 | 0 | 124 |
| audit_clean_equivalence | dense_on_cublas_audit002 | dense_on_cublas_clean001 | 6 / 12 | 12 | 12 | 2 | 33 |
| audit_clean_equivalence | dense_on_cublas_audit002 | dense_on_cublas_clean002 | 6 / 12 | 12 | 8 | 2 | 72 |
| same_method_technical_repeat | dense_on_cublas_clean001 | dense_on_cublas_clean002 | 6 / 12 | 12 | 12 | 0 | 43 |
| same_method_technical_repeat | dense_on_identity001 | dense_on_identity002 | 6 / 12 | 12 | 12 | 2 | 89 |
| same_method_technical_repeat | dense_on_live001 | dense_on_live002 | 6 / 36 | 36 | 30 | 6 | 191 |
| same_method_technical_repeat | dense_on_lt_small001 | dense_on_lt_small002 | 6 / 12 | 12 | 12 | 2 | 309 |
| same_method_technical_repeat | dense_on_observer001 | dense_on_observer002 | 6 / 18 | 18 | 18 | 3 | 65 |
| same_method_technical_repeat | dense_pw_on_clean001 | dense_pw_on_clean002 | 6 / 12 | 12 | 10 | 0 | 93 |

Within-engine interventions (not latency measurements):

| Engine | Unique question groups | Groups with any output/N/C/T change |
|---|---:|---:|
| dense_on_live001 | 6 | 0 |
| dense_on_live002 | 6 | 0 |
| dense_on_observer001 | 6 | 0 |
| dense_on_observer002 | 6 | 0 |

All source/runtime/consumer contrasts remain in summary.json. Different methods and consumers are not required to agree.
Repeated requests do not increase the unique-question count. No seed-robust superiority or reliable method ranking is inferred.
