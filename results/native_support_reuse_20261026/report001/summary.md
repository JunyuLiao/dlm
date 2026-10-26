# Controlled reuse development report

All 528 frozen distinct requests completed native decoding and official scoring. There are six question clusters and two paired seeds; three initial seed cells fall back to A.

Reuse gate passes: 0/32. Finalists: 0. Clean online latency remains unqualified.

Ratios below are arithmetic equal-question local held/refresh and geometric paired complete N held/refresh. Pooled totals are separately labeled in CSV/JSON.

| Selector | Budget | Window | Local held/refresh | Complete N held/refresh | Repeated-benefit questions | Quality regressions | Advance |
|---|---:|---:|---:|---:|---:|---:|---|
| current_error | 5% | 2 | 1.0046 | 0.9953 | 0 | 0 | False |
| current_error | 5% | 4 | 0.9345 | 1.1505 | 1 | 0 | False |
| mass | 5% | 2 | 0.9838 | 0.9281 | 0 | 0 | False |
| mass | 5% | 4 | 0.9434 | 1.0350 | 0 | 0 | False |
| support_sham | 5% | 2 | 1.0253 | 0.9794 | 0 | 0 | False |
| support_sham | 5% | 4 | 0.9857 | 1.0541 | 0 | 0 | False |
| readout_margin | 5% | 2 | 0.9917 | 1.1279 | 1 | 6 | False |
| readout_margin | 5% | 4 | 1.0045 | 1.0179 | 1 | 0 | False |
| response_qpast | 5% | 2 | 1.0790 | 0.9451 | 0 | 0 | False |
| response_qpast | 5% | 4 | 0.9731 | 1.0054 | 0 | 0 | False |
| margin_weighted | 5% | 2 | 1.0448 | 0.9590 | 0 | 0 | False |
| margin_weighted | 5% | 4 | 0.9602 | 1.0282 | 1 | 0 | False |
| signed_high | 5% | 2 | 0.9844 | 0.9732 | 0 | 0 | False |
| signed_high | 5% | 4 | 1.0420 | 1.0424 | 0 | 0 | False |
| signed_low | 5% | 2 | 1.0472 | 0.9967 | 0 | 0 | False |
| signed_low | 5% | 4 | 1.0618 | 1.0068 | 1 | 0 | False |
| current_error | 10% | 2 | 1.0134 | 0.9680 | 0 | 0 | False |
| current_error | 10% | 4 | 0.9149 | 0.9789 | 0 | 0 | False |
| mass | 10% | 2 | 0.9798 | 0.8998 | 0 | 0 | False |
| mass | 10% | 4 | 0.9127 | 1.0616 | 1 | 0 | False |
| support_sham | 10% | 2 | 1.0372 | 1.0153 | 0 | 0 | False |
| support_sham | 10% | 4 | 1.0491 | 0.9690 | 0 | 0 | False |
| readout_margin | 10% | 2 | 0.9697 | 1.0189 | 0 | 0 | False |
| readout_margin | 10% | 4 | 1.0386 | 1.0386 | 0 | 0 | False |
| response_qpast | 10% | 2 | 1.0506 | 1.0608 | 0 | 0 | False |
| response_qpast | 10% | 4 | 0.9284 | 0.9900 | 0 | 0 | False |
| margin_weighted | 10% | 2 | 0.9707 | 0.9273 | 1 | 0 | False |
| margin_weighted | 10% | 4 | 0.9884 | 0.9909 | 1 | 0 | False |
| signed_high | 10% | 2 | 1.0014 | 1.0775 | 0 | 0 | False |
| signed_high | 10% | 4 | 1.0532 | 1.0718 | 0 | 0 | False |
| signed_low | 10% | 2 | 0.9423 | 0.9977 | 0 | 0 | False |
| signed_low | 10% | 4 | 0.9880 | 0.9531 | 0 | 0 | False |

All configurations, losses, aliases, caps, missing final responses and infeasible fallbacks are retained. Absence of observed quality loss does not establish noninferiority. Diagnostic implementations and precomputed initial supports cannot establish clean online latency.

Resources: 657 complete requests, 115281 explicit model forwards, 1.527893 allocated H100-worker hours including startup, warmups, qualification and cleanup. Copied controls count once.
