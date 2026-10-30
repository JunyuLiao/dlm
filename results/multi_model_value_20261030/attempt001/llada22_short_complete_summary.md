# LLaDA2.2: HumanEval and AIME complete; RULER pending

Ten complete arm panels,1270primaryrequests. Full three-model campaign remains incomplete.
N=denoising calls,C=canvases,T=output tokens; N/C below is the pooled ratio.
Instrumented execution; no clean latency or end-to-end speedup claim.

| Task | Arm | Correct | N/request | C/request | N/C | T/request | Consumer sparsity |
|---|---|---:|---:|---:|---:|---:|---:|
| humaneval | Native dense | 141/164 | 28.76 | 8.57 | 3.357 | 243.06 | 0.000% |
| humaneval | All-kept FA4 | 138/164 | 31.42 | 8.77 | 3.581 | 249.21 | 0.000% |
| humaneval | V31 mass + reuse4 | 83/164 | 36.76 | 7.84 | 4.692 | 218.90 | 49.978% |
| humaneval | V2 Gaussian32 + reuse4 | 10/164 | 14.21 | 3.07 | 4.634 | 62.91 | 49.216% |
| humaneval | V3a singleton + reuse4 | 103/164 | 42.32 | 8.67 | 4.881 | 245.29 | 49.990% |
| aime26 | Native dense | 36/90 | 1539.02 | 220.14 | 6.991 | 7018.64 | 0.000% |
| aime26 | All-kept FA4 | 29/90 | 1717.70 | 230.74 | 7.444 | 7356.26 | 0.000% |
| aime26 | V31 mass + reuse4 | 21/90 | 1322.93 | 147.46 | 8.972 | 4691.69 | 50.000% |
| aime26 | V2 Gaussian32 + reuse4 | 9/90 | 424.47 | 43.89 | 9.671 | 1374.07 | 75.208% |
| aime26 | V3a singleton + reuse4 | 25/90 | 1399.02 | 165.21 | 8.468 | 5259.06 | 49.997% |

V2 AIME reaches75.208%consumer sparsity despite51.015%development confirmation;
the frozen threshold is preserved and this is not a matched50%comparison.
Other sparse panels in this scope meet48-52%.

V3a versus V31 onhumaneval: accuracy delta+12.20pp,95%problem-clustered CI[3.05,21.34]; paired geometric N/C ratio0.9975,Nratio1.1003.
V3a versus V31 onaime26: accuracy delta+4.44pp,95%problem-clustered CI[-8.89,17.78]; paired geometric N/C ratio0.9365,Nratio1.1308.

V3a improves HumanEval over V31 with more calls and remains below native dense.
Its AIME lead has an interval including zero. No reliable faster-convergence
or general dense-quality-preservation claim is established.
Native/all-kept controls are separate because their numerical consumers alter trajectories.

Evidence: `analysis_llada22_short_complete001/`; exact-scope audit:
`verification/llada22_short_complete002/receipt.json`. Earlier analyses remain preserved.
