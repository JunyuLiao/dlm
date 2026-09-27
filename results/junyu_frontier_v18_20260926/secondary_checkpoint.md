# CP3 secondary panel: redacted closeout

Both frozen CP3 stages closed with matching `rc=0` supervisor markers, one closed
`worker_end` interval per host/stage, and no failed run rows. The two hosts hold
byte-identical copies of each secondary ledger and calibrated policy file.
[`secondary_closed_receipt.json`](secondary_closed_receipt.json) records the
protocol IDs, ledger/scorer hashes, paired-key checks, host assignments and
resource arithmetic. It contains no prompts, gold or output tokens.

| Stage | Planned / recorded | First / warm | Paired question–seed blocks | Host split |
|---|---:|---:|---:|---:|
| RULER 70% | 936 / 936 | 780 / 156 | 390 | 468 / 468 |
| Allocation controls | 156 / 156 | 156 / 0 | 78 | 62 mpk / 94 dllm |

The 70% arms each scored 390 first outputs and accepted 78/78 strict warm
repeats. T70: mean score 0.899829, 348 correct, 5.603 decoder calls per first
output, whole PV support skipped 69.743%. U70: 0.890171, 341 correct, 7.174
calls, 69.997% skipped. The paired T70−U70 score difference is 0.009658
(95% question-cluster interval [−0.000940, 0.024359]); its same-stage warm
request-wall geometric ratio is 0.937427 [0.902822, 0.972827].

The allocation controls each scored 78 first outputs and had no warm repeats.
T60_shuffled: mean score 0.880342, 68 correct, 4.769 calls per first output,
whole PV support skipped 59.743%. T60_uniform: 0.884615, 69 correct, 4.756
calls, 60.673% skipped. Their paired score difference (shuffled−uniform) is
−0.004273 [−0.008546, 0]. No allocation warm latency is reported. Comparisons
of 70% arms with primary T60/U60 are cross-stage and marked descriptive in the
frozen summary because the runs were separated in time. PV support sparsity is
not measured QK savings.

Four density calibration points per group produced 208 core70 and 130 control
generation attempts, all with closed worker intervals. CP3 used 1,430 attempts;
the campaign count is 5,793/7,000 (mpk 3,281/3,500; dllm 2,512/3,500).
Conservatively charged GPU-process time, including the pre-CP3 baseline exactly
once and max(worker duration, worker wall interval), is 44,228.464 s on mpk
and 39,285.493 s on dllm; each is below 86,400 s. The combined 83,513.957 s
is below the 172,800 s cap. Worker wrapper durations are not added again.

The redacted artifacts are
[`ruler_secondary70_redacted_summary.json`](ruler_secondary70_redacted_summary.json),
[`ruler_secondary_allocation_redacted_summary.json`](ruler_secondary_allocation_redacted_summary.json),
and their corresponding scorer locks. Remote and local summary SHA-256 values
match exactly: `0c9828ec8ebcfeea2b07b8d053c823051ab929f91b644eda19712cfc15ef01e5`
and `b56b864dc37a38dfc883da1c0e3b18383adc26d162e96beb45e7045840b8913c`.
