# E12 HumanEval receipts (164 tasks x seeds 404-909 = 984 cells per arm; mpk + dlm2)

- 7,872/7,872 runs ok; every cell on one host (mpk 3,936, dlm2 3,936); 2 runs with timed new graphs (Wc).
- pass@1 scored on mpk by `scripts.v27_humaneval score` (unprivileged bwrap sandbox, preflight passed), 7,872/7,872 recorded.
- Effective method per arm (all 984 runs agree): main `threshold_shift=minus_ln2, carry_first`; gate arm adds
  `min_route_keys=2048` (204,035 of 238,755 GLOBAL calls ran native dense, 85%); V-term arms `risk_topk=k30, carry_first`
  with `proj_rank` 32/8/4, `mu_mode=pooled_compact` (M2c) or `risk_value=mass`.
- Per-host request-wall ratio vs dense: main 1.024 (mpk) / 1.022 (dlm2); gate 1.002 / 1.017; rank-32 at 70% 1.096 / 1.092.

Accuracy (exact McNemar on discordant cells):

| arm | correct (dense 947) | vs dense (p) | vs rank 32 at 70% (p) |
|---|---:|---|---|
| M3 + c0 (−ln2) | 957 | +31/−21 (0.21) | – |
| M3 + c0 + 2K gate | 948 | +4/−3 (1.00) | – |
| rank 32, 70% | 956 | +33/−24 (0.29) | – |
| rank 8, 70% | 963 | +31/−15 (0.026; not significant after correcting for 7 comparisons) | +26/−19 (0.37) |
| rank 4, 70% | 957 | +32/−22 (0.22) | +21/−20 (1.00) |
| M2c tile mean, 70% | 948 | +26/−25 (1.00) | +18/−26 (0.29) |
| mass only, 70% | 954 | +31/−24 (0.42) | +23/−25 (0.89) |
