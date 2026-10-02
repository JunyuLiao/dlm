# E11 receipts: V-term preview at 95% sparsity (LongBench-v2 64K, 24 items x seeds 404/505 = 48 cells per arm)

A preview (48 cells), not a confirmation panel.
- 192/192 runs ok; one host per cell; no timed new graphs.
- Effective method (all 48 runs per arm agree): risk_topk=k5 and carry_first=true in every method arm;
  projected-V arm mu_mode=exact; M2c arm mu_mode=pooled_compact; mass arm risk_value=mass.

| comparison | request W [CI] | accuracy discordant (p) |
|---|---|---|
| mass / rank-32 projected V | 1.004 [0.916, 1.099] | +5 / −3 (0.73) |
| M2c tile mean / rank-32 projected V | 1.020 [0.935, 1.120] | +5 / −3 (0.73) |
| rank 32 vs dense | — | +4 / −3 (1.00) |
| mass vs dense | — | +6 / −3 (0.51) |
| M2c vs dense | — | +6 / −3 (0.51) |
