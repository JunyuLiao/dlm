# E8 receipts and accuracy tests (AIME26, 30 problems x seeds 404-909 = 180 cells per arm)

Integrity: 1,440/1,440 runs ok; 180 cells, each on one host (dllm, mpk, dlm2: 480 runs each); 3 runs with timed
new graphs (excluded in the Wc column). Dense tokens are identical to E7's dense arm in 180/180 cells.

Effective-method fields from every run's receipt (all 180 runs of each arm agree):

| arm | V term | risk_topk | proj_rank | risk_value | mu_mode |
|---|---|---|---:|---|---|
| M3_R6_A64_fused_dp_async_topk70_fa4 | projected V, rank 32 | k30 | 32 | – | exact |
| ..._topk70_r16_fa4 | projected V, rank 16 | k30 | 16 | – | exact |
| ..._topk70_r8_fa4 | projected V, rank 8 | k30 | 8 | – | exact |
| ..._topk70_r4_fa4 | projected V, rank 4 | k30 | 4 | – | exact |
| M2c_R6_A64_fused_dp_async_topk70_fa4 | M2 tile-mean V | k30 | – | – | pooled_compact (54,105 pool builds) |
| ..._topk70_mass_fa4 | none (attention mass share) | k30 | – | mass | exact |
| SparseD_s70_skip1_fa4 | SparseD port, avg-pooled scores | keep 0.3 | – | – | – |

Accuracy, exact McNemar on discordant cells:

| arm | correct (dense 99) | vs dense | p | vs rank 32 | p |
|---|---:|---|---:|---|---:|
| projected V r32 | 87 | +14/−26 | 0.081 | – | – |
| projected V r16 | 86 | +10/−23 | 0.035 | +15/−16 | 1.000 |
| projected V r8 | 87 | +8/−20 | 0.036 | +16/−16 | 1.000 |
| projected V r4 | 94 | +18/−23 | 0.533 | +20/−13 | 0.296 |
| M2 tile-mean V | 92 | +13/−20 | 0.296 | +19/−14 | 0.487 |
| attention mass only | 91 | +15/−23 | 0.256 | +18/−14 | 0.597 |
| SparseD 70% (port) | 94 | +13/−18 | 0.473 | +26/−19 | 0.371 |
