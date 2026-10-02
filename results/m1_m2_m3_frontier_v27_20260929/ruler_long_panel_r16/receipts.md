# R16 receipts: RULER 32K/64K accuracy check of the current pipeline

- Protocol `v27_ruler_long_r16_1bf5362572d4e0a8`, deploy `v27_p15_5bf19f1`, dllm only.
- 390/390 runs ok (78 per arm), no timed new graphs.
- All 13 RULER tasks at 32K and 64K × seeds 404/505/606 (39 cells per arm per length).

| arm | ruler32k correct (dense 34) | ruler64k correct (dense 33) | W 32K | W 64K |
|---|---:|---:|---:|---:|
| M3 + c0 (main) | 34 (+2/-2) | 33 (+0/-0) | 0.991 | 0.992 |
| M3 + c0 + q64c | 34 (+2/-2) | 33 (+0/-0) | 1.001 | 0.992 |
| fixed 88% (k12), rank-32 projected V | 33 (+1/-2) | 33 (+0/-0) | 0.985 | 0.994 |
| fixed 88% (k12), attention mass only | 33 (+1/-2) | 33 (+0/-0) | 0.998 | 0.979 |

- **Accuracy is preserved on RULER 32K/64K by every arm, including a fixed 88% GLOBAL sparsity.**
- **V term vs mass only: 33 vs 33 at both lengths** (`vterm.md`). On long-context RULER the V term makes no difference
  in our selector.
  - This differs from the group member's RULER 4K/8K finding at 70–75% sparsity. Their selector uses current QK and
    also sparsifies LOCAL layers.
  - Caveats: n is small (39 cells) and dense is near ceiling.
- No speed effect, as expected: about 5 decoder calls per request, and prefill dominates.
