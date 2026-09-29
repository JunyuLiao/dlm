# v27 long-context RULER panel (32K, 64K): full results

**Panel.** Protocol `v27_long_ruler_42883cd36fcb51c5` (`../specs/v27_long_ruler.json`), frozen before generation: 13 RULER tasks × 2 seeds per length, 11 arms, first attempt plus one warm repeat, on 2 H100 hosts.
- 1,144 of 1,144 executions succeeded.
- **Where the data comes from.** The dllm half of ruler32k is in run 006. Everything else is in run 008. Between the two runs only memory changed: the O(n²) prefill mask mapping is now elided. The D_c64 tokens are identical between them (`../selector/README.md` §3).
- **GPU process time.**

  | host | ruler32k | ruler64k |
  |---|---:|---:|
  | dllm | (run 006) | 2,532 s |
  | mpk | 1,163 s | 2,586 s |

Files:
- `summary.md` / `summary.csv`: paired geometric means with a question-clustered bootstrap. S is the decode span, which excludes the initial prefill. N is the number of decoder calls. S/N is amortized per call.
- `paired_quality.csv`: per-cell score differences vs D_c64 and D_native.
- `cells.csv`: scored cells, with no text.

## Findings (vs D_c64, the strongest dense)

1. **No end-to-end effect on RULER.** Request wall is 0.96–1.05 for every arm at both lengths. The initial prefill dominates (about 4 s at 32K, about 9 s at 64K), and a request needs only about 5 decoder calls.
2. **Per call (S/N), the A64 shared/fused variants are the only arms below 1.**

   | arm | 32K | 64K |
   |---|---|---|
   | M3 R6/A64 shared+fused | 0.991 [0.982, 1.002] | 0.965 [0.945, 0.988] |
   | M3 R3/A64 shared+fused | 0.996 | 0.973 [0.956, 0.992] |

   Other arms:
   - Plain M1/M2c/M3 (A8, unfused observation): 1.10–1.14 at 32K, 1.14–1.22 at 64K.
   - T_scope: 1.19 / 1.35.
   - D_native: 1.24 / 1.46.
3. **Quality, from first-attempt RULER scores over 26 cells per length** (counts are cells worse / better than D_c64):
   - **Plain M1/M2c/M3 (A8) keep quality.** At 64K each is worse in 1–2 cells and better in none (mean −0.008 to −0.015). At 32K they are slightly better (+0.015 to +0.023).
   - **The A64 variants with one shared support across the 5 GLOBAL layers lose quality at 64K.** They are worse in 4–5 cells and better in none (mean −0.10 to −0.13).
   - **B/A64 fused, which has no sharing, does not lose quality** (2 worse, mean −0.012). The loss therefore points at cross-layer sharing, not at A64 or at fused observation.
   - **The two dense paths also disagree.** D_native vs D_c64 is 3 cells worse at 64K (mean −0.085). With about 5 short answers per request, numerical path changes alone move RULER scores by this much. Means without the paired counts are therefore not evidence either way.

## Consequence for the method

- **Per-call gain and quality pull apart.** The per-call gain comes from the shared-support A64 variants. The preserved quality comes from the unshared plain arms, whose per-call cost is dominated by the selector: each of the 5 GLOBAL layers scans every tile in order on each decision call (`../selector/README.md` §1).
- **Next variant to test.** Unshared A64/fused support, with a cheap decision call: either the pipelined exact selector for M3 R6, or M1-DP, the parallel dense-prefix risk, for per-call M1.
- **Where to measure end to end.** RULER cannot show an end-to-end effect. The long-generation LongBench-v2 32K/64K panel (`../specs/v27_long_lb.json`) is the end-to-end test.
