# Update for Fan (v13): frozen GLOBAL-only M1/M3, all 30 AIME26 x seeds 17/29

**What was run.** Five frozen arms on all 30 AIME26 questions x generation seeds 17 and 29 (user-authorized): D_native,
T_G (fresh Junyu T on GLOBAL), G1 (M1), G3 (M3, R=2), B8_G (frozen anchor bitmap).
- Every arm: LOCAL layers native; GLOBAL layers [5, 11, 17, 23, 29]; native adaptive; thinking ON; 8192 cap.
- Each cell got one first output (quality) and one warm timing repeat.
- 600/600 executions, 0 failures. All 300 warm repeats reproduced their first output exactly (tokens, per-canvas
  calls, termination) with no new compilation.
- Protocol frozen and pushed before inference. No tuning. v12's seed-42 results were not pooled.

**(1) Was the new seed actually used?** Yes. The seed is part of every cell ID, is threaded into the native request,
and the driver asserts it in every receipt. A seed-42 check through the new driver reproduced v12's G3 /2 token hash
exactly. The old driver's hard-coded 42 is fixed (9 CPU tests).

**(2) Does G3 still save request time away from the original examples? No; the v12 signal reversed.**
- G3/T_G: geometric time ratio **1.160**, 95% question-cluster CI [1.071, 1.259]; summed 1.127 [1.041, 1.227].
  The same holds for both seeds (1.163 / 1.157), the 6 development questions (1.330) and the 24 reserved
  questions (1.121). Leave-one-question-out range: 1.136-1.182.
- G3/D_native: 1.091 [1.011, 1.184]. G3/B8_G: 1.150 [1.039, 1.283].
- On the six development questions, v12 seed 42 had G3 0.76x T_G; seeds 17/29 give 1.33x. The earlier advantage
  was seed-specific trajectory luck.

**(3) Accuracy and caps** (attempt 0, correct out of 60 question-seed outcomes; caps out of 60):

| | T_G | G3 | B8_G | D_native | G1 |
|---|---:|---:|---:|---:|---:|
| correct | 34 | 34 | 34 | 32 | 30 |
| caps | 23 | 25 | 20 | 24 | 26 |

All paired quality intervals include 0 (G3 vs T_G: +0.000 [-0.083, 0.083]). This is not evidence of equivalence;
no margin was predefined.

**(4) Work count, per call, or both?** Almost entirely work count.
- G3/T_G = 1.124x the decoder calls x 1.003x the time per call. The extra calls come from more canvases (x1.042,
  i.e. longer outputs) and more steps per canvas (x1.079).
- Per call, all arms are within about 5% of native (149-156 ms per call, request-amortized).
- B8_G has the fewest calls: summed B8/D_native 0.927 [0.856, 0.997]; geometric 0.949 [0.851, 1.052].
- T_G/D_native: 0.941 [0.868, 1.019].
- These are trajectory effects, not attention arithmetic savings.

**(5) M1 vs the simple bitmap.** On these 30 x 2 outcomes the simple frozen anchor bitmap (B8_G) is at least as
accurate as M1 (34 vs 30) and faster (G1/B8 1.109 [1.016, 1.208]). Intermediate M1 re-selection (G3) did not beat
B8 either (34 vs 34, time 1.150). Refresh frequency and current-V information are still not separated. The data show
**no practical benefit of numerical-history re-selection over the frozen anchor bitmap in this regime.** They do not
prove numerical history can never help.

**Missing / next decision.** The GLOBAL-only M1/M3 line has no E2E benefit on this model and workload.
- Candidates for follow-up: (a) whether B8's small fewer-calls effect vs native (summed 0.93, geometric CI including
  1) survives more seeds or a same-mask native control; (b) a different workload/model where attention is a larger
  share of the step (longer contexts), with LLaDA2.1 weights to be authorized.
- Kernel work on this path is not justified by these results. Grouping (Haowei) is not integrated; M2 is not attempted.
