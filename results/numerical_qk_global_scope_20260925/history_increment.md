# v12: does numerical history + current V earn its cost? (GLOBAL-only, native LOCAL)

Question: under the SAME scope (5 GLOBAL layers routed, 25 LOCAL native in every arm), does M1's historical-score
selection with current projected V beat (a) fresh Junyu T and (b) a simple frozen value-aware anchor bitmap (B8)?
It must do so on quality or on complete-request time, enough to pay its extra cost.

## Per-call and per-step cost (same captured states; `same_state_costs.*`)
Sum over the 5 GLOBAL layers, in ms (canvas 1 / canvas 6):
- native 1.70 / 3.67;
- T_G 2.20 / 3.96;
- G ordinary decision 4.23 / 6.61;
- G/B8 held 1.16-1.24 / 2.26-2.34;
- anchors 5.8-6.0 / 10.6-10.7 (G anchors also build summaries).

Realized GLOBAL-call phase mix from the panel receipts:

| arm | anchor | decision | held |
|---|---:|---:|---:|
| G1 | 16.2% | 100% | 0% |
| G3 | 16.2% | 51.8% | 48.2% |
| B8 | 16.4% | 16.4% | 83.6% |

The anchor share is 16%, not the theoretical 12.5%, because short canvases add anchors. GLOBAL summaries peak at
346 MB resident; B8 builds none.

Headroom: the five native GLOBAL calls are 3.7 ms of a ~132 ms step (<= 3%). Every bound arm, fresh T_G included,
pays ~6-7 ms/step of fixed overhead vs D_native (T bookkeeping, binding hook, dispatcher). **Per step, no GLOBAL-only
method can beat native by more than ~3% even with free attention.**

## Complete requests (attempt 0 = quality; four-question panel warm repeats token-identical)
| | D_native | T_G | G1_split | G3_split | B8_G |
|---|---|---|---|---|---|
| correct, panel /2,/8,/14,/20 | 3/4 | 3/4 | 3/4 | 3/4 | 3/4 |
| correct, + /23,/30 (6 total) | 3/6 | **4/6** | 3/6 | **4/6** | 3/6 |
| panel warm total s / calls | 163.0 / 1086 | 168.0 / 1100 | 148.7 / 949 | 126.9 / 819 | 146.3 / 953 |
| amortized ms/call | 148-155 | 150-158 | 155-160 | 153-163 | 151-159 |
| achieved GLOBAL skip (/8 audit twin) | - | 22.5% | 21.2% | 19.8% | 31.3% |
All arms cap on /14. /30 fails everywhere; G3 ends at EOS there but is wrong.

Named ratios (warm time, panel 4 questions):

| ratio | geometric mean of per-question ratios | summed times |
|---|---:|---:|
| G1/T_G | 0.863 | 0.885 |
| G1/B8 | 1.103 | 1.016 |
| G1/G3 | 1.341 | 1.171 |
| G3/T_G | 0.643 | 0.756 |
| B8/T_G | 0.782 | 0.871 |
| G1/D_native | 0.886 | 0.912 |
| G3/D_native | 0.661 | 0.779 |
| B8/D_native | 0.803 | 0.897 |
| T_G/D_native | 1.027 | 1.030 |

## Answer
1. **Numerical history vs the simple frozen bitmap: no demonstrated increment.** G1 costs more per call than B8
   (decision every step vs held 84% of calls) and gives the same quality (3/4; 3/6). Its request time is not better
   (geometric 1.10x B8). B8 skips MORE tiles (31% vs 21%) with no quality loss on these questions.
2. **vs fresh T_G:** equal panel quality; G1 is 0.86x T_G's time only through fewer denoising calls (949 vs 1100).
   Per call G1 is ~2-4% MORE expensive. Across 6 questions T_G has one more correct answer than G1 (/23).
3. **G3 (the meeting's M3, R=2)** is the fastest arm (0.78x native summed, 0.76x T_G) and ties T_G for most correct
   (4/6). This comes from much shorter trajectories on /2 and /20 (84 and 94 calls vs native's 146 and 370), NOT
   from cheaper steps.
4. **Trajectory length dominates everything, and it varies chaotically.** On /20 the arms take 87-370 calls, all
   correct. With 4-6 questions and one seed these differences cannot be attributed to the selection rule.

The distinct outcomes, stated separately:
- **insufficient per-step headroom:** YES (GLOBAL attention <= 3% of a step);
- **harmful trajectory changes:** NO evidence (quality parity or better);
- **numerical interchangeability:** the Hopper consumer remains a numerically distinct variant (v11), and anchors use
  the Triton PV;
- **incremental numerical-information value** over the frozen bitmap: NOT demonstrated.

## Not done / not claimed
- No fused GLOBAL selector (CP3 skipped: no complete-path headroom).
- No seeds; no noninferiority; no novelty; no grouping integration.
- Only 2 extra authorized development questions existed (the spec allowed 4); the other 24 AIME26 problems are
  reserved for full-30 validation.
