# v26 quick seven-arm panel (84 executions): M1 / M2 / M3 / B / native / fresh T

**Protocol and setup.**
- Frozen protocol `v26_seven_99cf56d6ef8412d7`; source `e9b943de`; 84/84 executions, 0 failures, all warm repeats accepted.
- Inputs by fixed rules chosen before any output: LB odd-K `…067c5456` (K%16 = 7) and KDIV-8 `…067c4480`; AIME `/14` and `/20`; RULER `niah_multikey_1` and `niah_single_1`. Seed 101 only; one warm repeat per cell. All are exposed dev inputs.
- Every arm of a question runs on the same GPU; hosts alternate by question.
- All method arms use the native bootstrap, grouped-Q, `aligned16_odd`, FP32-score output, GLOBAL-only, Q128/KV64 and P0 thresholds.

Cell-level data: [paired_quality_latency.csv](paired_quality_latency.csv), [seven_summary.json](seven_summary.json) and `seven_scored.*`. Redacted generations: `generation_records_v26_seven/`.

**Arms.**
- `M1_R1_A8`: redecides every call.
- `M2_pool_R1_A8`: M1 with pooled mu (see [m2_contract.md](m2_contract.md)).
- `M3_R3_A8`: current mainline.
- `M3_R3_A16`: lower-refresh candidate, frozen a priori; later true-QK anchors only at 17, 33, …
- `B_A8`: matched held; same anchors, no redecision.

## Summary per task (geometric means of per-question ratios vs native; one seed)

| Task | Arm | Correct | Calls | Canvases | W / native | Calls / native | W/call / native |
|---|---|---:|---:|---:|---:|---:|---:|
| LB (2) | native | 1/2 | 333 | 25 | 1 | 1 | 1 |
| | fresh T | 1/2 | 418 | 28 | 1.192 | 1.235 | 0.965 |
| | M1 R1 | 1/2 | 445 | 30 | 1.239 | 1.274 | 0.973 |
| | M2 pool R1 | 1/2 | 417 | 31 | 1.177 | 1.217 | 0.967 |
| | M3 R3 A8 | 1/2 | 330 | 25 | 0.990 | 1.037 | 0.955 |
| | M3 R3 A16 | 1/2 | 255 | 23 | **0.819** | 0.852 | 0.960 |
| | B A8 | 1/2 | 377 | 26 | 1.063 | 1.141 | **0.932** |
| AIME (2) | native | 0/2 | 631 | 44 | 1 | 1 | 1 |
| | fresh T | 1/2 | 826 | 57 | 1.671 | 1.656 | 1.009 |
| | M1 R1 | 1/2 | 596 | 41 | 0.910 | 0.891 | 1.021 |
| | M2 pool R1 | 1/2 | 602 | 45 | 1.087 | 1.060 | 1.026 |
| | M3 R3 A8 | 1/2 | 701 | 49 | 1.287 | 1.260 | 1.021 |
| | M3 R3 A16 | 1/2 | 593 | 45 | 1.049 | 1.026 | 1.022 |
| | B A8 | 1/2 | 741 | 56 | 1.508 | 1.480 | 1.019 |
| RULER (2) | all arms | 2/2 | 7–9 | 2 | 0.92–1.07 | — | 0.97–1.05 |

Per-question detail:
- LB `…4480`: every arm correct.
- LB `…5456`: every arm wrong (EOS).
- AIME `/14`: every arm wrong, and every arm hit the 8,192-token cap.
- AIME `/20`: native wrong, all six other arms correct.
- RULER: all correct, 3–6 calls per request.

## What this panel shows and does not show

1. **M2 now runs end to end.** On these 6 inputs it matches M1 on quality (4/6 each). Its per-call amortized cost on LB (0.967) sits between M1 (0.973) and M3 (0.955). This is the first measured M2 result, not evidence that M2 is equivalent or better.
2. **Per-call cost** (W/N) behaves as the direct profiles predicted.
   - On LB every sparse arm is 3–7% cheaper per call than native; B is cheapest per call (0.932) because it never redecides.
   - On AIME every method is ~2% more expensive per call than native, so short contexts have no attention saving.
3. **Whole-request time is dominated by how many calls each trajectory takes.** That varies strongly per question and per arm even at equal correctness: for example LB `…5456` took 154 calls under M3-A16 vs 246 native and 344 for M1, and AIME `/20` took 86 calls under M1 vs 275 for T. With one seed per question these call-count differences are mostly trajectory variance, not method effects. The M3-A16 LB headline (0.819) comes almost entirely from one question (0.615 on `…5456`).
4. **Quality** is 4/6 for every method vs 3/6 for native. The whole difference is AIME `/20`, a single seed — not a quality effect.
5. **A16 halves later anchors as predicted**, e.g. `…5456` 21 → 3 A and AIME `/14` 44 → 10 A. Its per-call LB cost (0.960) does not beat A8 (0.955) on these two inputs, because more D calls replace the saved A calls.

**Direct answer so far.**
- In long LB contexts, M3/B/M2/M1 each skip GLOBAL attention work and are 3–7% cheaper per call than native. B is cheapest per call, and aligned16 adds ≈2.6% on odd-K requests.
- On short contexts (AIME early canvases, RULER) there is no per-call saving.
- The total calls per request and the whole-request time are dominated by trajectory variance at this sample size.
- No incremental advantage of M3 over matched B or fresh T is established, and M2 does not beat M1.

The next stage (expansion under the v26 budget) needs more distinct questions and two seeds before any method-level claim.
