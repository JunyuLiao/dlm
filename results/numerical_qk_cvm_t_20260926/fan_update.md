# v14 update for Fan: CVM-T (cached value-risk margin + live temporal protection)

**Not M1, and not specified by the meeting.** CVM-T is a new, explicitly named simplification motivated by M1's cost failure (v13). The frozen v13 M1/M3 results stand under their original names.

## What was built
- **Anchor exporter.** A private v5 copy of Junyu's fresh value-direction kernel (ABI 5, own namespace/symbols; Junyu's binary untouched).
  - At genuine anchors (new canvas / A8) it exports each row's unweighted log value-risk `log rho[i,J]` from the same pass that already computes current QK and the value-direction risk.
  - It also enforces a mandatory set inside `decide()`: the current canvas and edge tiles are always kept.
  - Export on/off and protect-off are bit-identical to Junyu v4.
- **Ordinary steps.** One tiny Triton planner restores a skipped immutable-prefix tile iff `max_valid_rows(log rho + log s_t) >= log tau`, where s_t is the live causal T weight. The keep set is add-only inside an anchor epoch. The v11 preselected-support consumer then computes current QK/softmax/PV on the retained keys.
- **Exact T-only controller fast path**, applied to every T arm.
  - Unit tests show bit-identical weight sequences.
  - End to end, attempt-0 tokens of D_native and T_G_original equal the v13 cells in 12/12 and 12/12 cases.
- **Matched controls**, all GLOBAL-only with LOCAL native:
  - T_P: fresh T with the same mandatory set.
  - B8_P: the same v5 anchors and consumer, bitmap held with no live protection.

## Cost (complete denoising forward, real states, production mode)
- **Per GLOBAL call, late AIME state (prefix 6275, 5 layers per step):**
  - native 10.75 ms;
  - CVM ordinary 7.17 ms;
  - B8_P ordinary 4.97 ms;
  - anchor 11.2 ms.
  So the decision plus the consumer on retained support does beat dense attention once the prefix is ≳2K.
- **Complete forward, phase-weighted by native canvas lengths:**
  - CVM_T c = 1.020 (prefix 365), 1.009 (2157), 0.989 (6275).
  - The request-level AIME estimate is **1.003**.
  - The measured request-level per-call factor in the pilot is **1.003**.
  - GLOBAL attention is ≤8% of an AIME step, and every T-using arm pays ~1.2–1.8 ms/step outside the GLOBAL calls.
- **Regime diagnostic** (real LongBench-v2 16–18K prompts, first canvas, not scored): CVM_T c = 0.93–0.94; B8_P 0.90–0.92.

## Pilot (6 dev IDs × seeds 17/29, attempt 0 + 1 accepted warm; 120/120, 0 failures)

| arm | correct/12 | caps | calls | warm s (sum) | per-call vs D |
|---|---:|---:|---:|---:|---:|
| D_native | 6 | 6 | 3514 | 514.7 | 1.000 |
| T_G_original | 8 | 4 | 2874 | 428.4 | 1.018 |
| T_P | 5 | 5 | 3707 | 549.8 | 1.013 |
| B8_P | 9 | 3 | 3114 | 458.4 | 1.005 |
| CVM_T | 6 | 6 | 3955 | 580.9 | 1.003 |

- CVM_T/B8_P: geometric time 1.23 [1.08, 1.50], entirely from calls (×1.27; per call 0.998). Paired outcomes: 1 CVM-only success, 4 B8_P-only.
- CVM_T/T_P: 1.08 [0.96, 1.28]; paired 2 vs 1.
- CVM_T/D_native: 1.15 [0.98, 1.49]; paired 1 vs 1.
- Development data, small pilot: the intervals are descriptive, not tests.

## Reading
- Live protection did not improve the quality–latency operating point over the matched frozen bitmap (B8_P) or fresh T on this pilot.
- The per-forward saving the method can produce on AIME is ~0 at request level. It exists only at long prefixes.
- Most rows that later diverge from native do so during T's two-step bootstrap (s = 1). There the anchor is fresh, so stale margins are not the failure mechanism.
- Extension and repair were not run (gate decisions were recorded before the scores).
