# P16 receipts: held-out check of the 64K accuracy gap (2026-10-02)

- Spec `specs/v27_lb64_item_check_p16.json`, protocol `v27_lb64_item_check_p16_4d31029328774e37`, deploy
  `v27_p16_959ae20`, run dir `v27_p16_001`.
- 288/288 runs ok on dllm, mpk and dlm2, one host per cell, substrate piecewise_v5. GPU time about 4,400 s
  (sum of the three worker spans).

**Question.** Over E13–E15 the main arm had 228 vs dense 238 correct of 432 at 64K (item-clustered p 0.29). Is that
gap a real item-specific failure, or selection noise?

**Design** (frozen before generation; discover on E13–E15, test on fresh seeds):
- 6 items chosen from E13–E15 only: the 3 with the most negative and the 3 with the most positive main-minus-dense
  correct count.
- 12 never-used seeds (2222–3333), so 36 cells per item group.
- Arms:
  - dense FA4 (`D_fa4_allkept`);
  - main (M3 R6 DP −ln2 + `carry_first`);
  - main + output protection (generated-token key tiles never skipped);
  - main at the more conservative −2ln2 threshold.

**Path receipts (per-cell counters, all three hosts).**
- The protection arm protected every routed call: `protected_routes` equals `dp_routes` (16,950 of 16,950). The
  other arms have 0 protected routes.
- The −2ln2 arm ran `threshold_shift = minus_2ln2`; main and protection ran `minus_ln2`.
- No arm fell back to native or fresh-fused calls (`layer_native_calls`, `fresh_fused_calls` = 0).

## All 72 cells per arm vs dense

| arm | accuracy (+/-) | W [CI] | S | per step S/N [CI] | N |
|---|---|---|---:|---|---:|
| dense | 44 / 72 | 1 | 1 | 1 | 1 |
| main | 45 (+12/−11) | **0.857 [0.800, 0.923]** | 0.791 | **0.808 [0.788, 0.826]** | 0.979 |
| main + protection | 38 (+8/−14) | 0.893 [0.847, 0.942] | 0.838 | 0.809 [0.791, 0.824] | 1.035 |
| main −2ln2 | 44 (+12/−12) | 0.888 [0.824, 0.957] | 0.827 | 0.852 [0.831, 0.868] | 0.971 |

## Per item group: discovery panels vs held-out seeds (main vs dense)

| group | E13–E15 (discovery, 54 cells) | P16 (fresh seeds, 36 cells) |
|---|---|---|
| 3 worst items | main 17 vs dense 28 (−20 pp) | main 14 vs dense 17 (+5/−8, McNemar p 0.58; −8 pp) |
| 3 best items | main 45 vs dense 41 | main 31 vs dense 27 (+7/−3, p 0.34) |

On the 3 worst items, P16 protection gives 11 (+4/−10, p 0.18) and −2ln2 gives 14 (+6/−9, p 0.61).

## Reading

- **Mostly selection noise.** The −20 pp gap on the selected items shrinks to −8 pp on fresh seeds, which is not
  significant. Over all six items main equals dense (45 vs 44).
  - A small item-specific effect cannot be excluded at this sample size.
  - There is no evidence of a systematic 64K accuracy loss.
- **Output protection does not help.** It is the only arm below dense (38 vs 44), it takes more steps (N 1.035) and
  its per-step cost equals main's. Not adopted.
- **−2ln2 adds no accuracy and costs speed.** It reaches 44 vs main's 45 and is 5% slower per step (0.852 vs 0.808).
  Not adopted.
