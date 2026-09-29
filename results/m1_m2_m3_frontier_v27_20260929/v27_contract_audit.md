# v27 contract audit: effective A/R identity, R6, hold-only B16, test guards

As of commit `4226b96b5` (branch `research/m3-output-numerics-20260927`). CP0 was `aafe6172`.

## 1. A16 had two contradictory clock fields (fixed; report-only defect)

- **Defect.**
  - `v21.install()` rebinds `owner.cache.score_period = 16` for A16 arms.
  - The v20 counter closure still wrote its constant parent value `score_refresh_period = 8`.
  - v26 A16 receipts therefore carry `score_period: 16`, `v26_score_period: 16` and `score_refresh_period: 8` at the same time.
- **Why it was report-only.** The executed clock was A16. For example, AIME `/14` has anchors only at 17 and 33, and the 84-panel A counts drop exactly as the A16 clock predicts.
- **Fix: an authoritative `effective_method`.**
  - Every v21 counter set now carries `effective_method`, read from the **bound runtime**, never from the parent arm name.
  - Fields: `score_period`, `decision_interval` (`null` when hold-only), `hold_only`, `score_clock_origin`, `mu_mode`, `route_storage`, `scope`, `bootstrap_policy`, `threshold_shift`, the actual `log_thresholds`, output precision/layout, producer, selector and `parent_v20_arm`.
  - The v20 parent values are kept as `parent_v20_score_refresh_period` and `parent_v20_decision_interval`.
  - The top-level `score_refresh_period` and `decision_interval` now report the effective values.
- **Old raw receipts are not modified.** For the v26 84-panel, the A16 rows should be read with `v26_score_period` (16). Their `score_refresh_period: 8` is the parent-provenance value.

## 2. R6, A64 and hold-only B are now real configuration points

Old state: v20 arms were only R1, R2, R3 and B_R8. v21 had added only the A (score period) override.

v27 adds these optional v21 keys. When a key is absent the old fingerprint is unchanged, so the frozen v26 arms still reproduce.

| key | allowed | parent arm | meaning |
|---|---|---|---|
| `score_period` | 8 (default), 16, 64 | bootstrap mainline | A: true current-QK re-observation period (origin 1: 9/17/…, 17/33, none inside 48 calls) |
| `decision_interval` | 6 | `M3_R3_A8_current_output` | R: M1 redecision interval |
| `hold_only` | true | `B_A8_matched` | B redecides only at a true anchor, the first decision or an identity change |
| `threshold_shift` | −ln2, +ln2, +2ln2, +3ln2, +4ln2 | bootstrap mainline | named log-threshold shift; realized sparsity is measured, not inferred |

**B16.** Changing A to 16 while keeping the B parent's R8 would redecide at call 9 and make B no longer anchor-held. `ScoreCache.hold_only` prevents that; a test pins the unfixed clock's D9 as the counterexample. B8 with `hold_only` gives the identical clock to the old B8, so the v26 B_A8 results remain valid.

**CPU clock tests** (`tests/test_v27_clock.py`, over the full canvas 0..47 with origin 1):
- anchor positions for A8, A16 and A64;
- R1 means M1 (every call redecides);
- the decision age resets at an anchor, for A{8,16,64} × R{3,6};
- R6/A8 D calls fall at 7, 15, 23, 31;
- B16 has no D and anchors at 17 and 33; A64 has no in-canvas anchor;
- an early stop gives a prefix-consistent clock;
- a canvas, encoder-epoch, key-extent or mask change forces a refresh, even under hold-only.

## 3. Test guards

The old `test_v21_config_rejects_pooled_without_bootstrap_and_bad_period` used an incomplete base. It could fail on the missing policy before reaching the guard under test, and it had no bad-period case. It now does three things:
1. builds a valid minimal base and a passing control;
2. breaks one field at a time;
3. checks the specific error message.

`tests/test_v27_config.py` covers:
- the controls for M1, pooled M2, A16, A64, R6, B8, B16, B64 and threshold shifts;
- ten single-field violations;
- tampered identity rejected at validation;
- fingerprint separation of the new points.

Test status on the H100 hosts, for the project test subset:
- 400 pass;
- 15 fail identically at `3d48ebcd` (HEAD before v27). They need result files that the scratch runner does not ship. They are not regressions.

## 4. Do the v26 84 still stand?

Yes.
- No generation code path used by the 84 changed its arithmetic:
  - `hold_only` defaults to false;
  - absent keys keep old fingerprints;
  - the compact M2 is a new `mu_mode`;
  - the kernel refactor only moved the sketch load into the exact branch, which keeps the same arithmetic.
- The only defect in those receipts is the parent-provenance `score_refresh_period` field on A16 rows (§1).
- No generation was rerun.
