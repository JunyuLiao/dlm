# v24 errata and strict audit of the v23 bootstrap6 panel (append-only)

This note does not replace the v23 reports. It adds a strict union audit and five-phase accounting, and corrects over-stated wording. No generation was rerun. Machine-readable rebuild: [v24_audit.json](v24_audit.json), [v24_audit.csv](v24_audit.csv), [v24_audit.md](v24_audit.md), produced by `scripts/v24_bootstrap6_audit.py`.

## Strict segment union: PASS

- **Sources.** Two scored segments: `preview_aime`, bound at deploy `6af4d42`, and `lb_rest`, bound at deploy `dc52522` in a separate run directory. Both have the same protocol `v21_bootstrap6_b8b2c59449fd0ebf` (sha256 `e447785e…`), and each binding's `panel_protocol_sha256` matches it.
- **Method source.** `experiments/` and `src/` are byte-identical between the two commits; the diff is only runner, scorer and report files.
- **Cell checks.** 120/120 cells, each executed in exactly one segment, with placeholders resolved explicitly (no last-writer-wins). Every cell has a successful, quality-eligible, scored first run and an accepted warm run with finite positive wall time. Host, GPU UUID and cell id match the protocol assignment.
- **Ledger checks.** The raw ledgers have exactly one first run and one warm run per cell, and the call counts agree.
- **Cell states.** 0 failed, 0 missing, 0 unscored, 0 invalid. No failed or unscored row was counted as a wrong answer.
- **Tests** (`tests/test_v24_bootstrap6_audit.py`) cover duplicate and conflicting cells, placeholder resolution, host mismatch, missing and unscored cells, warm rejection, the estimand identity and conservation.

## Five disjoint GLOBAL phases (from raw receipt counters)

Here B0 = native-only output (call 0), BO = native output plus observation/route (call 1), and A, D and H are as before. Layer-calls ÷ 5 gives model calls. Conservation B0 + BO + A + D + H = 5 × decoder calls holds for every first run.

| Task | Arm | Decoder calls | B0 | BO | A | D | H | Observations (BO + A) per call |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LB | M3 incumbent | 2604 | 0 | 0 | 388 | 614 | 1602 | 0.149 |
| LB | M1 bootstrap | 2623 | 147 | 147 | 228 | 2101 | 0 | 0.143 |
| LB | M3 bootstrap | 2415 | 143 | 143 | 199 | 537 | 1393 | 0.141 |
| LB | B bootstrap | 2524 | 144 | 144 | 212 | 0 | 2024 | 0.141 |
| AIME | M3 incumbent | 1855 | 0 | 0 | 306 | 418 | 1131 | 0.165 |
| AIME | M1 bootstrap | 1847 | 156 | 156 | 120 | 1415 | 0 | 0.149 |
| AIME | M3 bootstrap | 1967 | 164 | 164 | 131 | 408 | 1100 | 0.150 |
| AIME | B bootstrap | 1931 | 169 | 169 | 121 | 0 | 1472 | 0.150 |

B0 = BO = canvases in every bootstrap arm, so every canvas reached call 1 in this panel. The existing v23 CSV A/D/H columns exclude B0 and BO. The incumbent's A count includes its call-0 anchor.

**Physical-work boundary.**
- B0 and BO run full native QK/PV on all five GLOBAL layers.
- BO additionally runs a full observed QK (grouped-Q producer), the projected-V lease and the route.
- A runs the legacy observed QK, then a second retained-tile current QK for the FP32 output.
- D and H run retained-tile QK/PV only; D also runs the route.

The receipt counters are dispatch-element upper bounds, not DRAM bytes or exact dot counts. The bootstrap arms therefore do **more** full-QK work per canvas than the incumbent at calls 0–1, which is where native output replaces a sparse one.

## Estimands (same accepted cells, same aggregation)

The paired geometric identity W = N × (W/N) holds exactly. For LB bootstrap-M3 / native: **geoW 0.9901 = geoN 1.0399 × geoW/N 0.9521**. The summed-call ratio 2415/2360 = 1.0233 is a different estimand and must not be used to explain away the W/N term.

W/N is amortized request cost per decoder call. It is not a direct model-forward price.

| Contrast (LB) | geoW | geoN | geoW/N |
|---|---:|---:|---:|
| M3 boot / native | 0.990 | 1.040 | 0.952 |
| M3 boot / incumbent | 1.000 | 0.994 | 1.006 |
| M3 boot / B boot | 0.997 | 0.976 | 1.022 |
| M3 boot / fresh T | 1.095 | 1.145 | 0.956 |
| fresh T / native | 0.904 | 0.908 | 0.996 |

The host-stratified summed decompositions in `v24_audit.md` go in opposite directions on the two hosts, e.g. LB M3-boot/native 0.863 on mpk vs 1.109 on dllm. Because host = seed here (seed 101 → 149.165.151.254, seed 202 → 149.165.159.64), host and seed effects cannot be separated. Pairing within a cell is still same-GPU and fair. Future blocks should counterbalance host by question × seed; existing first runs are not relocated.

## Corrected statements (v23 wording → supported wording)

| v23 wording | Supported wording |
|---|---|
| "each forward is ~5% cheaper" | Request wall per call is ~5% lower on LB (geoW/N 0.952). Direct bootstrap forward cost was **not measured**. |
| "clearly beats original M3" | Full-panel paired geoW vs incumbent is **0.9997**; the ~0.77 was a 2-question preview and does not carry to the full panel. |
| "bootstrap restores native behaviour" | On 6 exposed LB questions the correctness vector matches native (0 discordant cells) and aggregate calls/canvas is close; individual outputs still diverge; no general guarantee. |
| "grouped-Q improves requests by ~0.1%" | Producer-level extrapolation only; no accepted before/after request A/B. |
| "the setup is deterministic" | Within-arm warm repeats matched and native/T reproduced the v20 LB counts; this is not universal backend equivalence. |
