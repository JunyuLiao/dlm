# Summary — value-direction-aware selectors inside the v31 cross-step reuse pipeline

**CALIBRATION COMPLETE / TARGET NOT RUN / NEGATIVE RESULT.** A complete 16-arm matched panel ran on
the reserved 15-cell LongBench-v2 `0shot_think` dev subset under the frozen v31 substrate. No
value-aware selector beat the inherited mass-only control on accuracy, sparsity, or per-step cost.
No held-out accuracy or end-to-end speed claim is made, and **no `final_complete.json` is written**
because no target-suite panel completed.

`CALIBRATION_RESULT.md` holds the full table, the mechanism, and the limits. `README.md` is the
narrative record, including three protocol corrections. Frozen configuration, arm list, suite
identities, seeds, provenance and the withdrawn runs: `config.json`. Selector proof: `receipt.json`.
Active-control analysis: `SOURCE_NOTE.md`.

## 1. Headline

| | Overall % | GLOBAL sparsity | S/N (s) |
|---|---|---|---|
| `current_v31_control` (inherited, mass-only) | **73.3** | **87.60%** | **0.02443** |
| best value-aware arm (`v3a`, `v3b_shortlist`) | 66.7 | 6.72% | 0.0446 / 0.0480 |
| worst (`v3b_drop025`) | 46.7 | 10.01% | 0.0468 |
| `allkept_fa4` | 66.7 | 0.00% | 0.0326 |
| `dense_full_fix51994` | 60.0 | n/a | 0.0311 |

The control wins all three columns simultaneously. The value arms' accuracies cluster at the
all-kept consumer's 66.7, which is what keeping ~94% of tiles buys.

## 2. Requested metrics, and their status

| metric | status |
|---|---|
| official accuracy | **MEASURED on the calibration suite only** (15 cells, 1 seed, 1 repeat) via `scripts/v31_score_longbench_official.py`. Not measured on any target suite or AIME26. |
| `N`, `C`, `N/C`, `T`, `S/N` | pooled over all 15 dev cells per arm, in `calibration_dev15/aggregate.json` |
| GLOBAL physical eligible / kept tiles and sparsity, with the denominator | **MEASURED** for every adapter arm; denominators in `calibration_dev15/tables.md` |
| LOCAL physical sparsity | **zero by construction and unrouted**: LOCAL is native dense in this study, so the adapter never routes it and emits no LOCAL tile denominator. Reported as unrouted, not as 0%. |
| count-weighted overall sparsity with its stated denominator | GLOBAL eligible tiles, stated per arm; LOCAL excluded because it is unrouted |
| end-to-end `W`, per-request time | pooled per-arm means in `aggregate.json`; **no clean synchronized timing pass was run**, so no speedup claim |
| selection counters, forced keep/skip, blocked calls, sketch objective | **MEASURED**, in `aggregate.json` and the counters table of `tables.md` |

## 3. What is NOT claimed

* No held-out accuracy. The dev15 cells chose the threshold; they are not a target result.
* No end-to-end speedup. `S/N` is an amortized per-step decode cost, not a per-forward price, and no
  clean CUDA-synchronized timing pass was run.
* No inference about full-dimensional attention work removed. Sparsity is physical tile skipping in
  the adapter's GLOBAL scope only.
* No AIME26, RULER v33 or HumanEval number. The 90-cell AIME26 dense/control run was **withdrawn**
  (wrong substrate pins) and no AIME26 panel completed under the frozen pins.
* No cross-ledger bitwise comparison. The v31 ledger's per-cell hashes are not reproducible on this
  host (0/1 match at identical pins and both memory settings).

## 4. Provenance

The cross-step-reuse implementation and the online projected-V routing state are Yuhan's inherited
work. Gaussian32 / value-direction-aware routing and query-sensitivity protection are Junyu's
earlier families. The `v1`, `v2`, `v3a`, `v3b`, `v3b_drop`, `v3b_shortlist` selectors, the fused
Triton scan, the observation pass, the calibration harness, the pin guard and the panel aggregator
are this study's work, integrated inside the unchanged inherited pipeline.
