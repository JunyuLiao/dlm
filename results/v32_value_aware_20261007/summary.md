# Summary — value-direction-aware selectors inside the v31 cross-step reuse pipeline

**PARTIAL / BLOCKED.** Correctness, integration and per-arm H100 smoke are complete; the accuracy and
clean-timing panels did not run. `README.md` gives the full statement; this file is the metric table
and the explicit list of what is and is not claimed. No `final_complete.json` is written.

Source commit `cdc12221` (branch `value-aware_cross-step-reuse`). Frozen configuration, arm list,
suite identities, seeds and provenance: `config.json`. Selector proof: `receipt.json`. Active-control
analysis: `SOURCE_NOTE.md`. Sanitized per-cell records: `smoke_records.json`.

## 1. Requested metrics, and their status

| metric | status |
|---|---|
| official accuracy, per-item/per-seed counts | **NOT MEASURED** (panels blocked) |
| total denoising calls `N`, canvases `C`, `N/C`, output length `T`, `S/N` | one cell per arm only (table below); no pooled value |
| GLOBAL physical eligible / kept / skipped tiles and sparsity | **MEASURED** for the value arms on one 120K cell; the control's own global counters are in the same records |
| LOCAL physical sparsity | **zero by construction** (no LOCAL router, no LOCAL budget; every receipt states `local_layers_sparse: []`) — not measured on a sparse LOCAL arm |
| count-weighted overall decoder-attention sparsity with its denominator | **NOT MEASURED** (needs the pooled panel) |
| end-to-end `W` with `W_dense / W_sparse` | **NOT MEASURED** under the required separate clean pass |
| decode-only `S` with speedup | **NOT MEASURED** under the required separate clean pass |
| initial/refresh/held-call counts, carry/reselection counts | **MEASURED** per cell (`counters.*`, `value_passes`) |
| forced-keep / forced-skip rates | **MEASURED** per cell (`value_forced_keep`, `value_forced_skip`, `value_threshold_keeps`) |
| fallback / invalid-row counts | `value_blocked` measured; invalid-row counts are covered by the CPU/GPU gates, not by a panel counter |
| C_gate counters | not applicable: the active control does not use C_gate, and this study did not add it |
| selection/discovery latency, peak memory, candidate evaluations, amortized cost | **MEASURED** on one cell: the Triton scan is 10.0x faster than the batched reference at an identical map; `value_stat_bytes` and `value_mu_passes` are recorded per request |
| retained attention mass, mask overlap, fresh-vs-reused error by distance from refresh, full-dimensional attention-output error offline | retained mass **MEASURED** (sketch space, post-selection); overlap, distance-resolved and full-dimensional diagnostics **NOT MEASURED** |

## 2. One-cell H100 measurements (not a panel)

LongBench-v2 `0shot_think`, cell 0, 120017 prompt tokens, thinking on, budget 16384,
`MAGE_K = 1728` (27 KV64 tiles/unit), refresh trigger 0.15 on the settle signal, sticky 1.386,
LOCAL disabled, `FIX_51994=1`, `FA4_LOCAL_FIX=1`, dense FULL / sparse PIECEWISE. One request per arm.

| arm | N | C | N/C | T | decode s | prefill s | objective_max (sketch) | retained mass | prefix tiles kept / eligible | blocked |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `dense_full_fix51994` | 271 | 12 | 22.58 | 2982 | 9.95 | 5.54 | — | — | — | — |
| `current_v31_control` | not run on this cell | | | | | | | | | |
| `value_v1_online_discard_mass` | 238 | 12 | 19.83 | 3042 | 11.77 | — | 0.0605 | 0.9642 | 6635520 / 7284480 | 0 |
| `value_v2_online_preserve_mass` | 273 | 12 | 22.75 | 3026 | 13.14 | — | 0.0742 | 0.9683 | 6635520 / 7284480 | 0 |
| `value_v3a_singleton_delete` | 220 | 11 | 20.00 | 2661 | 10.96 | — | **0.0136** | **0.9945** | 6100160 / 6670400 | 0 |
| `value_v3b_greedy_exact` | 348 | 12 | 29.00 | 3045 | 11.07 | — | — | — | 103680 / 7284480 | **120** |
| `value_v3b_drop_r025` | 417 | 11 | 37.91 | 2739 | 25.95 | — | 1.2024 | 0.7040 | 6100160 / 6670400 | 0 |

AIME26 cell 0 (187-token prompt), one request per arm:

| arm | N | C | T | decode s |
|---|---:|---:|---:|---:|
| `dense_full_fix51994` | 66 | 8 | 1870 | 1.35 |
| `current_v31_control` | 60 | 8 | 1827 | 1.39 |
| `value_v1_online_discard_mass` | 60 | 8 | 1827 | 2.27 |
| `value_v2_online_preserve_mass` | 60 | 8 | 1827 | 2.29 |
| `value_v3a_singleton_delete` | 60 | 8 | 1827 | 1.54 |

The control reproduces the frozen 2026-10-06 trajectory exactly, which is the proof that the reuse
contract is unchanged. At 187 prompt tokens the 27-tile budget does not bind, so all value arms keep
the whole prefix and their maps coincide with the control's; that cell cannot separate the selectors.

## 3. Paired / matched comparisons

**None.** A paired comparison requires the panel, which did not run. The only comparison that holds
is the *within-cell selector-quality* comparison in section 2: V1, V2 and V3a at the same budget, the
same cell, the same seeds and the same runtime pins, scored by the same masked-attention objective in
sketch space. On that one cell V3a is 4.4x better than V1 and 5.4x better than V2 on the objective and
retains 3.0 percentage points more tile mass. One cell is not evidence of a population difference.

## 4. Honest reading of the one result that is unambiguous

The **kernel** result is unambiguous and is a selection-cost result, not a speed result: the fused
Triton V1 scan reproduces the batched reference's map and counters exactly and is **10.0x faster**
(11.77 s vs 118.14 s of decode on the same request, same decision, same objective, same retained
mass). The cost of the value-aware statistics at a 120K prefix is real and is recorded per request in
`value_stat_bytes` (≈123 GB of tile-major statistics materialized across the 120 selection calls of
that cell) and `value_mu_passes`.

## 5. Blocked, and why

See `README.md`. Summary: the accuracy and clean-timing panels are blocked by a CUDA illegal memory
access inside vLLM's own fused-MoE / inductor kernels that reproduces on the **dense** arm with no
adapter attached; four clean-cache reproductions did not clear it. The RULER v33 and HumanEval pools
are additionally unobtainable on this host, with the reasons recorded per suite in `config.json`.
`scripts/v32_aime26_panel.sh` is committed, ordered and resumable so the panel can be relaunched
unchanged.