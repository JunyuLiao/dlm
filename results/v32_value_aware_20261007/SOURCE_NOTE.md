# Source note: the ACTIVE v31 control, and what this study changed

Branch `value-aware_cross-step-reuse`, base commit `cdc12221` ("add value-aware cross-step reuse
delegation prompt"). This note is the source-level record the study is required to keep. It is
written BEFORE any new result and is not edited afterwards.

## 1. Which implementation is the active control

The branch carries two selector paths and the words in the code do not tell you which one runs:

* `arm='method'` (`VllmMethodAdapter`, `experiments/numerical_qk_reuse/integration.py`,
  `cached_executor.py`) is the INHERITED ONLINE PROJECTED-V router: it keeps an online retained-state
  risk in log space (`risk = log(||alpha (mu - projected)|| / ref) + log(sensitivity)`), scans tiles in
  ascending order and keeps a block when the worst row's risk clears a threshold. This is Yuhan's
  inherited line.
* `arm='mage'` (`VllmMethodAdapter._mage_select_fa4`, `experiments/numerical_qk_reuse/vllm_adapter.py`)
  is the fixed-budget qblock selector and it is what the active control actually runs.

The word `mage` must not be read as the MAGE paper's formula. The active control's effective
settings, read out of a real record
(`results/v31_20261006_aime_global_local/attempt001/full/public/aime_mage_k1728_*.jsonl`), are:

| setting | effective value |
|---|---|
| adapter arm | `mage` |
| `MAGE_SELECT` | `fa4` |
| `MAGE_GRAN` | `qblock_max` |
| `MAGE_K` (GLOBAL KV budget) | **1728 tokens = 27 physical KV64 tiles per unit** |
| `MAGE_STEP` | 1 |
| `MAGE_CARRY` | 1 (canvas call 0 runs on the previous canvas's map) |
| `MAGE_RESELECT_TRIGGER` | `[0.15]` |
| `MAGE_TRIGGER_SIGNAL` | `settle` |
| `MAGE_STICKY` | 1.386 |
| `MAGE_SINK` / `MAGE_RECENT` | 0 / 0 (no in-budget protection in the active config) |
| `KV_COPY` / `MERGE` | `triton` / `triton` |
| `LOCAL_KV_BUDGET` | 512 in that panel; **disabled for this study** |
| cudagraph mode | `PIECEWISE` (sparse) / `default` = FULL (dense) |
| dense mask fix | `FIX_51994=1`, `FA4_LOCAL_FIX=1` |
| LOCAL scope | 5 GLOBAL layers `[5, 11, 17, 23, 29]`, 25 LOCAL layers native dense, window (1023, 1023) |

## 2. The active control's ACTUAL formula

`_mage_select_fa4` calls `v31_fa4_observe.observe_dense`, the official FA4 SM90 dense forward with
an extended `ObservingMask` that writes, per (query row, wholly-prefix 64-key tile),

    z[h, qb, pt, 128] = logsumexp_u ( scale * q_row . k_u )

with the FP32 tile max and an `ex2.approx` exponential inside the kernel; the dense output and LSE
are bit-identical to FA4's. The remaining canvas/boundary tiles come from an FP32 tail product with
TF32 disabled. The selector then builds, per row, the softmax-normalized tile mass and aggregates it
over the 128 rows of the map's query block:

    share[i, j] = z[i, j] - logsumexp_l z[i, l]           (the row's PREFIX log-share)
    score[h, b, j] = max over the block's rows of share[i, j]        <- `qblock_max`

then takes the top `k = MAGE_K // 64` tiles of that score per `(query head, 128-row block)`. The
canvas/boundary tiles `j >= pt` are always kept and are **not** charged against the budget.

Therefore, to answer the distinctions the study asks for:

* **This is a MASS-ONLY score.** It never touches V. It is not MAGE eq. 5 either: MAGE eq. 5 averages
  the tile mass over the canvas queries AND over the query heads of a KV head and keeps one set per
  KV head; the active control is per (query head, 128-row block) and uses the WORST row's prefix
  log-share. The `_mage_select` torch path (`mage_select='torch'`) is the eq.-5 shape; the active
  config uses the FA4 path.
* **Not `qblock_max`/worst-row prefix-mass share of a projected value.** No projected value is
  involved at all.
* **Not the inherited online projected-V risk.** That is `arm='method'`.
* **C_gate / row weights:** the active control does not use them (`mage_row_weight` is null). They are
  available only as a refresh-time row weight on `qblock_max` (`MAGE_ROWW`) and are used only at a
  re-selection, never at the first selection.
* **Budget convention:** `MAGE_K` is in TOKENS and is divided by 64, so `1728 -> 27` physical KV64
  tiles per unit; `min(k, prefix_tiles)` is used.
* **Mandatory blocks:** the canvas/boundary tiles `j >= pt`. Sink/recent protection exists but is 0.
* **Tie rule:** `torch.topk` on the score; ties are resolved by `topk`'s index order, which is not
  documented as stable and is therefore NOT reused by this study's selectors.
* **Scan order:** ascending physical KV tile index (the scan order for the causal quota).
* **Refresh clock:** the settle-based progress clock at `MAGE_RESELECT_TRIGGER=0.15`, one re-selection
  per crossing, carry-first across canvases, sticky `+1.386` for already-held tiles.

## 3. What this study changed, and what it did not

Changed: **only the block-selection formula at an initial/refresh call**, behind a new optional
adapter parameter `value_selector` (default `None`).

Not changed, and asserted unchanged by `tests/test_v31_value_adapter.py`: the discovery/refresh
clock, the first-call behaviour, the carry policy, sticky retention, protected/sink/current-canvas
tiles, the structural masks, GQA/cache semantics, the mask format, the fixed budget, layer scope,
the sampler, the canvas, and the later sparse consumer. The FA4 observation call's output is still
FA4's own native current-step BF16 output: the value-aware logic runs beside it and writes only a
keep map of the same `[1, H, QB, KT]` bool type. No hidden forward, no fixed-step loop, no LOCAL
routing, no new kernel.

New files:

| file | role |
|---|---|
| `experiments/numerical_qk_reuse/v31_value_select.py` | the selectors, the statistics oracle and the counters |
| `experiments/numerical_qk_reuse/v31_value_observe.py` | the observation-only projected-mean pass |
| `experiments/numerical_qk_reuse/v31_value_scan.py` | the fused Triton V1/V2 scan |
| `tests/test_v31_value_select.py` | 26 CPU mathematical gates |
| `tests/test_v31_value_select_gpu.py` | 26 H100 gates against the real kernels and oracles |
| `tests/test_v31_value_adapter.py` | 37 adapter/contract gates |
| `scripts/v32_value_arm.sh` | the one-arm-at-a-time launcher |

Two hygiene repairs to existing tests were needed and are recorded here: `tests/test_v31_progress_aware.py`
and `tests/test_v31_progress_clock.py` replaced `v27_fa4.dense` / `sparse_lists` /
`block_sparse_tensors` at module level and never restored them, so any later real-kernel test
inherited the stubs. Both now save and restore the originals in their `finally` blocks.

## 4. Where the value-aware selectors get their statistics

At an initial/refresh call the value-aware path makes the SAME FA4 observation call as the control
(`v31_fa4_observe.observe_dense`) and therefore the same output and the same tile masses `z`. It then
adds one observation-only Triton pass, the repository's existing `v27_consumer64.fused_observe` with
`OUT=0, MU=1`: no value load, no PV, no output partials, so it cannot change the call's output and is
not a second model forward. That pass writes the per-row, per-tile attention-weighted projected mean

    mu_ij = ( sum_{u in j} exp(s_iu) z_u ) / ( sum_{u in j} exp(s_iu) ),    z_u = v_u R

with `R in R^{512 x 32}` drawn ONCE per (layer, native KV head) from the repository's frozen
Gaussian32 bank, seed **1729**, `R_ab ~ N(0, 1/32)`, derived from
`sha256('jl_output_v1/gaussian/32/1729/<layer>/<head>/<width>')`. The bank is cached for the process
and is never regenerated per step, canvas or variant; the per-head matrix sha256 goes into every
record. The canvas/boundary tiles' statistics are reduced from the FP32 tail logits the control
already forms, so V3's reference `O_i` covers the complete eligible support.

`nu` is the INHERITED valid-KV reference scale: per native KV head, `sqrt(sum_u ||v_u||^2 /
count_valid)`, clamped at 1e-12, broadcast from the KV head to its query heads. Row aggregation is
`max` over the valid rows of the 128-row block, i.e. exactly the active control's `qblock_max`
aggregation, preserved across every arm and never tuned per method.