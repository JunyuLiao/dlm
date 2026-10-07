# Value-direction-aware selectors for the v31 cross-step reuse pipeline — 2026-10-07

**Status: PARTIAL.** The selectors are implemented, mathematically qualified on CPU and on the H100
against the real kernels, and integrated into the unchanged inherited reuse pipeline. No accuracy,
speed or end-to-end claim is made.

## Protocol deviations found on 2026-10-07 (read before using any number in this directory)

Three findings materially change what the earlier measurements in this directory mean.

1. **Every generation run so far was produced under the WRONG substrate pins; none of it is study
   evidence.** This covers `smoke_records.json` *and* the 90-cell AIME26 dense and
   `current_v31_control` panel run on 2026-10-07 13:18-13:43; their records carry
   `block_size=32, chunk=16384, max_model_len=13312`, not the frozen AIME26
   `(64, 4096, 9216)`. Consequently the AIME26 accuracies obtained from that panel (dense
   avg@k 56.67 / 51 of 90, control avg@k 54.44 / 49 of 90) are **not valid study results** and are
   withdrawn; they are recorded here only so the numbers are not silently rediscovered later.
   `scripts/v32_panel_aggregate.py --require-pins` now refuses off-pin records so this cannot recur.
   The underlying smoke/probe runs were collected with
   `CHUNK=16384` and `BLOCK=32`, because `scripts/v31_vllm_paired_bench.py` defaults to those values.
   The frozen v31 AIME26 panel runs `CHUNK=4096` and `BLOCK=64`
   (`results/v31_20261006_aime_global_local/attempt001/full/public/*.jsonl` records
   `chunk=4096`, `block_size=64`). Block size changes the selector's tile geometry and chunk size
   changes forwards per canvas. The smoke records are retained as a **non-protocol probe** (they did
   prove the code paths run end-to-end); the 90-cell AIME26 panel must be re-run. Per-suite pins are
   now frozen once in `scripts/v32_value_arm.sh`: AIME26 `BLOCK=64 CHUNK=4096 MAX_MODEL_LEN=9216`,
   LongBench-v2 `BLOCK=32 CHUNK=16384 MAX_MODEL_LEN=141312` (no frozen v31 LongBench pin exists in
   this repository; the LongBench values match the earlier probe's implied `ceil(120017/16384)=8`
   prefill steps).
2. **The blocking GPU fault was self-inflicted and is fixed.** The `illegal memory access` fault
   below was caused by launching vLLM with a private `HOME`/`XDG_CACHE_HOME`/`TRITON_HOME`, which
   forced FlashInfer to JIT-compile fresh kernels; those kernels faulted inside the
   `flashinfer_autotune` dummy runs during `compile_or_warm_up_model`. Re-running with the host's
   default cache paths removes the fault, confirmed by a clean dense AIME26 cell
   (`output_tokens=1870`) and a clean 90-cell dense panel. It was never a host or driver fault.
3. **The v31 ledger's per-cell hashes are not bitwise reproducible on this host.** With the frozen
   AIME26 pins restored (`chunk=4096 block=64 max_model_len=9216`, confirmed in the run records),
   the dense arm still matched **0 of 1** historical `output_hash` values on the probed cell, at both
   `gpu_memory_utilization` 0.80 and 0.88. Conclusions must therefore rest on arms run together in
   one panel under identical pins; cross-ledger comparison is metric-level only and any control delta
   must be reported rather than hidden.
4. **The GPU is shared and currently held by another tenant.** A separate `run_aime_eval.py` job
   occupies ~61.5 GiB of the single H100 for long stretches, so a vLLM engine needing ~59 GiB cannot
   always start. It is never touched. `scripts/v32_wait_gpu.sh` waits for free memory, and
   `scripts/v32_value_arm.sh` retries an arm on the `Free memory on device` refusal, writing each
   retry to a new attempt directory. `MEM=0.75` is now used, which still leaves ~8.9 GiB of KV for
   the 141,312 tokens LongBench-v2 needs (measured KV rate ~0.03234 MiB/token), and each successful
   arm asserts that the reported KV cache size really covers `MAX_MODEL_LEN`.

Read `SOURCE_NOTE.md` first: it identifies the active control's actual formula (it is a MASS-ONLY
selector, not MAGE eq. 5) and states exactly what this study changed.

## Provenance

* The first cross-step-reuse implementation and the online projected-V routing state in
  `integration.py` / `reference.py` are **Yuhan's inherited work**.
* Gaussian32 / value-direction-aware routing and the query-sensitivity / `C_gate` protection are
  **Junyu's earlier method families**.
* Every selector here (`v1`, `v2`, `v3a`, `v3b`, `v3b_drop`, `v3b_shortlist`), the Triton scan, the
  observation pass and the tests are **new work of this study**, integrated *inside* the unchanged
  inherited pipeline. Nothing is attributed to either earlier author.

## What was built

`experiments/numerical_qk_reuse/v31_value_select.py` holds the shared Gaussian32 statistics, the four
selector families and their counters. `v31_value_scan.py` is a fused Triton scan for V1/V2 whose
decisions must equal the batched reference exactly. `v31_value_observe.py` supplies the per-row
attention-weighted projected within-tile mean with **one observation-only Triton pass** that loads no
value, forms no PV and writes no output partials. `vllm_adapter.py` gained one optional parameter,
`value_selector`; with `None` the control path is untouched.

At an initial/refresh call the value-aware path makes the **same** FA4 observation call as the
control, so the output is FA4's own native current-step BF16 output and the tile masses `z` are the
control's own. It then adds the observation-only statistics pass and a selector that returns the same
`[1, H, QB, KT]` bool map. **No skipped block's old mass or old value is ever carried into reused
attention**: nothing but the map is stored, and held calls run the unchanged FA4 consumer with the
current step's Q/K/V.

## Correctness gates (all run; all pass)

| gate | result |
|---|---|
| CPU mathematics (`tests/test_v31_value_select.py`) | **26 passed** |
| H100 kernels vs direct oracles (`tests/test_v31_value_select_gpu.py`) | **26 passed** |
| Adapter / reuse-contract (`tests/test_v31_value_adapter.py`) | **37 passed** |
| existing suite regression | 265 passed, 4 pre-existing failures unrelated to this study |

The listed gates from the brief are each covered and named:

* V1/V2 match a simple sequential reference under identical decisions — `test_v1_and_v2_match_a_simple_sequential_reference`.
* V1 excludes skipped mass; V2 includes skipped mass but leaves its **normalized** routing output
  unchanged **including when the running max moves** — `test_v1_excludes_skipped_mass_but_v2_keeps_it`,
  `test_v2_skip_advances_the_denominator_only`. V2's skip advances the denominator *and* the numerator
  by `o_hat * Z_ij`; advancing only the denominator would be wrong, and advancing the numerator by
  `Z_ij mu_ij` would silently make V2's state V3's full-support mean instead.
* First valid support, strict threshold ties, mandatory blocks, partial tiles, empty rows, invalid
  scores, GQA mapping and the fixed-k quota — `test_first_valid_support_is_always_kept`,
  `test_quota_forced_skip_and_forced_keep_and_ties`, `test_protected_tiles_are_mandatory`,
  `test_partial_tiles_and_empty_rows`, `test_tile_statistics_handle_dead_rows_and_tiles`,
  `test_gqa_mapping_broadcasts_the_kv_head`, `test_fixed_k_quota_is_never_exceeded`.
* Identity projection equals a direct full-dimensional formula reference —
  `test_identity_projection_agrees_with_a_full_dimensional_reference` (GPU, FP64).
* V3a/V3b match direct masked-attention recomputation in sketch space —
  `test_v3a_matches_a_direct_singleton_deletion_ranking`,
  `test_v3b_exact_matches_a_direct_cumulative_greedy`, `test_masked_attention_sketch_matches_direct_recomputation`.
* Optimized V3b equals exact V3b with the approximation features disabled —
  `test_v3b_exact_equals_the_path_with_approximations_disabled`; the constructor **refuses** to run
  `v3b_drop`/`v3b_shortlist` without the feature that makes them approximations.
* Identical masks give identical downstream outputs regardless of selector —
  `test_identical_maps_give_identical_fa4_sparse_outputs` (all six arms, real FA4 consumer).
* The equal-mass regression: `mu_A = mu_B = a`, `mu_C = b`, skip B and retain C. V2's internal state
  ends at `(2a + b)/3` while ordinary reused attention over `{A, C}` produces `(a + b)/2`; the test
  distinguishes them, asserts they differ by more than the comparison tolerance, and adds **no
  executor compensation** — `test_equal_mass_regression_v2_state_vs_masked_attention` (CPU) and
  `test_equal_mass_regression_survives_on_gpu` (H100, which also checks the real FA4 consumer on an
  equivalent `{A, C}` map is *not* mass-preserving).

Two hygiene repairs were needed in existing tests and are recorded in `SOURCE_NOTE.md`:
`tests/test_v31_progress_aware.py` and `tests/test_v31_progress_clock.py` replaced
`v27_fa4.dense` / `sparse_lists` / `block_sparse_tensors` at module level and never restored them, so
every later real-kernel test inherited the stubs. Both now restore the originals.

## Superseded probe: one cell per arm (H100, sanitized) — NOT protocol evidence

These were collected under `CHUNK=16384 / BLOCK=32` and must not be cited as study results; see
deviation 1 above. They are kept because they establish that every code path executes end-to-end.
A pinned, correctly warmed re-run of the same cells follows in the pinned panels.


`smoke_records.json`. Same seed, same cell, same sampler, same pins; `MAGE_K = 1728` (27 KV64 tiles
per unit), refresh trigger `0.15` on the settle signal, sticky `1.386`, `FIX_51994=1`,
`FA4_LOCAL_FIX=1`, LOCAL disabled. **One request each.** These are single-cell observations, not a
panel: they carry no accuracy and no pooled comparison.

### AIME26 cell 0 (187-token prompt, thinking on, budget 8192)

| arm | N | canvases | output tokens | decode s | kept/prefix tiles | retained mass |
|---|---:|---:|---:|---:|---|---:|
| `dense_full_fix51994` | 66 | 8 | 1870 | 1.35 | — | — |
| `current_v31_control` | 60 | 8 | 1827 | 1.39 | 35680 / 36160 | — |
| `value_v1_online_discard_mass` | 60 | 8 | 1827 | 2.27 | 48160 / 36160 | 1.000 |
| `value_v2_online_preserve_mass` | 60 | 8 | 1827 | 2.29 | 48160 / 36160 | 1.000 |
| `value_v3a_singleton_delete` | 60 | 8 | 1827 | 1.54 | 48160 / 36160 | 1.000 |

The control reproduces the frozen 2026-10-06 trajectory exactly (60 forwards, 8 canvases, 1827
tokens), which is the integration check that the reuse contract is unchanged. **At this prompt length
the budget is not binding** (the whole prefix fits inside 27 tiles), so every selector keeps
everything and the maps are identical; this cell cannot separate the selectors and is reported only
as an integration proof.

### LongBench-v2 `0shot_think` cell 0, 120017 prompt tokens (thinking on, budget 16384)

| arm | N | canvases | output tokens | decode s | objective_max | retained mass | prefix tiles kept/eligible | blocked |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `dense_full_fix51994` | 271 | 12 | 2982 | 9.95 | — | — | — | — |
| `value_v1_online_discard_mass` (Triton scan) | 238 | 12 | 3042 | 11.77 | 0.0605 | 0.9642 | 6635520 / 7284480 | 0 |
| `value_v1_online_discard_mass` (batched reference) | 238 | 12 | 3042 | 118.14 | 0.0605 | 0.9642 | 6635520 / 7284480 | 0 |
| `value_v2_online_preserve_mass` (Triton scan) | 273 | 12 | 3026 | 13.14 | 0.0742 | 0.9683 | 6635520 / 7284480 | 0 |
| `value_v3a_singleton_delete` | 220 | 11 | 2661 | 10.96 | **0.0136** | **0.9945** | 6100160 / 6670400 | 0 |
| `value_v3b_greedy_exact` | 348 | 12 | 3045 | 11.07 | — | — | 103680 / 7284480 | **120** |
| `value_v3b_drop_r025` | 417 | 11 | 2739 | 25.95 | 1.2024 | 0.7040 | 6100160 / 6670400 | 0 |

What these single cells do and do not support:

* The **fused Triton V1 scan is 10.0x faster than the batched reference and produces a bit-identical
  map and identical counters** (same N, same token count, same objective, same retained mass). That is
  the kernel result the study needed; it is a *selection-cost* result, not a speed result.
* **V3a reaches a 4.4x lower masked-attention objective than V1/V2** (0.0136 vs 0.0605/0.0742) and
  retains 99.45% of the tile mass versus 96.4%. On this cell it is also the closest to dense in
  decode time (10.96 s vs dense 9.95 s, control-comparable).
* **Exact `v3b` is honestly blocked at a 120K prefix.** Its candidate-evaluation count grows as
  `O(PT^2 N R)`; at 1875 optional tiles it reported `value_blocked` on 120 of 120 selection calls and
  fell back to the inherited control, which is why its kept/eligible count collapses to
  103680 / 7284480 (that is the control's map, not a value-aware one). An approximation was **not**
  silently substituted.
* **The `v3b_drop` approximation is poor at `drop_fraction = 0.25`**: objective 1.20 and 70.4%
  retained mass, both far worse than V3a, and 2.5x the decode time of V3a. A coarser batch loses too
  much before the refine pass can recover it. This is a negative result for that setting, on one
  cell.
* `value_v3b_shortlist64` is implemented and unit-tested but its 120K run did not produce a record.

## Blocked suites — precise reasons

| suite | status | reason |
|---|---|---|
| **RULER official v33** (13x15 at 32K/64K/128K, 585 cells) | **BLOCKED, not attempted** | the frozen v33 *source* pool lived at `/media/volume/dllm-1/dyh/ruler_long_v33`, which does not exist on this host. The writer that produced the v33 manifest/gold/`raw/` tree is a private orchestrator that is not in the repository, so the pinned manifest **row order** is unrecorded and a rebuilt pool could not be hash-verified against the pinned `4142be0da5…`, `e4d15823f0…`, `834a1374d7…`. The RULER checkout itself is present and verified at `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`, and its generator is deterministic under seed 6464, but an unverifiable pool is not the official pool and rebuilding prompts is out of scope. |
| **HumanEval 164** | **BLOCKED** | no official 164-task pool and no `openai_humaneval` data on this host; `scripts/v31_build_official_pools.py` has no humaneval builder. |
| **AIME26** | available | the pool of the frozen 2026-10-06 control panel (manifest `85ebd2dd…`, 30 problems, seeds 42/43/44, thinking on, budget 8192), so it is directly comparable to the control. It is **not** the official `aime26_b32k` pool. |
| **LongBench-v2 `0shot_think`** | available and **byte-identical to the official pool** | rebuilt here with the pinned builder against the pinned LongBench checkout; every output sha256 equals `pools_summary.json`, including `longbench_v2_0shot_think_generation_manifest.json` = `3209c749b5cb74a02c9198e71e8da94909752cdabd2947a4b74bd1487b6440fe` and `cells_longbench_v2_0shot_think.json` = `d48c39d68533467fd4942931745377a42a9d57e3e5714aa8913f5d897e7d4957`. |

## The (now-fixed) blocking fault

Every end-to-end launch began to fail, **including the plain `dense_full_fix51994` arm with no
adapter at all**, inside vLLM's own kernels and never inside this study's code:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
  ... vllm/v1/worker/gpu/model_runner.py:1172 add_requests -> execute_model -> warmup
      vllm/model_executor/layers/fused_moe/experts/triton_moe.py:449 apply
  and, in a later reproduction, torch/_inductor/runtime/triton_heuristics.py:2086 run
```

It reproduces on `arm='dense'` with `value_selector=None`, so it is not caused by the study. Runs
that had already completed before the fault appeared are the 12 records above. Attempts to recover by
deleting the Triton cache, the CUDA cache, the vLLM `XDG` cache and the run directories did **not**
help; the fault persisted across four clean-cache reproductions. The panel driver
`scripts/v32_aime26_panel.sh` is committed, ordered and resumable (one arm at a time, `attemptNNN`
directories, failed attempts retained), so the panels can be relaunched unchanged once the host GPU
fault is resolved.

## Limitations

* Everything above is **one cell per arm**. There is no pooled accuracy, no paired comparison, no
  uncertainty interval, no clean-timing pass and no `final_complete.json`; the run is incomplete by
  design of the blockage.
* The V3 objective and the retained mass in the table are measured **after** selection as receipt
  values and are never selector inputs. They are in **sketch space (rank 32)**; the full-dimensional
  attention-output error diagnostic was not reached.
* The `v1`/`v2` threshold `0.01` was set from a single probe cell, not from a reserved development
  subset, and is therefore **not frozen** in the sense the brief requires. The reserved 15-item
  stratified LongBench development subset (`cells_lb2think_dev15.json`, drawn with seed 20261007
  before any target run) exists for that purpose but was never used.
* LOCAL sparsity is zero by construction here, not by measurement: no LOCAL router is active and no
  LOCAL budget is set, and every value receipt states `local_layers_sparse: []`.
* The historical AIME26 V31 `0.951x` end-to-end figure is a diagnostic reference only and says
  nothing about any arm of this study.