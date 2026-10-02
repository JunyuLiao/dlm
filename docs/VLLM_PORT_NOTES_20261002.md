# Porting the method into vLLM's DiffusionGemma (design notes, 2026-10-02)

Why: vLLM 0.30.0's native DiffusionGemma is the fastest official serving system we measured.
- Its per step is 0.76× / 0.90× / 0.98× ours at 32K / 64K / 96K, and prefill is about 0.5×
  (`results/.../vllm_dense_check_1002/README.md`).
- Paper-grade speed claims need dense and sparse inside the same official system, differing only in skipped blocks.

Environment: dllm, `/home/exouser/dyh/dlm_models_20261002/envs/vllm` (vLLM 0.30.0, torch 2.13.0+cu130). Paths below are
relative to its `site-packages/vllm`.

## What vLLM does today (read from the installed code)

- **Model:** `model_executor/models/diffusion_gemma.py` (1,427 lines).
  - One Gemma4 backbone run in two modes: the encoder is causal and writes the KV cache; the decoder is bidirectional
    over the canvas, reads the encoder KV and does not commit it.
  - The sampler is `_compiled_sample_step` (temperature schedule, Gumbel, entropy confidence, stability threshold,
    entropy bound), with parameters from the model's generation config.
  - Per-position argmax and acceptance live in `DiffusionGemmaRequestStates` (`canvas`, `argmax_canvas`, `step_tensor`).
- **Attention:** `v1/attention/backends/flash_attn.py`.
  - The FA4 version is selected for diffusion models on SM90 (`fa_utils.py`: "Per-sequence causal (dynamic_causal)
    requires FA4").
  - The decoder's GLOBAL call is `_FA4_DENSE_ATTENTION_KERNEL(q, k=key_cache, v=value_cache, cu_seqlens_q,
    seqused_k, max_seqlen_k, block_table, dynamic_causal, num_splits=attn_metadata.max_num_splits, ...)`. It is a
    varlen call over the paged cache.
  - For SM90 hd512 it is a pre-compiled dense family (warm-up over split counts 1/32/64/128/256; page size = KV block
    size, `MultipleOf(16)`). It has **no block-sparse argument**.
- **Memory:** one H100 in bf16 serves 96K only with chunked prefill, because vLLM keeps full-length KV for every
  layer (about 224 KB per token).
- **Seeds:** vLLM rejects per-request `seed`/`temperature` for diffusion models, so panels cannot pair trajectories by
  seed.

## What we verified (`results/.../vllm_port_probe_1002/README.md`)

- The raw FA4 interface (`vllm_flash_attn/cute/interface._flash_attn_fwd`) with `page_table`:
  - dense attention is exact at page size 64;
  - page size 16 fails through this raw entry point (vLLM itself goes through the compiled dispatch);
  - **block-sparse lists + page_table read the wrong pages** (error about 22; contiguous is exact).

- Follow-up facts:
  - vLLM's GLOBAL layers use KV page size 32 (LOCAL 16).
  - Real requests' block tables are non-contiguous from the first request on.
  - The raw FA4 entry runs paged attention only at page 64; 32 and 128 fail there, while vLLM's own pre-compiled dense
    dispatch handles 32.
  - So option a) below needs an allocator change, and the realistic route is b): kernel work in the CuTe FA4 sparse
    iteration at page 32 (two pages per 64-key tile), or page 64 for the GLOBAL group plus the TMA paged fix.

## Port plan

1. **Block-sparse over the cache.** Pick the cheapest correct option:
   - a) Contiguous view. With batch 1 and a fresh engine, check whether the request's block table is a contiguous run
     (`block_table[0, :n] == arange(start, start + n)`). If it is, pass `key_cache.view(-1, hk, d)[start*bs :
     start*bs + seqused]` as contiguous K/V to the raw FA4 call with our lists. This is our tested path, exact.
     Otherwise force the KV block size to 64 and allocate contiguously for the panel's single request.
   - b) Fix FA4's paged block-sparse iteration: translate each sparse n-block through the page table, as the dense
     path does. Add `scripts/v27_vllm_paged_sparse_probe2.py` as a regression test, and report upstream
     (vllm-project/flash-attention).
   - c) FlashInfer BSR over paged KV, only if its hd 512 SM90 speed is close to FA4's (unverified).
2. **Dense reference inside vLLM.**
   - Keep vLLM's own `_FA4_DENSE_ATTENTION_KERNEL` call as the dense arm (official, unchanged).
   - Also time FA4 "all tiles kept" through the sparse entry, as in `D_fa4_allkept`, to show that the sparse entry
     itself costs nothing.
3. **Method hooks** (a thin adapter, no edits to the method core):
   - per canvas: call 0 uses the carried map; call 1 is the observation over the cache's prefix (the observation
     kernel needs the same contiguous-or-paged K/V access) plus the dense-prefix risk table;
   - held maps for calls 2–7, re-decision at 8, 14, …;
   - query sensitivity T from the per-position argmax flips in `DiffusionGemmaRequestStates.argmax_canvas` between
     steps;
   - async route on a side stream, as now.
4. **Measurement.**
   - vLLM dense vs vLLM + M3 + c0 on the same items. Panels cannot pair by seed, so use many items × repeats.
   - Report the full metric set plus a direct per-forward timing on common states (no seed pairing).
   - Keep vLLM's chunked-prefill and memory settings identical across arms.

## Update: the paged block-sparse bug is fixed (2026-10-02, `patches/`)

- **Cause:** FA4's SM90 block-sparse producer handed logical KV-block indices to the paged TMA load.
- **Fix:** translate them through the page table, as the dense path does. A 30-line patch in our dyh vLLM env.
- **Verified:** paged block-sparse now equals contiguous exactly (page size 64).
- **Remaining for the port:**
  1. Run vLLM's GLOBAL KV group at page size 64 (TMA path) and check that vLLM dense is not slower than at its default
     page 32. If it is slower, the dense arm keeps page 32 and we also need a cp.async block-sparse port.
  2. Add `block_sparse_tensors` to vLLM's FA4 dense dispatch.
  3. Hook the method into `diffusion_gemma.py`.
  4. Measure vLLM dense vs vLLM + M3 + c0.
  5. With the user's go-ahead, report the bug and the patch upstream.

## Page size: what the official stack does and the decision rule (2026-10-02)

- **No model-specific page size.** The official recipe (`vllm/vllm-openai:gemma`) sets none. vLLM's default
  `--block-size 16` applies to the largest-page layers (LOCAL, 8 KV heads × 256). `unify_kv_cache_spec_page_size`
  then raises the block size of smaller-page layers until every page has the same bytes, so the GLOBAL layers
  (2 KV heads × 512) get 32-token pages.
- **`--block-size 32` (a standard vLLM flag) gives GLOBAL 64-token pages.** That is the page size our patched FA4
  block-sparse path supports (TMA path, page = tile_n).
- **Literature.** The vLLM paper (PagedAttention, SOSP'23, §7.2, Fig. 18b) varied the block size. On ShareGPT,
  16–128 give the best performance; only short sequences (Alpaca) degrade with large blocks. 16 is the default for
  fragmentation reasons, not speed.
- **Rule.** Measure vLLM dense at its default (GLOBAL 32) and at `--block-size 32` (GLOBAL 64) on the same long
  prompts.
  - If equal within noise, run dense and sparse both at GLOBAL 64 (identical configuration), and report the default
    measurement and the vLLM paper's ablation.
  - If 64 is slower, the dense arm keeps the default and we add a cp.async (page 32) block-sparse path.
- The patch does not touch dense attention (the change is inside the block-sparse branch, compile-time gated on
  paged KV). Dense run-to-run differences of about 2% seen between probe runs are noise; an earlier run had paged and
  contiguous dense equal at 2.84 ms.

## Upstream report (pending the user's go-ahead)

- **The same bug is in Dao-AILab/flash-attention `main`.** `flash_attn/cute/flash_fwd_sm90.py` has the identical
  "Block sparsity: use TMA closures directly (not paged)" branch. vLLM's fork syncs from it.
  - The right target is a PR to Dao-AILab/flash-attention `main`; optionally mirror it to vllm-project/flash-attention
    `main`. Both accept external PRs (vLLM fork: about 51 open PRs); no CLA found.
- **Contents:**
  - the 10-line fix;
  - a regression test in `tests/cute/` with paged KV and block sparsity: hd ≤ 256 upstream (SM90 caps there), page
    size = tile_n, unused pages filled with other data, compared with contiguous;
  - the reproducer and numbers in the description.
- **Effort:** about half a day (CuTe DSL is pure Python, so there is no C++ build; tests need an H100).
- **Needs:** the user's explicit approval and the GitHub account to use (public posting).

## Page-size measurement (2026-10-02, `results/.../vllm_dense_check_1002/vllm_block_size_64k.jsonl`)

vLLM 0.30.0 dense on the same 6 E14 64K prompts:
- default (GLOBAL page 32): per step 40.53 ms (median), prefill 2.42 s;
- `--block-size 32` (GLOBAL page 64): per step **39.69 ms** (−2.1%), prefill **2.29 s**.

**Decision:** run both the dense and the sparse arm at `--block-size 32` (GLOBAL page 64). It is the faster official
dense configuration, so the baseline gets stronger, and it is the page size the patched FA4 block-sparse path supports.

## Expected effect in vLLM (ESTIMATE, not a result)

Assumption: the method's absolute per-request saving in decode time S (dense S − sparse S on our substrate) transfers
unchanged to vLLM, at equal step counts. Inputs are the E14 cells' medians:

| bin | ours: P / dense S → W ratio | vLLM: P / dense S → W ratio |
|---|---|---|
| 32K | 2.03 s / 7.62 s → 0.92 | 0.89 s / 5.76 s → 0.89 |
| 64K | 5.05 s / 5.22 s → 0.89 | 2.29 s / 4.61 s → 0.83 |
| 96K | 7.20 s / 13.68 s → 0.81 | 4.05 s / 13.34 s → 0.77 |

The same saving weighs more because vLLM's prefill and non-attention parts are faster.

Risks that could shrink this:
- the sparse arm may lose part of vLLM's CUDA-graph coverage if the routing must run eagerly;
- paged sparse attention is about 3–5% slower than contiguous;
- host-side routing logic may stall vLLM's async scheduling.

Only the port measures it.

## vLLM calling convention: varlen + paged + block-sparse (2026-10-02 13:35 UTC−5)

Probe: `scripts/v27_vllm_varlen_sparse_probe.py`, raw output `results/.../vllm_port_probe_1002/varlen_probe_p64.jsonl`.
It calls `flash_attn_varlen_func` exactly as vLLM's decoder GLOBAL call does (`cu_seqlens_q`, `seqused_k`,
`block_table`, page 64, `fa_version=4`), adds FA4's varlen block-sparse lists, and uses the GLOBAL geometry (canvas 256,
16 / 2 heads, head_dim 512).

**Per-call time** (ms, median; dense is vLLM's call without lists, best of num_splits 1 / 32 / 64):

| keys | dense | all tiles kept (sparse entry) | keep 25% | keep 12% |
|---|---:|---:|---:|---:|
| 32K | 1.56 | 1.43 | 0.49 | 0.32 |
| 64K | 2.95 | 2.73 | 0.83 | 0.51 |
| 94K | 4.25 | 3.88 | 1.17 | 0.67 |

**Findings and limits:**
- The sparse entry with every tile kept is not slower than vLLM's dense call.
- Varlen block sparsity accepts only 64-row maps (`sparse_block_size[0] = q_stage × tile_m = 64`).
- **Correctness is not yet established.**
  - With every tile kept, the output matches the fp32 reference (max abs error 1e-4).
  - With 25% / 12% kept, the max abs error is 0.07–0.20, too large for bf16.
  - The fixed-length paged path was exact in `probe3`. The suspect is therefore the varlen block-sparse indexing
    (or its combination with paging).
  - `scripts/v27_vllm_varlen_sparse_isolate.py` separates fixed vs varlen and contiguous vs paged. It is queued on
    dllm after R17.
- **Isolation result (15:12 UTC−5, `vllm_port_probe_1002/varlen_isolate.out`; 32K keys, 64-row maps, keep 25%;
  reference magnitude 0.014):**

  | path | max abs err | mean abs err |
  |---|---:|---:|
  | fixed-length, contiguous | 3.2e-4 | 3.2e-5 |
  | fixed-length, paged | 3.2e-4 | 3.2e-5 |
  | varlen, contiguous | 0.121 | 0.013 |
  | varlen, paged | 0.121 | 0.013 |
  | varlen, paged, ascending lists | 0.121 | 0.013 |
  | varlen, paged, dense (no lists) | 1.5e-4 | 1.6e-5 |

  - Varlen + block-sparse is wrong regardless of paging or list order. Its mean error equals the output
    magnitude, yet it skips work (it is fast), so the lists are applied to the wrong rows, heads or blocks.
  - The cause is either our reading of the varlen list layout (`[H, total_m_blocks]` counts, `[H, total_n_blocks]`
    indices at offset `m_block × num_n_blocks`) or an FA4 bug.
  - **Cause found and confirmed (15:28 UTC−5, `varlen_isolate.out` case g):**
    - `flash_fwd_sm90.py` builds `SeqlenInfoCls = partial(SeqlenInfoQK.create, ...)` without `tile_m` / `tile_n`
      (comment: "Don't need to pass in tile_mn because we won't access offset_padded").
    - So `num_n_blocks = ceil(seqlen_k / 128)` (the default tile_n). The varlen block-sparse offset uses it:
      `block_idx_offset + m_block × num_n_blocks`.
    - At head_dim 512 the kernel runs tile_n 64. From the second q block on, the kernel therefore reads the previous
      block's list.
    - Laying our lists out at a stride of ceil(seqlen_k / 128) gives 3.2e-4 max error, exact.
    - The fix is one line: pass `tile_m` / `tile_n` to the partial. Configurations with tile_n 128 (head_dim ≤ 128)
      are unaffected, which is probably why tests miss it.
  - **Impact:** none on batch-1 work (the fixed-length path is exact); needed only for batch > 1 serving. Report it
    upstream together with the paged bug (pending the user's go-ahead).
- **The fixed-length paged path is exact, so the port uses it** (`_flash_attn_fwd` with `page_table`, 4D lists,
  batch 1, verified exact at page 64). Batch 1 is also what the panels run.

## Thin vLLM adapter for the unchanged method core (2026-10-02 13:50 UTC−5)

Code: `experiments/numerical_qk_reuse/vllm_adapter.py`. Bench: `scripts/v27_vllm_method_bench.py`.

**Principle.** The method core is not reimplemented. The adapter feeds the core
(`v21.install` → `v20.install` → `global_scope.install` → `integration.Attention` + `NativeReuseState`) the same
inputs that the HF hooks give it, from inside vLLM:

| core input (HF substrate) | vLLM source |
|---|---|
| encoder pre-hook (`invalidate`) before prefill / commit | patched `DiffusionGemmaModelState.prepare_attn` when the request is in the encoder phase |
| `State.begin(remaining_step, canvas)` before each denoising call | same patch: `48 − step_tensor[slot]` (one host read per step) |
| `State.observe_logits(processed logits)` after sampling | wrapped `_compiled_sample_step` (its temperature-scaled logits; fast T takes only the argmax) |
| decoder GLOBAL attention `(q, k = prefix ⊕ canvas, v)` | patched `FlashAttentionImpl.forward` on the 5 GLOBAL layers in denoising calls |
| encoder cache prefix (identify hook) | per layer and canvas: one contiguous K/V buffer; the prefix copied once from the pages, the canvas region refreshed every call |

- **Frozen configs validate unchanged.** It binds with the frozen R17 main-arm config, verified on CPU in the vLLM env:
  source-byte hashes and fingerprints pass, and `effective_method` is identical to the panels'.
  - The core runs from a panel deploy dir; the adapter is loaded from an overlay directory appended to the package
    path, and its sha256 is recorded per request.
  - The HF module binding (`runner._install_dense`, whose import pulls in datasets/scipy) is replaced by a stub
    module that returns the adapter's per-request binding.
  - The core's hooks are registered on stub modules (`DiffusionGemmaEncoderModel` and 30 attention stubs).
    `GlobalScopeAdapter` selects GLOBAL layers 5 / 11 / 17 / 23 / 29, as in HF.
- **Costs the method arm pays and vLLM dense does not:**
  - one host read per step;
  - one prefix copy per GLOBAL layer per canvas, plus the canvas-region refresh per call;
  - eager attention (cudagraph mode PIECEWISE instead of vLLM's default FULL decode graph).

  The dense arm is measured in both cudagraph modes, and an `allkept` arm (same buffers, FA4 with every tile kept)
  isolates the adapter's own cost.
- **Env change:** scipy 1.15.3 installed with `--no-deps` into the dyh vLLM env (numpy unchanged). It is no longer
  needed after the runner stub and can be removed.
- **Status:** CPU bind check passed. The GPU smoke (one 32K cell, all four arms) waits for R17 to free a GPU.
- **Next optimisation, once correct:** feed FA4 the paged cache plus page table directly (the patched fixed-length
  path), so the per-call canvas refresh and the per-canvas prefix copy disappear for the consumer. Only the
  observation call would still read a contiguous K.

## The strongest dense GLOBAL call, and a split consumer that matches it (2026-10-02 16:00 UTC−5)

All files are in `results/.../vllm_port_probe_1002/`. Scripts: `v27_fa4_split_sparse_bench.py`,
`v27_vllm_dyncausal_probe.py`, `v27_fa4_dyncausal_sparse_bench.py`, `v27_fa4_sparse_split_alias_bench.py`.

**1. vLLM's dense GLOBAL call is much faster than the FA4 dense configuration our panels used.**
- vLLM calls FA4 with `causal=True` plus a per-request `dynamic_causal=False` tensor and `num_splits=0` (heuristic,
  2 splits here). Captured in-engine at 35K keys: **1.00 ms**; the same call with `num_splits=1` takes 1.81 ms.
- Standalone (`dyncausal_sparse.jsonl`, dense, contiguous):

  | path | 32K | 64K | 94K |
  |---|---:|---:|---:|
  | dynamic-causal, num_splits=0 (vLLM) | 0.92 | 1.71 | 2.42 |
  | plain non-causal, num_splits=1 (our panels' dense) | 1.49 | 2.96 | 4.17 |

  The outputs are identical (same relative error vs fp32).
- Cause: on SM90, split-KV only divides the work on the dynamic-causal path. With `causal=False`, and with block
  sparsity, `num_splits > 1` re-does the whole key range or list in every split (`split_sparse.jsonl`: 4 splits ≈ 2×
  the time, 8 splits ≈ 4×). This is a candidate third upstream issue; SM100's sparse producer has a split index,
  SM90's does not.
- **Consequence for our panels.** `D_fa4_allkept` (and `dense_plain`) ran the slow configuration. Our HF-substrate
  speed ratios were therefore measured against a dense GLOBAL attention about 1.65× slower than the strongest
  official FA4 configuration, and they overstate the gain against vLLM-grade dense. The dense comparisons that
  count from now on are vLLM's own serving, or the dynamic-causal configuration.

**2. Split consumer without kernel changes (`split_alias.jsonl`).**
- S batch entries alias the same paged K/V through the page table. Each entry gets one contiguous share of every
  kept-tile list, and the partial outputs are merged exactly by LSE.
- Relative error is 0.4–0.5% (bf16 partials) vs 0.3% unsplit. The time includes the merge.

  | ms per call | 32K | 64K | 94K |
  |---|---:|---:|---:|
  | vLLM dense (dynamic-causal, auto split) | 0.96 | 1.79 | 2.53 |
  | sparse entry, all tiles kept, S=2 | 0.91 | 1.61 | 2.28 |
  | keep 25%, S=1 → S=2 | 0.46 → **0.36** | 0.83 → **0.56** | 1.14 → **0.72** |
  | keep 12%, S=1 → S=2 | 0.29 → **0.27** | 0.48 → **0.37** | 0.65 → **0.46** |

- The vLLM adapter now runs every FA4 call the core makes (consumer and dense calls through
  `v27_fa4.sparse_lists`) this way, on vLLM's own paged cache with S = 2. The routing is unchanged.

**3. vLLM smoke, LongBench-v2 32K, 2 cells per arm** (`smoke_1/2/3.jsonl`; the two cells are one item × 2 seeds;
batch 1, `--block-size 32`, memory 0.85).

| arm | per step (median, ms) | GLOBAL call (mean, ms) |
|---|---:|---:|
| vLLM dense, default cudagraphs (FULL decode) | 29.94 / 29.97 | – |
| vLLM dense, PIECEWISE | 30.16 / 30.38 | 0.90 (hooks only, `native`) |
| adapter all-kept, before the split consumer | 33.3 / 33.5 | 1.56 |
| adapter all-kept, split consumer | 30.33 / 30.52 | 0.97 |
| **method (main), split consumer** | **28.80 / 28.74** | 0.72 (includes observation calls) |

- Receipts:
  - every denoising call was observed exactly once (`order_errors` 0);
  - the core's counters show the full schedule (carried first calls, fused observations, held decisions; no
    native fallback);
  - `effective_method` is identical to the panels'.
- At 32K the method is 0.96× of vLLM's default dense per step. That is only a smoke result; 32K is the weakest
  length.

**4. vLLM smoke, LongBench-v2 64K** (`smoke_4.jsonl`; one item of 71.5K prompt tokens × 2 runs per arm; memory 0.90).

| arm | per step (median, ms) | GLOBAL call (mean, ms) | prefill (s) | steps / wall (s) |
|---|---:|---:|---:|---|
| vLLM dense, default cudagraphs | 40.18 / 40.18 | – | 2.40 | 126 / 7.28, 120 / 7.04 |
| vLLM dense, PIECEWISE | 40.78 / 40.19 | – | 2.40 | 106 / 6.53, 88 / 5.77 |
| adapter all-kept, split consumer | 40.00 / 40.12 | 1.75 | 2.40 | 76 / 5.30, 89 / 5.81 |
| **method (main)** | **34.56 / 34.03** | 0.96 | 2.40 | 94 / 5.72, 94 / 5.72 |

- **Per step, the method is 0.85× of vLLM's own default dense serving at 64K, inside vLLM.** All-kept through the
  adapter equals dense, so the adapter costs nothing measurable.
- Step counts differ between runs because vLLM cannot fix per-request seeds. Request-level ratios therefore need
  many items; per-step is the robust smoke metric.
- With equal step counts (120), an illustrative end-to-end figure is
  (2.40 + 120 × 34.3 ms) / (2.40 + 120 × 40.2 ms) ≈ 0.90.
  This is an ESTIMATE, not a measured panel ratio. Prefill is not optimized, and its share is larger in vLLM.
- **Next:** a vLLM panel on many items at 32K / 64K / 96K, with accuracy scored from the private completions.
  - Arms: vLLM dense (default) and method; all-kept on a subset.
  - Memory: 96K needs memory 0.92 and a check that the method's score cache fits.
