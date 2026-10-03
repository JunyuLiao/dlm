# V31: why "dense" trajectories differed in vLLM, and what the dense baseline is (2026-10-03)

Branch `research/vllm-paired-20261003`, worktree `E:/dlm/vllm_paired_20261003`, parent
`research/cooperative-sparsity-20261003` (V30). Evidence: `results/v31_20261003/graphmode001/README.md`.

## The confusion

V18b–V30 compared the method with two dense arms inside vLLM 0.30.0:
- `dense`: vLLM's default execution, FULL CUDA graphs for decode;
- `native`: the same official dense attention, run in PIECEWISE mode with our hooks.

`native` and every other PIECEWISE arm needed 20–30% fewer denoising forwards and produced shorter outputs than
`dense`, so `native` looked like a "faster dense". Its per-step cost was in fact equal or slightly higher.

## Root cause: an upstream vLLM bug in FULL-graph decoding of DiffusionGemma

- [vLLM PR #51994](https://github.com/vllm-project/vllm/pull/51994), merged 2026-09-30 and not in the latest
  release 0.30.0 (2026-09-22): "Fix DiffusionGemma silently freezing attention mask under CUDA graph replay".
- Mechanism:
  - `DiffusionGemmaModelState._causal_buf`, the per-request causal (commit / encoder) vs bidirectional (denoise)
    flag, was a bool tensor.
  - `FlashAttentionMetadataBuilder.build()` cast it out of place to int32 on every call.
  - FULL CUDA graphs bind the capture-time copy, so every replayed step reads a frozen mask.
- Our measurements match this:
  - With per-request reseeding, FULL and PIECEWISE are bitwise identical at the first denoising call and diverge
    from the second.
  - Over 6 cells, eager / PIECEWISE / FULL need 10.5 / 10.3 / 15.7 denoising calls per canvas.
  - On the V30 short-task panel, FULL dense has more forwards and longer outputs than every PIECEWISE arm.

## Consequences for the earlier vLLM results (V18b, V28, V29, V30)

- Every "method vs default dense" ratio there compares against a dense that runs a frozen attention mask. Those
  ratios mix a real per-step effect with the bug's extra forwards. **Do not quote them.**
- "method vs native" compared against a correct dense, but without seed pairing and with few items. The V18b
  `native` arm also had a smaller warm-up inventory. These are previews only.
- PIECEWISE arms (native, all-kept, main) were not affected. They were run correctly, though unpaired.

## Randomness is controllable

vLLM rejects per-request `seed` for diffusion models, but its sampler draws the initial canvas and all Gumbel noise
from the default torch generators. Reseeding them before each request makes batch-1 runs token-identical across
repeats; verified for 6 cells × 2 repeats in every mode. The method uses private generators, so equal seeds give
equal noise to every arm. All v31 panels use `seed = sha256(base, dataset, index, panel seed, repeat)`.

## The dense baseline from now on

**Primary: vLLM 0.30.0 default execution (FULL decode graphs) plus the exact upstream fix of PR #51994**, seed-paired.
- This is the official default configuration with a merged upstream bug fix, not a hand-made baseline.
- The fix is applied at runtime (`FIX_51994=1` in `scripts/v31_vllm_paired_bench.py`: the causal buffer is allocated
  as int32, so vLLM's own slice assignment updates it in place). No installed file is modified.
- Also reported:
  - PIECEWISE dense, no hooks: the method's own execution mode;
  - the unfixed default, to document the bug's effect.
- Optional cross-check once a GPU is free: a vLLM nightly that contains the fix on one host; then vLLM 0.31 when
  released.

## Metrics on paired requests

- Per-forward cost: S/N (decode span / denoising forwards), plus direct timing on common states.
- Forward count: N = C (canvases) × N/C (denoising forwards per canvas).
- Request time: W (with prefill) and S (generation only).
- Accuracy with the panels' unchanged LongBench-v2 scorer (`scripts/v31_score_paired.py`, on mpk).
- Summary: `scripts/v31_paired_summary.py`, with paired geometric means and item-clustered 95% CIs.

## Running now (started 2026-10-03 05:22 UTC−5)

- E14 LongBench-v2 32K + 64K items × 2 panel seeds = 96 cells, sharded across dllm / mpk / dlm2. Every arm of a
  cell runs on the same host.
  - Panel a: method (main), dense PIECEWISE, dense default (unfixed).
  - Panel b: dense default with the fix.
  - Panel c, all with the fix: dense PIECEWISE, main, main + C gate, base threshold ± C gate, +ln2 threshold
    ± C gate.
- The C gate in vLLM uses the sampler's acceptance mask, recomputed from its own logits with the official
  entropy-bound rule (`vllm_adapter.accepted_mask`, `tests/test_v31_accepted_mask.py`). Variant configs come from
  `scripts/v31_make_variant_config.py`: only `threshold_shift` / `sensitivity` change, validated by v21.
- Then on dllm: `scripts/v31_packgqa_sparse_bench.py`. This is the regroup lever: sharing K/V tiles across the 8
  query heads of a KV head (pack-GQA), which our per-head lists currently disable.

## Prior art and the MAGE baseline (added 2026-10-03 06:40 UTC−5)

- Novelty check against current preprints. The closest prior art:
  - **MAGE** ([arXiv 2602.14209](https://arxiv.org/html/2602.14209), Feb 2026): exact attention at the first step of a
    block, per-KV-head top-k (fixed budget, 512/1024 tokens) reused for the whole block. FlashInfer dense baseline;
    6.82× at 128K. This overlaps our "observe once per canvas, hold the maps" core.
  - **LoSA** ([arXiv 2604.12056](https://arxiv.org/html/2604.12056v1), Apr 2026): stable vs active query tokens by
    query change between steps; stable tokens reuse cached prefix attention. This overlaps the query-protection idea.
  - Also FlashBlock (2602.05305) and PulseCol (2605.20813).
  - None of them analyses how sparsity changes the denoising trajectory or the number of forwards.
- **MAGE port as a baseline arm** (`vllm_adapter.py` arm `mage`; tests `tests/test_v31_mage_port.py`):
  - the same paged split FA4 execution as our method; only the selection rule differs;
  - first call of each canvas: exact dense output plus MAGE eq. 5 at 64-key tile granularity;
  - later calls reuse the selection; the canvas tiles are always kept;
  - all 5 GLOBAL layers are sparsified (MAGE keeps layers 1–2 dense; no GLOBAL layer is among them).
- Panel d (all three hosts, after panel c): MAGE at k = 1024 and 4096 tokens, the same 96 cells and seeds.
- Proposed contribution framing, conditional on panels c / d:
  1. a forward-count-aware evaluation (latency = forwards × per-forward cost, with seed-paired trajectories);
  2. step-stable sparsity: query protection (C gate, collaboration with Junyu) on top of cached reuse, at higher
     sparsity;
  3. sparse execution that matches the strongest production dense path (alias split, pack-GQA).

## Panel a (seed-paired, 48 cells per length, all three hosts) — 2026-10-03 05:45 UTC−5

Reference: PIECEWISE dense, no hooks. Paired geometric means arm/ref with item-clustered 95% CIs
(`scripts/v31_paired_summary.py`; private inputs `E:/dlm/v31_private/paired/`).

| arm | bin | W | N/C (forwards per canvas) | S/N (mean per-forward) | median step | identical outputs |
|---|---|---|---|---|---|---|
| vLLM default dense (FULL, **unfixed**) | 32K | 1.185 [1.04, 1.37] | **1.192 [1.12, 1.26]** | 0.971 | – | 0/48 |
| vLLM default dense (FULL, **unfixed**) | 64K | 1.222 [1.11, 1.37] | **1.275 [1.18, 1.37]** | 0.979 | – | 0/48 |
| main (PIECEWISE) | 32K | 1.006 [0.86, 1.17] | 0.982 [0.93, 1.03] | 1.086 [1.02, 1.18] | 0.955 | 0/48 |
| main (PIECEWISE) | 64K | 1.001 [0.91, 1.10] | 1.065 [1.01, 1.13] | 0.966 [0.92, 1.04] | 0.860 | 0/48 |

**The bug at scale.** Seed-paired, the unfixed FULL default needs 19–28% more forwards per canvas than correct dense.
Its per-forward cost is 2–3% lower, from the FULL graph.

**The method against correct dense: no end-to-end gain yet.**
- A typical sparse forward (per-request median step) is 4.5% (32K) and 14% (64K) cheaper, consistent across hosts.
- The mean per-forward cost keeps little of that, because the per-canvas observation and the re-decisions are
  expensive. The fused observation kernel takes 4.2 ms per GLOBAL layer at 65K, versus about 1.7 ms for vLLM's dense
  call (`observe_split_1002`).
- At 64K the method also takes 6.5% more forwards per canvas.
- The earlier HF-substrate gains came largely from a slower dense reference. The vLLM smoke "0.85× at 64K" was a
  median step against the buggy FULL default.

**Next levers, queued or planned:**
- overhead: observe every 2nd canvas (`cc2`), re-decide every 12 calls (`r12`), 64-row maps (`q64`) — panel e;
  a cheaper observation without the projected-V mu (V term is a measured negative), timed in
  `scripts/v31_observe_cost_bench.py`;
- step inflation: the C gate at the main / base / +ln2 thresholds — panel c;
- prior art at its own budgets: MAGE k = 1024 / 4096 — panel d;
- kernel: pack-GQA — `scripts/v31_packgqa_sparse_bench.py`;
- 96K, where attention is a larger share of the step.

## Panel b (partial, 05:55 UTC−5): the backported fix works

Fixed FULL default (`FIX_51994=1`) vs PIECEWISE dense, seed-paired:

| bin | cells | N/C | S/N | identical outputs |
|---|---:|---|---|---|
| 32K | 43 | 0.988 [0.951, 1.030] | 0.976 | 16/43 |
| 64K | 15 | 1.025 [0.909, 1.154] | 0.983 | 1/15 |

- With the fix, the forward-count inflation disappears.
- The remaining divergences come from small FULL vs PIECEWISE numeric differences and are statistically
  equivalent.
- The FULL graph is 1.7–2.4% cheaper per forward.
- **The fixed FULL default is therefore the dense baseline: correct and the fastest.**
- Against it, main currently has per-forward ≈ 1.11 (32K) / 0.98 (64K) and W ≈ 1.0, so there is no end-to-end gain
  yet. The bottleneck is the per-canvas observation and the re-decisions.
- Planned kernel work: fold the observation into FA4 itself.
  - The per-(row, 64-key tile) log-mass comes from the online-softmax pass, with one extra quad reduction and one
    store per tile, only on the observation call.
  - The projected-V term becomes optional: negative in our GLOBAL-only regime, positive in Junyu's all-layer 70–75%
    regime.
  - In-kernel block statistics have precedent for prefill (Block Sparse Flash Attention, FA-2, block-max gating;
    CoSA, a separate proxy pass). Decode-time cross-step reuse in diffusion LLMs is open.

## Panels b–e complete (2026-10-03 09:50 UTC−5)

Reference: vLLM default FULL + the PR #51994 fix (`dense_default_fix`). Seed-paired, LongBench-v2 32K / 64K, 24 items
× 2 panel seeds per length; panel e has 32 of 48 cells so far (dllm shard still running). Full tables:
`results/v31_20261003/panels/panels_bcde_vs_fixed_default.md`. Accuracy: the unchanged v15 scorer
(`scripts/v31_score_paired.py`, booleans in `results/v31_20261003/panels/scores_correct_bool.json`).

| arm | 32K W | 32K N/C | 32K S/N | 32K correct | 64K W | 64K N/C | 64K S/N | 64K correct |
|---|---|---|---|---|---|---|---|---|
| dense default + fix (ref) | 1 | 1 | 1 | 28/48 | 1 | 1 | 1 | 23/48 |
| dense PIECEWISE | 1.035 | 1.024 | 1.023 | 29 | 1.021 | 1.008 | 1.017 | 25 |
| main (−ln2) | 0.966 | 1.006 | 1.018 | 35 | 0.981 | 1.072 | **0.927** | 23 |
| main + C gate | 1.020 | 1.002 | 1.101 | 31 | 1.078 | 1.038 | 1.000 | 27 |
| base threshold | 1.184 | 1.185 | 1.051 | 29 | 1.161 | 1.268 | 0.942 | 28 |
| base + C gate | 1.087 | **1.054** | 1.078 | 27 | 1.098 | **1.114** | 0.968 | 27 |
| +ln2 threshold | 1.507 | 1.587 | 1.005 | 28 | 1.437 | 1.694 | 0.898 | 25 |
| +ln2 + C gate | 1.306 | **1.358** | 1.043 | 29 | 1.247 | **1.418** | 0.930 | 24 |
| MAGE k=1024 (port) | 1.347 | 1.350 | 0.984 | **20** | 1.047 | 1.133 | 0.957 | 25 |
| MAGE k=4096 (port) | 1.045 | 1.070 | 1.041 | 31 | 0.998 | 1.068 | 0.979 | 28 |
| r12 (32 cells) | 1.133 | 1.078 | 1.001 | 23 (21) | 0.952 | 1.014 | 0.923 | 17 (17) |
| q64 (32 cells) | 1.088 | 1.052 | 1.011 | 23 (21) | 1.038 | 1.053 | 0.929 | 20 (17) |
| cc2 (32 cells) | 1.090 | 1.052 | 0.995 | 23 (21) | 1.007 | 1.063 | 0.907 | 19 (17) |
| r12q64 (32 cells) | 1.089 | 1.068 | 1.003 | 24 (21) | 0.960 | 1.064 | 0.917 | 19 (17) |

Paired geometric means arm / ref; "(ref)" in brackets = the reference's correct count on the same cells. CIs are in
the full table.

What this shows:
- **Accuracy:** no arm is below the dense reference beyond noise, except MAGE k=1024 at 32K (20 vs 28).
- **Forward count:** higher sparsity inflates denoising forwards per canvas strongly (base +19–27%, +ln2 +59–69%).
  The C gate removes most of that inflation (base 1.185 → 1.054 at 32K, 1.268 → 1.114 at 64K; +ln2 1.587 → 1.358,
  1.694 → 1.418). This is the clearest method-level effect so far (contribution 2, collaboration with Junyu).
- **Per forward:** only 64K gains (main 0.927, cc2 0.907, r12q64 0.917). At 32K no arm beats dense per forward.
- **End to end:** no arm has a significant W gain; main 64K 0.981, r12 64K 0.952 (32 cells).
- The panel-a "main" timings (S/N 1.112 / 0.985) were slower than the identical panel-c run (1.018 / 0.927) with
  identical trajectories; the panel-c run (warm caches) is the one to quote.

## Where the per-forward time goes: step profile (2026-10-03 09:30 UTC−5)

`scripts/v31_step_profile.py` brackets every GLOBAL attention call, the FA4 sparse call inside it, the K/V refresh and
the sampler hook with `torch.cuda.synchronize()` (3 requests per arm; diagnostic only: syncs expose CPU time and remove
async overlap). Per GLOBAL layer call, ms:

| 64K | total | FA4 sparse | K/V copy | rest (core) | kept tiles |
|---|---|---|---|---|---|
| vLLM dense | 1.99 | – | – | – | 100% |
| main, held call | 1.58 | 0.67 | 0.20 | 0.71 | 11% |
| main, re-decision call | 2.73 | 0.97 | 0.20 | 1.56 | 13% |
| main, first call of a canvas (carried map) | 3.13 | 0.94 | **1.20** (prefix copy) | 1.0 | 13% |
| main, observation call | **8.69** | – | 0.20 | 8.5 | – |
| main + C gate, held call | 1.85 | 0.91 | 0.21 | 0.73 | 21–28% |
| MAGE port, reused call | 1.00 | 0.45 | 0.19 | 0.36 | 1.9% |
| MAGE port, selection call (unoptimized fp32 QK) | 20.0 | – | 1.19 | – | – |

At 32K the main held call (1.41 ms, 22% kept) is more expensive than vLLM dense (1.11 ms). The FA4 sparse call has a
fixed cost of ≈0.4 ms (0.45 ms at 1.9% kept, 0.67 ms at 11%, 1.0 ms at 28%).

Per step: the sampler hook costs 0.24 ms (fast T) and 1.92 ms with the C gate. The observation step of a canvas
takes 75 ms vs ≈41 ms for a dense step.

**Efficiency work, in order of measured cost:**
1. C-gate hook: one-pass Triton row statistics (`v31_logit_stats.py`; argmax / max / logsumexp / entropy) replace
   the repeated torch passes over the 256 × 262144 FP32 logits: 2.67 → 0.35 ms in isolation; unit-tested against the
   torch path (`tests/test_v31_logit_stats.py`). Opt-in `LOGIT_STATS=fused`.
2. Observation call: the M2 compact pooled V term (`mu_mode=pooled_compact`, Fan's M2) drops the per-row projected-V
   output from the fused kernel and the 0.5 GB `mu` summary from the DP build. Panel f (`m2c`, `m2c_r12`) is running.
   Next: the FA4 in-kernel observation (per-row tile log-mass), which also removes the per-canvas prefix copy.
3. Fixed per-call cost (K/V refresh, split lists, core hooks): to be measured on the GPU timeline
   (`scripts/v31_kernel_profile.py`, CUPTI) before changing code, since syncs inflate CPU-side costs.
4. MAGE's selection should also use the in-kernel observation, so that the prior-art port is not penalized by an
   unoptimized selection pass.

## Observation-path kernels (2026-10-03 10:15 UTC−5, dllm H100)

`scripts/v31_kernel_bench.py`, GLOBAL geometry (16 q / 2 kv heads, head_dim 512, 256 canvas queries), CUDA-event
medians in ms per GLOBAL layer call; `results/v31_20261003/kernels/`.

| keys | vLLM FA4 dense | FA4 + in-kernel observation | Triton observation (mu / no mu) | DP build exact: v27 → chunked | DP build compact: v27 → chunked |
|---|---|---|---|---|---|
| 32K | 0.93 | 0.93 | 2.26 / 1.96 | 1.12 → 0.29 | 0.95 → 0.19 |
| 64K | 1.76 | 1.63 | 4.43 / 3.83 | 2.20 → 0.55 | 2.03 → 0.29 |
| 94K | 2.46 | 2.25 | 6.33 / 5.27 | 3.13 → 0.79 | 3.02 → 0.43 |

- **The FA4 observation is free:** the dense output is bit-identical to plain FA4, and the per-row tile log-mass matches
  the FP32 reference and the Triton observation to FP32 rounding (`tests/test_v31_fa4_observe.py`). Its time equals
  the dense call (pack-GQA off in both observation forms).
- **The chunked dense-prefix build is 4–7× faster** and matches the sequential build to FP32 rounding, with identical
  eligibility and bad flags (`tests/test_v31_dp_chunked.py`).
- Observation call at 64K, per layer:
  - main (exact mu): 4.4 + 2.2 ms → 4.4 + 0.55 ms;
  - compact-mu configs: 3.8 + 2.0 ms → 1.6 + 0.3 ms.
- Panel h runs every efficiency switch together (`_fast` labels) after panel g; see `scripts/v31_h_chain.sh`.

## Prior-art update (2026-10-03 10:25 UTC−5)

- **SparseD** ([arXiv 2509.24014](https://arxiv.org/pdf/2509.24014), ICLR 2026,
  [code](https://github.com/INV-WZQ/SparseD)):
  - full attention in the early denoising steps;
  - head-specific sparse patterns computed once and reused for all later steps;
  - up to 1.50× over FlashAttention at 64K with 1,024 steps.
  - It is the closest prior art to "observe once, hold the maps", next to MAGE, LoSA and PulseCol.
  - It is evaluated with a **fixed** number of denoising steps.
- **SeerAttention** ([arXiv 2410.13276](https://arxiv.org/pdf/2410.13276)) modifies the FlashAttention-2 kernel to emit
  block-level (max-pooled) attention statistics alongside the output, as training targets for its gate.
  - In-kernel block statistics are therefore not new.
  - The FA4 observation here is an engineering contribution (zero-cost observation inside the production decode
    kernel, for decode-time reuse) and must cite SeerAttention, BSFA and CoSA.
- What remains distinct, to re-check before writing:
  1. **Adaptive-step samplers.** vLLM's DiffusionGemma decodes with entropy-bound acceptance and early convergence. There,
     sparsity changes the number of denoising forwards: +19–69% per canvas at higher sparsity in panel c. The
     fixed-step evaluations of SparseD, MAGE and LoSA cannot see this. The evaluation pitfall is also concrete: the
     PR #51994 bug inflated the default dense baseline's forwards by 19–28%.
  2. **Step-stable selection.** The query-sensitivity C gate (collaboration with Junyu) removes most of that inflation
     at equal threshold.
  3. **A sparse path that matches the strongest production dense path inside vLLM.** This covers FA4 page-alias split
     sparse execution, the FA4 observation, the chunked dense-prefix scan and fused sampler-hook statistics, with
     MAGE ported onto the same execution for a fair prior-art comparison.

## Panel g: fused C-gate hook (2026-10-03 10:30 UTC−5)

Setup:
- Same configs, cells and seeds as panel c.
- `LOGIT_STATS=fused`, labels `*_fs`.
- Reference = the legacy-hook run of the same config, so these are paired ratios of fused vs legacy.

| config | bin | W | N/C | S/N | identical outputs |
|---|---|---|---|---|---|
| main + C gate | 32K | 0.967 [0.919, 1.019] | 0.994 [0.971, 1.019] | **0.952 [0.950, 0.955]** | 20/48 |
| main + C gate | 64K | 0.968 [0.907, 1.032] | 0.993 [0.971, 1.018] | **0.963 [0.959, 0.967]** | 15/48 |
| base + C gate | 32K | 0.952 [0.914, 0.979] | 0.996 [0.971, 1.014] | **0.950 [0.948, 0.952]** | 29/48 |
| base + C gate | 64K | 0.941 [0.877, 0.999] | 0.977 [0.941, 1.011] | **0.964 [0.959, 0.972]** | 18/48 |

- The one-pass statistics remove 4–5% of the per-forward cost of every C-gate arm.
- The forward count is unchanged.
- The cells that are not token-identical diverge through FP32-rounding flips of the acceptance mask or the C-gate
  sensitivity. The selector formulas are the same.
- Against the fixed dense reference, main + C gate becomes S/N ≈ 1.048 (32K) and ≈ 0.963 (64K).

## Panels h2 / i / j: every efficiency switch, fair MAGE, 96K (2026-10-03 14:30 UTC−5)

Setup:
- All switches on (`LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4`),
  overlay7. h2 is the warm repeat of h; h itself had in-request Triton recompiles from a compile-time chunk count.
- Reference: vLLM FULL + PR #51994 fix.
- Full table: `results/v31_20261003/panels/frontier_all_arms_vs_fixed_default.md`.

| arm | 32K W | 32K N/C | 32K S/N | 32K correct | 64K W | 64K N/C | 64K S/N | 64K correct |
|---|---|---|---|---|---|---|---|---|
| dense (ref) | 1 | 1 | 1 | 28/48 | 1 | 1 | 1 | 23/48 |
| main | 0.962 | 1.023 | 0.997 | 30 | 0.946 | 1.053 | 0.911 | 24 |
| main + C gate | 0.983 | 0.987 | 1.033 | 35 | 1.007 | 1.034 | 0.946 | 27 |
| r12 | 0.982 | 1.052 | 0.984 | 35 | 1.026 | 1.085 | 0.899 | 23 |
| m2c (M2 compact) | 1.020 | 1.076 | 0.965 | 31 | 0.949 | 1.056 | 0.880 | 25 |
| m2c + C gate | 1.010 | 1.020 | 0.999 | 28 | 0.974 | 1.025 | 0.907 | 23 |
| m2c r12 | 1.144 | 1.145 | 0.948 | 30 | 0.975 | 1.062 | 0.870 | 29 |
| base + C gate | 0.962 | 1.020 | 1.015 | 31 | 0.983 | 1.084 | 0.917 | 27 |
| MAGE k=4096 (same execution) | 0.978 | 1.085 | 0.920 | 29 | **0.921** | 1.088 | 0.840 | 26 |
| MAGE k=1024 (same execution) | 1.207 | 1.301 | 0.891 | **22** | 0.969 | 1.168 | 0.825 | 23 |

96K (22 cells; dense 11/22 correct):
- m2c: W 1.092, N/C 1.194, S/N 0.831, 11/22 correct.
- MAGE k=1024: W 0.834 [0.61, 1.10], N/C 1.117, S/N 0.797, 12/22 correct.
- main and r12 (exact mu) ran out of memory at MEM 0.92: the FP32 rank-32 mu summary is about 0.77 GB per GLOBAL
  layer at 94K keys, on top of the contiguous K/V buffers.

**Reading:**
- With the efficient execution applied to both, the MAGE port is cheaper per forward than every configuration of the
  method, because it keeps fewer tiles (k=4096 ≈ 6% of the prefix tiles, vs 7–11% for the method). End to end it
  is not worse.
- The method's consistent advantage is forward-count stability. With the C gate, N/C is 0.99–1.03, against 1.09
  (MAGE k=4096) and 1.17–1.30 (MAGE k=1024). The price is a higher kept fraction.
- No end-to-end ratio is significant (every W CI crosses 1).
- **Novelty status: the risk-based selector alone does not beat a fixed-budget mass top-k at this point.**
  The open question is selection quality at **matched sparsity**: fixed-budget risk top-k (`risk_topk`) vs MAGE at the
  same budget, and the C gate applied on top of MAGE.
