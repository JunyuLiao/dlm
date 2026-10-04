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

## Panel k: selection quality at a fixed budget (2026-10-03 18:05 UTC−5)

Setup:
- M2 compact mu, every efficiency switch.
- Fixed keep fraction of the prefix tiles per head and 128-row block: k12 = 12%, k5 = 5%.
- Ranking by the method's risk (V term) or by attention mass only (`risk_value='mass'`), each with and without the
  C gate.
- MAGE k=4096 keeps about 12.6% of the prefix tiles at 32K and 6.3% at 64K.
- Table: `results/v31_20261003/panels/panel_k_fixed_budget.md`.

| arm | 32K W | 32K N/C | 32K S/N | 32K correct | 64K W | 64K N/C | 64K S/N | 64K correct |
|---|---|---|---|---|---|---|---|---|
| risk k12 | 1.131 | 1.166 | 0.948 | 25 | **0.917 [0.843, 0.989]** | 1.041 | 0.881 | 25 |
| risk k12 + C | 1.079 | 1.117 | 0.953 | 26 | 0.947 | 1.036 | 0.887 | 26 |
| mass k12 | 0.981 | 1.075 | 0.954 | 32 | 0.946 | 1.053 | 0.881 | 27 |
| mass k12 + C | 0.941 | 1.026 | 0.966 | 32 | 0.989 | 1.051 | 0.885 | 27 |
| risk k5 | 1.303 | 1.320 | 0.931 | 25 | 1.036 | 1.112 | 0.863 | 26 |
| risk k5 + C | 1.293 | 1.278 | 0.938 | 25 | 0.937 | 1.047 | 0.872 | 27 |
| mass k5 | 0.983 | 1.117 | 0.942 | 31 | 0.955 | 1.070 | 0.865 | 26 |
| mass k5 + C | 1.024 | 1.122 | 0.948 | 29 | 0.979 | 1.075 | 0.869 | 28 |
| MAGE k=4096 | 0.978 | 1.085 | 0.920 | 29 | 0.921 | 1.088 | 0.840 | 26 |

Dense reference: 28/48 correct at 32K, 23/48 at 64K.

**Reading:**
- At a fixed budget, the method's risk ranking (V term) is **not better** than ranking by attention mass. At 32K it is
  clearly worse: more forward inflation (1.17 vs 1.08 at k12, 1.32 vs 1.12 at k5) and 25 vs 31–32 correct. This
  agrees with the earlier R17 result that mass alone beats the V term in the GLOBAL-only regime.
- The C gate at a fixed budget only re-ranks, and its effect is small and inconsistent (mass k12 at 32K:
  1.075 → 1.026). Its large effect in panel c came from **allocating more tiles** where queries are unstable.
- Our mass top-k and MAGE k=4096 have similar forward inflation and accuracy. MAGE is cheaper per forward: it uses one
  shared list per KV head and has no dense-prefix build or routing.
- **Status of the method contribution:** the selector itself does not improve on a fixed-budget mass top-k such as
  MAGE. The remaining method question, being tested in panel l, is whether stability-aware **budget allocation** (the
  C gate on the threshold selector) beats a uniform budget at the same realized mean kept fraction. It uses the new
  kept-fraction receipts (overlay8).

## Trajectory stability (2026-10-03 18:35 UTC−5)

`scripts/v31_trajectory_stability.py`, results in `results/v31_20261003/panels/trajectory_stability.md`.

- **Numerics alone move a request's forward count a lot.** Dense PIECEWISE vs dense FULL + fix differ only
  numerically, yet the per-cell log N ratio has s.d. 0.33 (32K) / 0.31 (64K), i.e. about ±35% per request.
  - Per-request comparisons are meaningless; only paired means over many cells are.
  - Every arm's perturbation is measured as excess over this null: sqrt(sd² − sd_null²).
- Excess perturbation and tail (p90 of the N ratio):

| arm | 32K excess | 32K p90 | 64K excess | 64K p90 |
|---|---|---|---|---|
| m2c | 0.23 | 1.68 | 0.35 | 2.17 |
| m2c + C gate | **0.12** | 1.63 | **0.16–0.22** | 1.61 |
| main | 0.27–0.38 | 1.63–1.80 | 0.26–0.27 | 1.73–1.82 |
| main + C gate | 0.19 | 1.52 | 0.25 | 1.58 |
| +ln2 | 0.54 | 3.21 | 0.52 | 3.60 |
| +ln2 + C gate | 0.42 | 2.24 | 0.35 | 2.36 |
| MAGE k=1024 | 0.59 | 3.28 | 0.58 | 3.16 |
| MAGE k=4096 | 0.42 | 2.04 | 0.45 | 1.60 |

- The C gate lowers the excess perturbation and the tail in most comparisons. m2c + C gate is the sparse arm closest
  to numerical noise.
- The bootstrap CIs over 24 items overlap, so this needs the larger confirmation set before it is claimed.
- A candidate framing for the C-gate contribution: sparsity perturbs adaptive-step denoising trajectories (mean and
  tail forward counts), and query-stability-aware budgeting reduces that perturbation.

## Panel l: adaptive (C gate) vs uniform budget at the realized kept fraction (2026-10-03 18:45 UTC−5)

Realized kept fractions come from the new receipts (overlay8):
- m2c threshold: 17% (32K) / 9.5% (64K);
- m2c + C gate: 33% / 18%.

Table: `results/v31_20261003/panels/panel_l_adaptive_vs_uniform.md`.

| comparison | kept | N/C | S/N | W | correct (dense) |
|---|---|---|---|---|---|
| 32K C gate (adaptive) | 33% | 1.025 | 1.000 | 1.004 | 29 (28) |
| 32K uniform k30 | 30% | 1.045 | 0.975 | 1.041 | 32 |
| 32K uniform k20 | 20% | 1.048 | 0.965 | 0.973 | 34 |
| 64K C gate (adaptive) | 18% | 1.028 | 0.907 | 0.976 | 23 (23) |
| 64K uniform k20 | 20% | 1.060 | 0.895 | 0.975 | 26 |
| 64K uniform k30 | 30% | 1.010 | 0.914 | 0.949 | 29 |
| MAGE k=2048 / 4096, 64K | ≈3% / 6% | 1.080 / 1.088 | 0.836 / 0.840 | 0.965 / 0.921 | 25 / 26 |

**Reading:**
- At a matched kept fraction, the C gate's adaptive allocation lowers N/C by only about 2–3 points against a uniform
  budget, at a slightly higher per-forward cost. End to end and in accuracy it is no better.
- Mean forward inflation is mostly a function of how much is kept, not of how it is allocated.
- The C gate's clearer effect is on trajectory perturbation and tails (previous section), which needs the
  confirmation set.
- Mean-W frontier at 64K: MAGE k=4096 0.92, main / m2c / k30 about 0.95. No CI excludes 1.

## Panel m: where the extra denoising steps come from (2026-10-03 19:00 UTC−5, partial)

Setup:
- `TRACE=1` receipts record the denoising steps of every canvas and the canvas mean token entropy after every step,
  i.e. the quantity the sampler compares with its confidence threshold (0.005).
- Each arm is compared with the dense trace (PIECEWISE through the adapter) on the same cells.
- Files: `scripts/v31_entropy_phases.py`, `scripts/v31_canvas_steps.py`,
  `results/v31_20261003/panels/panel_m_*.txt`.

| 64K, 32 cells | early (≥1) | mid | tail [0.005, 0.1) | conv | steps/canvas | canvases ≥ 40 steps |
|---|---|---|---|---|---|---|
| dense | 7.96 | 3.10 | 2.16 | 1.52 | 14.73 | 1 |
| m2c | 8.08 | 3.14 | **2.91** | 1.80 | 15.94 | **12** |
| m2c + C gate | 8.39 | 3.18 | 2.37 | 1.52 | 15.46 | 2 |

At 32K:
- dense 13.10 steps/canvas, 1 capped canvas;
- m2c 13.81, 6 capped;
- m2c + C gate 13.07, 0 capped.

**Mechanism:**
- Sparsity-induced forward inflation is concentrated at the end of a canvas: the tail and converging phases.
- Most of it comes from a few canvases that linger just above the convergence threshold until near the step cap.
  At 64K there are 12 capped canvases for m2c against 1 for dense. At about 30 extra steps each, they account for
  most of the +1.2 steps per canvas.
- The C gate's benefit is mainly that it prevents such stuck canvases.

Canvas step distribution:
- median 12–15, p90 23–26 steps; dense itself has 19–25% of canvases at ≥ 20 steps;
- steps after step 19 are 7.8% of all dense steps vs 11.5–14% for m2c;
- canvases ≥ 30 steps: dense 2.1%, m2c 3.9–6.5%.

**Variant under test (panel n):** a step-triggered dense rescue, `DENSE_WHEN=step:S`. GLOBAL attention runs dense
from the S-th denoising step of a canvas on, so only long canvases pay. Arms:
- m2c with S = 20 and S = 28;
- mass k5 with S = 20, to see whether the rescue unlocks higher sparsity;
- MAGE k=1024 with S = 20, to see whether the rescue generalizes to another selector;
- an entropy trigger (`conv:4`) as control.

## Confirmation set — pre-registered 2026-10-04 00:25 UTC (19:25 UTC−5), before any confirmation result

- **Cells:** the same 24 (32K) / 24 (64K) / 11 (96K) items with NEW panel seeds 3, 4, 5 (3, 4 at 96K): 166 cells
  (`cells_confirm.json` on every host, private), sharded over the three hosts.
- **Limitation:** new trajectories but not new items. The LongBench-v2 pools hold exactly 24 items per length bin.
- **Arms** (every efficiency switch on, no traces, MEM 0.92, vLLM FULL + PR #51994 fix as the reference):
  1. dense default (reference);
  2. m2c + step-20 dense rescue (`DENSE_WHEN=step:20`): the primary candidate;
  3. m2c (ablation of the rescue);
  4. mass k5 + rescue (a budget matched to MAGE k=4096 at 64K);
  5. MAGE k=4096 (prior art at its best development point);
  6. MAGE k=4096 + rescue (does the rescue generalize?).
- **Primary metrics:**
  - W, S/N and N/C as paired geometric means with item-clustered CIs;
  - accuracy (v15 scorer);
  - trajectory stability (excess s.d. over the dense numerical null is not available here: no PIECEWISE dense arm,
    so only N/C tails are reported).
- Script: `scripts/v31_q_confirmation_chain.sh`. Development evidence behind the choice: panels m and n.

## Panels n and p (development, 32 cells per length on dllm + mpk) — 2026-10-03 19:45 UTC−5

**Step-triggered dense rescue (panel n).** GLOBAL attention runs dense from step S of a canvas on.

| arm (dllm+mpk, 32 cells) | 32K N/C | 32K S/N | 32K W | 64K N/C | 64K S/N | 64K W |
|---|---|---|---|---|---|---|
| m2c | 1.058 | 0.965 | 0.990 | 1.073 | 0.881 | 0.972 |
| m2c + rescue S=20 | **1.020** | 0.977 | 0.945 | **1.035** | 0.896 | 0.958 |
| MAGE k=4096 | 1.055 | 0.920 | 0.961 | 1.109 | 0.838 | 0.924 |

On the mpk shard (16 cells per length):
- The rescue removes the capped canvases (2/2 → 0/0) and brings steps per canvas back to dense level, with 6.7% of
  GLOBAL calls dense.
- S=28 is too late (64K N/C 1.065).
- An entropy trigger (dense once the canvas mean entropy is below 4× the threshold) costs more (11% dense calls,
  64K S/N 0.929) and helps less (64K N/C 1.045). The step trigger is the right form.
- On MAGE k=1024 the rescue helps partially: 32K N/C 1.238 → 1.153, W 1.199 → 1.017. MAGE's inflation is mostly a
  uniform slowdown of early and mid denoising (panel m), not stuck canvases.

**Regrouping (panel p, chw/value_aware q64r).** The regroup diagnostic records the kept prefix-tile fraction of the
same need matrix at different row granularities:

| granularity | 32K | 64K |
|---|---|---|
| 128-row worst row (executed) | 16.1% | 8.4% |
| natural 64-row halves | 12.6% | 6.6% |
| 64-row regrouped by need count | 11.9% | 6.2% |
| 64-row regrouped by random projection | 13.3% | 6.9% |
| per row (ideal) | 2.9% | 1.4% |

- Executed q64r keeps 25–28% fewer tiles than m2c (13.0% / 6.9% vs 17.2% / 9.5%), but per-forward cost does not drop:
  S/N 0.975 / 0.893 vs 0.969 / 0.884.
- The sparse calls are already cheap, and the row permutation, output scatter and 64-row list refinement eat the
  saving. Forward inflation rises slightly (32K N/C 1.085 vs 1.026).
- **Regrouping does not pay at FA4's 64-row granularity.** The real headroom is per row (4× fewer tiles than 64-row),
  which the head_dim-512 FA4 kernel cannot exploit.
- C gate + regroup (mpk shard):
  - realized kept falls from 32.9% / 17.7% to 27.1% / 13.9%;
  - S/N is again slightly higher (1.015 / 0.924 vs 0.995 / 0.908);
  - regrouping by need count (25.7% at 32K) is barely better than the natural 64-row halves (26.1%): unstable rows do
    not concentrate.
  - The C gate's cost comes from the 128-row worst-row aggregation (per row: 8.0% vs 31.6% at 128 rows). SM90 wgmma
    has M = 64 minimum, so FA4 at head_dim 512 cannot use a finer row granularity. Regrouping is closed for this
    kernel; a row-granular decode kernel is future work.

## Confirmation set, interim: 2 of 3 shards (dllm + mpk), 2026-10-04 02:15 UTC

`results/v31_20261003/panels/confirmation_q_partial_dllm_mpk.md`. New seeds 3–5; 48 cells at 32K / 64K, 15 at 96K.
The dlm2 shard is pending.

| arm | 32K W | 32K correct | 64K W | 64K correct | 96K W | 96K correct |
|---|---|---|---|---|---|---|
| dense (ref) | 1 | 29/48 | 1 | 28/48 | 1 | 8/15 |
| MAGE k=4096 | **0.928** [0.84, 1.03] | 33 | **0.882** [0.82, 0.95] | 27 | **0.697** [0.58, 0.83] | 10 |
| MAGE k=4096 + rescue | 0.925 | 32 | 0.905 | 27 | 0.838 | 9 |
| m2c | 1.006 | 30 | 0.914 [0.84, 0.99] | 25 | 0.751 [0.63, 0.90] | 6 |
| m2c + rescue | 0.983 | 31 | 0.885 [0.80, 0.97] | 24 | 0.866 | 7 |
| mass k5 + rescue | 1.072 | 31 | 0.945 | 28 | 0.705 | 9 |

**Interim reading:**
- With the same efficient execution, sparse GLOBAL attention gives a significant end-to-end gain against the official
  fixed FULL-graph dense at 64K (≈12%) and 96K (≈25–30%). This holds for MAGE and for m2c.
- MAGE k=4096 is the best or tied arm at every length, with accuracy at or above dense.
- The step-20 rescue lowers m2c's forward inflation on new seeds (64K N/C 1.040 → 0.987), but its W benefit is not
  consistent (better at 32K and 64K, worse at 96K), and it does not help MAGE.
- m2c accuracy at 64K / 96K is 3–4 / 1–2 items below dense: within noise, but in the wrong direction.

## RULER 32K / 64K accuracy on a new pool, interim (2 of 3 shards, 174 cells) — 2026-10-04 02:45 UTC

Pool:
- `ruler_long_v31` on dllm: the pinned RULER checkout and DiffusionGemma tokenizer, a new generator seed 4242;
- 13 tasks × 10 samples per length;
- private gold.
- Scorer: `scripts/v31_score_ruler.py`, the official RULER metrics; correct = 100. RULER answers are a single short
  canvas, so this panel is accuracy only.

| arm | 32K | 64K | total | lost / gained vs dense | McNemar p |
|---|---|---|---|---|---|
| dense (ref) | 72/87 | 68/87 | 140/174 | – | – |
| m2c (± rescue) | 69 | 66 | 135 | 8 / 3 | 0.23 |
| MAGE k=4096 (± rescue) | 65 | 64 | 129 | 13 / 2 | **0.007** |

- Head to head, m2c alone is correct on 6 cells and MAGE alone on 0 (p = 0.031).
- By task:
  - all eight needle tasks stay at or near 100% for every arm;
  - both sparse methods fail **common-words extraction** (cwe: dense 10/14, m2c 2/14, MAGE 1/14), an aggregation
    over the whole context;
  - MAGE additionally loses multi-hop QA with distractors (qa_1 + qa_2: dense 15, m2c 15, MAGE 12) and multi-value.
  - vt is 0 for every arm, a model limit.
- Reading:
  - The method's worst-row selection keeps the QA evidence that MAGE's mean-over-queries fixed top-k drops.
  - Aggregation tasks are a shared weakness of sparse GLOBAL attention here.
  - This is the first accuracy result that separates the method from MAGE; the dlm2 shard completes it.

## Confirmation set and RULER, complete (all three shards) — 2026-10-04 03:35 UTC

LongBench-v2 confirmation (`results/v31_20261003/panels/confirmation_q_full.md`; 72 / 72 / 22 cells, new seeds):

| arm | 32K W | 32K correct | 64K W | 64K correct | 96K W | 96K correct |
|---|---|---|---|---|---|---|
| dense (ref) | 1 | 45/72 | 1 | 40/72 | 1 | 11/22 |
| MAGE k=4096 | **0.960** | 49 | 0.887 [0.82, 0.96] | 42 | **0.750** [0.62, 0.89] | 13 |
| MAGE k=4096 + rescue | 0.962 | 50 | 0.898 | 40 | 0.863 | 11 |
| m2c | 1.035 | 43 | 0.904 [0.84, 0.97] | 38 | 0.839 [0.73, 0.96] | 8 |
| m2c + rescue | 1.022 | 45 | **0.870** [0.80, 0.94] | 38 | 0.907 | 9 |
| mass k5 + rescue | 1.098 | 44 | 0.959 | 42 | 0.744 | 12 |

RULER, complete (260 cells; `results/v31_20261003/panels/ruler31_scores_correct_bool.json`):

| arm | 32K | 64K | lost / gained vs dense | McNemar p | cwe | qa (1+2) | multivalue |
|---|---|---|---|---|---|---|---|
| dense | 109/130 | 106/130 | – | – | 16/20 | 25/40 | 14/20 |
| m2c (± rescue) | 106 | 100 | 13 / 4 | 0.049 | 4/20 | 26/40 | 16/20 |
| MAGE k=4096 (± rescue) | 103 | 99 | 17 / 4 | 0.007 | 3/20 | 23/40 | 16/20 |

**Correction of the interim RULER reading:**
- With the third shard, m2c vs MAGE is no longer significant: 6 vs 2 cells correct by one only, p = 0.29.
- Both sparse arms lose significantly against dense, and almost all of it is **common-words extraction (cwe)**, an
  aggregation over the whole context. Without cwe the arms are equal: dense 199/240, m2c 202, MAGE 199.

**Overall status:**
1. Sparse GLOBAL attention gives significant end-to-end gains against the official fixed FULL-graph dense at 64K and
   96K.
2. MAGE (fixed shared budget) and the method are comparable in speed and accuracy. MAGE is faster at 32K / 96K, the
   method with rescue at 64K.
3. The accuracy weakness common to both is aggregation, where attention is diffuse.

**Next (in progress):**
- Panel s: a two-level selection that adds critical tiles to MAGE's budget. τ = 0.05 adds about 2% of tiles and
  recovers nothing (RULER 130/174 vs MAGE 129/174), so the QA losses are not single missed tiles.
- Panel t: RULER at matched kept fractions.
- Next variant: a coverage-adaptive budget (keep the fewest tiles covering p of each row's or head's mass), which
  targets the diffuse-attention failure mode.

## Correction: realized-sparsity accounting, and RULER at matched compute (panels t, u, v) — 2026-10-04 04:20 UTC

**Accounting bug (found by a code audit, confirmed on the receipts).**
- The method's bootstrap dense GLOBAL call reaches FA4 as an all-kept block list (`v27_fa4.dense`). The legacy
  `kept_prefix_fraction` counted it as 100% kept, against its own docstring.
- RULER answers are one canvas of about 5 to 8 steps, so this dominated the RULER receipts. For example, k12 was
  reported at 0.35 kept, but its genuinely sparse calls keep 0.12, which is what `risk_topk=k12` specifies.
- On multi-canvas LongBench requests the effect is smaller, but the earlier m2c kept fractions there are also
  upper bounds.
- The MAGE kept fractions were correct: its selection call bypasses the lists.
- Fixed in commit 78644966f. The receipts now add:
  - `sparse_kept_prefix_fraction`: all-kept lists excluded;
  - `global_prefix_work_fraction`: every GLOBAL call, with each dense step at full cost. This is the
    compute-matched axis for arms that take different numbers of dense steps.
- The method takes two exact steps per canvas (bootstrap and observation). MAGE takes one (selection at step 0).
- `scripts/v31_ruler_compare.py` reconstructs both quantities for older records from the call counters. This is
  exact for single-canvas RULER answers.
- Other fixes in the same commit:
  - MAGE coverage / critical now refuse `MAGE_SELECT=torch`; they were silently ignored before.
  - The bench records `dense_when`, `mage_critical` and `mage_coverage`.

**RULER, 174 common cells (shards 0 and 1 of the v31 pool), compute-matched**
(`results/v31_20261003/panels/ruler_rstuv_compare.md`). "work" is the GLOBAL prefix attention work over dense.
"sparse kept" covers the sparse calls only.

| arm | correct | vs dense FULL lost/gained (p) | cwe | sparse kept | work |
|---|---|---|---|---|---|
| dense FULL (ref) | 140 | – | 10/14 | – | 1 |
| MAGE k=4096 | 129 | 13/2 (0.007) | 1/14 | 0.084 | 0.401 |
| MAGE k=6144 (panel t) | 130 | 12/2 (0.013) | 2/14 | 0.125 | 0.430 |
| MAGE at 0.347 kept per length (panel v: 11136 / 22528 tokens) | 136 | 6/2 (0.29) | 7/14 | 0.347 | 0.575 |
| MAGE coverage p=0.90 (panel u) | 137 | 4/1 (0.38) | 8/14 | 0.587 | 0.732 |
| MAGE coverage p=0.95 (panel u) | 141 | 3/4 (1.0) | 11/14 | 0.750 | 0.840 |
| m2c (risk threshold) | 135 | 8/3 (0.23) | 2/14 | 0.052 | 0.372 |
| m2c k12 (risk) (panel t) | 132 | 10/2 (0.039) | 2/14 | 0.121 | 0.421 |
| m2c k12 mass (panel t) | 138 | 5/3 (0.73) | 5/14 | 0.122 | 0.428 |

**Reading (exploratory, on reused cells)**
- At matched total GLOBAL work (≈0.43), m2c k12 mass beats MAGE k=6144 head to head: 8 cells correct by it alone,
  0 by MAGE (p = 0.008).
- MAGE reaches the same accuracy only at 0.58–0.84 of dense work.
- cwe (aggregation) needs most of the context: only the coverage budget, at near-dense work, restores it.
- Caveats:
  - These 174 cells were reused across adaptively chosen variants (k5–k30, risk vs mass, MAGE options), so the
    p-value is not a confirmatory test.
  - No dense PIECEWISE RULER arm exists yet, so the trajectory-noise level of discordant pairs is unknown.
  - The method selects at step 1 after two exact steps, MAGE at step 0. Matched work does not separate selection
    timing from the selection rule.
- Hence the pre-registered confirmation below.

## Pre-registered held-out confirmation: panel w (registered 2026-10-04 04:20 UTC, before any w result)

**Pool.** `ruler_long_v32` on mpk: same pinned RULER checkout, task set and generator as v31, new seed 5353.
- 13 tasks × 15 samples per length (32K, 64K): 390 cells, panel seed 1, one repeat.
- Datasets `ruler32k_v32` / `ruler64k_v32`; private gold.
- Sharded 2 ways (mpk 0/2, dlm2 1/2). dllm is retired.

**Arms.** Each runs one fresh engine per arm with the same per-request seeds. All sparse arms use the efficient
execution (`LOGIT_STATS=fused DP_BUILD=chunked OBSERVE=fa4 KV_COPY=triton MERGE=triton MAGE_SELECT=fa4`) and the
fixed vLLM (`FIX_51994=1`).

| # | arm | role |
|---|---|---|
| 1 | dense FULL (default graph) | reference |
| 2 | dense PIECEWISE | noise floor (trajectory noise of discordant pairs) |
| 3 | MAGE k=6144 | primary comparator (matched work ≈0.43 on v31) |
| 4 | m2c k12 mass | primary candidate |
| 5 | MAGE k=4096 | standard MAGE budget |
| 6 | m2c (risk threshold, default) | the method's default selector |
| 7 | MAGE k=6144, `MAGE_STEP=1` | selection-timing control (select from step-1 queries after one exact step) |

**Primary test.** Arm 4 vs arm 3: exact two-sided McNemar on the 390 cells, α = 0.05.
- Reported with both arms' `global_prefix_work_fraction` (exact receipts).
- If the two work fractions differ by more than 0.03, the comparison is reported as not compute-matched.

**Secondary (descriptive, no multiplicity claim)**
- Each arm vs dense FULL.
- Dense PIECEWISE vs dense FULL discordance.
- Arm 7 vs arm 3.
- Arm 6 vs arm 5.
- cwe / qa / multivalue by arm.

**Rule.** No arm, budget or option is added to panel w's analysis after this registration. A new variant needs a
new pool.

## Panel w result: the pre-registered primary test is NOT confirmed — 2026-10-04 05:30 UTC

Held-out RULER v32 (seed 5353), 390 cells, 7 arms, all complete; scored on mpk with the official metrics.
Full tables: `results/v31_20261003/panels/panel_w_heldout_confirmation.md`. "Work" is exact, from the new receipts.

| arm | correct | vs dense FULL lost/gained (p) | cwe | without cwe | sparse kept | work |
|---|---|---|---|---|---|---|
| dense FULL (ref) | 323 | – | 19/30 | 304/360 | – | 1 |
| dense PIECEWISE | 323 | 0/0 (1.0) | 19/30 | 304/360 | – | 1 |
| m2c k12 mass | 310 | 18/5 (0.011) | 9/30 | 301/360 | 0.122 | 0.438 |
| MAGE k=6144 | 308 | 18/3 (0.001) | 5/30 | 303/360 | 0.126 | 0.433 |
| MAGE k=4096 | 307 | 20/4 (0.002) | 3/30 | 304/360 | 0.084 | 0.407 |
| MAGE k=6144, select at step 1 | 304 | 22/3 (<0.001) | 4/30 | 300/360 | 0.126 | 0.598 |
| m2c (risk threshold) | 299 | 26/2 (<0.001) | 1/30 | 298/360 | 0.053 | 0.389 |

**Primary test (as registered).**
- m2c k12 mass vs MAGE k=6144 at matched work (0.438 vs 0.433; within the 0.03 rule): 310 vs 308.
- Discordant 12 / 10, exact McNemar p = 0.83. **Not confirmed.**
- The exploratory 8 / 0 (p = 0.008) on the reused v31 cells did not replicate. It is best read as selection on reused
  cells (winner's curse).

**Secondary (descriptive).**
- **Dense FULL and dense PIECEWISE agree on all 390 cells.** Short RULER answers carry no trajectory noise at the
  correctness level, so every discordance between a sparse arm and dense comes from the attention approximation.
- **Every sparse arm loses significantly against dense, and the whole deficit is cwe.** Without cwe all arms are
  within 6 cells of dense (298–304 / 360). cwe: dense 19/30, sparse 1–9/30. This replicates the v31 finding: the
  aggregation failure is shared by MAGE and the method.
- **Selection timing does not help MAGE.** Step-1 selection gives 304 vs 308 (p = 0.42), at 0.60 work, because RULER
  canvases are only a few steps long and the extra exact step dominates.
- **The method's default (risk threshold) is not better than MAGE k=4096.** It scores 299 vs 307 (13 / 5,
  p = 0.096), at slightly less work (0.389 vs 0.407).

**Consequences.**
- On RULER at matched compute, the method's selector and MAGE are statistically indistinguishable. The accuracy
  lever is the diffuse-attention (aggregation) regime, not which concentrated tiles are kept.
- This is what the exploratory panels y (pooled residual for dropped tiles) and z (dropped-mass guard) target. Any
  winner from x / y / z needs its own pre-registered confirmation on a fresh pool (v33) and LongBench seeds.
- No novelty claim rests on selector quality for RULER.

## Re-scoring RULER with the official metric: earlier conclusions revisited — 2026-10-04 06:40 UTC

**The scoring rule mattered.** RULER's official score is the per-sample metric (0–100) averaged per task, then over
the 13 tasks. The multi-answer tasks give partial credit: cwe scores 70 for 7 of 10 words. Our McNemar analyses used
a STRICT boolean (score = 100 and a stop/eos finish), which turns every partially correct cwe answer into a failure.
Both are now reported. Tables:
- `results/v31_20261003/panels/ruler_v31_official_vs_strict.md`: all v31-pool arms, 174 common cells, exploratory;
- panel w (held-out v32) below.

**Held-out v32 (panel w, 390 cells), official score.** All differences are per cell vs dense FULL.

| arm | official | cwe | diff vs dense [95% CI] |
|---|---|---|---|
| dense FULL / PIECEWISE | 85.8 | 84.7 | – |
| m2c k12 mass | 85.5 | 88.3 | −0.29 [−1.65, +1.04] |
| MAGE k=6144 | 84.7 | 77.7 | −1.12 [−2.42, +0.10] |
| MAGE k=4096 | 84.1 | 67.3 | – |
| m2c (risk) | 84.0 | 71.7 | – |

- k12 mass − MAGE 6144 at matched work: +0.82 [−0.37, +2.01]; better/worse cells 23/10, sign test p = 0.035.
- On cwe the same difference is +10.7 [+5.3, +16.3] (p = 0.004).
- This is POST HOC: the registered metric was strict correctness. The next confirmation registers the official
  score as primary.

**v31 pool (exploratory), official score; dense 83.8.**

| arm | work | official diff |
|---|---|---|
| m2c (risk) | 0.372 | −0.66 |
| MAGE 2048 | 0.372 | −5.20 |
| MAGE 4096 | 0.401 | −3.25 |
| MAGE 4096 + critical tiles (tau 0.02) | 0.410 | −1.55 |
| MAGE 6144 | 0.430 | −2.27 |
| m2c k12 mass | 0.428 | +0.78 |
| m2c k20 mass | 0.485 | +0.98 |
| coverage 0.90 | 0.73 | +0.29 |
| coverage 0.95 | 0.84 | +0.72 |

- The pooled (centroid) residual lowers every arm by 0.4–1.6 points.

**Earlier conclusions, revisited.**
- "m2c vs MAGE not significant (strict p = 0.29)": the strict rule understated the method. Under the official score
  the method is 2.6–4.5 points above MAGE at matched work on v31 (exploratory), and +0.8 on held-out v32 (post hoc,
  CI includes 0). To be confirmed with the official score registered.
- "Critical tiles recover nothing (panel s)": wrong under the official score. tau = 0.02 recovers 1.7 of MAGE's 3.25
  points (cwe 63 → 75) at +0.009 work. Revived; an arm-agnostic version is being built.
- "Pooled residual (panel y)": negative under both rules. Rejected as implemented (uncalibrated centroid). Calibrated
  and sampled versions are being built.
- "Selection timing (MAGE step 1)": +1.9 on v31 but at 0.58 work, and no gain on v32. Stays rejected.
- Selector granularity (panel x ladder), at step 1: per-block max beats KV-head mean by about 1 point. This is mild
  support for the method's granularity.
- Unchanged:
  - LongBench-v2 conclusions: single-choice, no partial credit;
  - regroup: closed on cost;
  - LLaDA2.1: Amdahl;
  - prefill / per-step / N-C breakdowns: timing only.

## Final evaluation suite (decided with the user, 2026-10-04 06:55 UTC)

Criteria: recent; long where possible; covering the benchmarks of 2025–26 DLM efficiency work; convincing in the
paper. LongBench v1 is NOT included: it is older, mostly under 32K tokens, and its open-ended generation role is
covered by MRCR / GraphWalks. Add a small v1 subset only if a line-by-line comparison with MAGE's v1 numbers is
required.

| group | dataset | lengths | protocol / metric | role |
|---|---|---|---|---|
| long-context (main) | LongBench-v2 (ACL'25) | official: all 503 items, official middle truncation to the 128K-class budget; analysis: natural-length bins 32K / 64K / 96K / 128K | accuracy (official extraction), easy/hard and length breakdowns | realistic deep-reasoning QA; comparable with the leaderboard and prior work |
| | RULER (13 tasks) | 32K / 64K / 128K, fresh pool v33 (seed 6464, 15 per task) for the final numbers | OFFICIAL score (per-task mean, partial credit) primary; strict all-correct secondary (McNemar) | the DLM sparse-attention standard (SparseD, PulseCol, BA-Att), at longer lengths |
| | OpenAI MRCR 2-needle (2025) | 32K / 64K / 128K natural | official SequenceMatcher ratio (prefix check) | precise retrieval among distractors in long multi-turn input; open-ended generation |
| | OpenAI GraphWalks (2025) | 32K / 64K / 128K natural | official set F1 of the final answer | aggregation of scattered facts (the cwe-type weakness) |
| long-generation guards | AIME26 (2026) | short prompt, 8K thinking budget | exact numeric | newest math reasoning, low contamination |
| | HumanEval | short prompt, 8K budget | pass@1 (official tests, sandbox) | the standard code guard of DLM papers |

- Speed is reported on LongBench-v2 (official protocol and natural bins) and RULER, per length: end-to-end time,
  prefill, per-step cost, steps per canvas, canvases.
- All final numbers use fresh seeds or pools and a pre-registration.
- Scoring definitions are under audit against the official implementations (report pending).

## Where the method's extra per-step cost comes from (no-sync GPU-event profile) — 2026-10-04 08:10 UTC

Panel pf, profile part: CUDA events around every GLOBAL call, no synchronization. LongBench-v2 64K, 3 cells per host
on mpk and dlm2. The warm-up request is excluded (it carries JIT compiles). GPU time per GLOBAL call:

| call kind | m2c | m2c + C gate | MAGE k=4096 |
|---|---|---|---|
| held / reused sparse call | 0.334 ms (FA4 0.286) | 0.523 ms (FA4 0.476) | **0.159 ms (FA4 0.135)** |
| observation / MAGE selection (once per canvas) | 2.51 ms | 2.49 ms | 2.28 ms |
| re-decision (dp_route) | 1.035 ms (FA4 0.405) | 1.139 ms | – |
| first call of a canvas (carried map) | 0.682 ms | 0.816 ms | – |
| GLOBAL total per step (5 layers, mean) | **2.96 ms** | 3.65 ms | **1.56 ms** |

**Reading.**
- The method spends about 1.4 ms more per step on GLOBAL attention than the MAGE port. That matches the end-to-end
  per-step gap (1.2–1.5 ms).
- Two sources:
  1. The FA4 part of a held call is 2.1× MAGE's, about +0.9 ms per step, although the kept fraction is only about
     1.5× MAGE's.
  2. Re-decisions add about +0.6 ms per step.
- Hypothesis for (1): the method's keep lists differ per query head, so the 8 query heads of a KV head do not share
  K/V tile reads. MAGE's lists are identical across those heads and reuse them through L2. The kernel audit tests
  two fixes:
  - GQA-packed execution over the union of the heads' lists. This is a selection superset, so it needs an accuracy
    check.
  - A shared traversal order. This leaves the tile set unchanged.

**Other diagnostics from panel z.**
- The adapter's all-kept path costs 0.984× vLLM's own dense attention per step. The generic adapter path is not the
  overhead.
- Query drift between consecutive steps: only 11–12% of rows change by ≤10%, and about 0.2% of 128-row blocks do.
  Reusing attention OUTPUTS across steps is not viable. Reusing the SELECTION across steps is, because the tile-level
  pattern is stable (this is what the method does).
- Dense determinism:
  - dense PIECEWISE re-run in a separate engine with the same seed reproduces all 12/12 outputs;
  - dense FULL vs dense PIECEWISE with the same seed agree on 6/12 at 64K. This is a deterministic numerical
    difference between execution modes; it appears only at long contexts (24/24 identical at short contexts).
- Drop guard (DROP_GUARD) is rejected under the official score:
  - MAGE: −3.33 (guard 0.5) and −4.20 (0.3) vs −3.25 plain, at 0.50 and 0.68 work vs 0.40;
  - m2c k12 mass: +0.37 and +0.66 vs +0.78, at higher work.

## Group-shared selection and the MAGE first-call carry (branch `research/v31-group-shared-select-20261004`) — 2026-10-04 09:00 UTC

**Why.** The no-sync profile (section above) puts most of the method's extra per-step cost in the held calls: their
FA4 part costs 2.1× MAGE's for about 1.5× the kept tiles. The leading explanation is that the method keeps a
different tile list per query head. The 8 query heads of a KV head then do not share K/V tile reads, whereas MAGE's
lists are identical inside a group and reuse them through L2. The kernel audit (branch
`research/v31-kernel-audit-20261004`) attacks this at execution time (`SPARSE_GQA=align|union`). This branch attacks
it at selection time, keeping each head's budget.

**What (two opt-in options on the MAGE port; the ladder's other rungs are unchanged).**
- `MAGE_GRAN=kvblock_max`
  - Per (KV head, 128-row block), rank prefix tiles by the max, over the block's rows AND the group's 8 query heads,
    of the row's share of its prefix mass. This is the method's `risk_value='mass'` statistic from the exact FA4
    observation.
  - Keep the top k tiles. The set is shared by the group's heads.
  - The per-head budget is unchanged, so the work is the same as `qblock_max`. Only the list structure changes.
- `MAGE_GRAN=kvhead_max`
  - The same max share over all canvas rows: one set per KV head for both blocks.
  - This is exactly MAGE's list structure and cost, with the max-share statistic instead of MAGE's mean mass.
- `MAGE_CARRY=1` (needs `MAGE_STEP >= 1`): the method's `carry_first` on the MAGE port.
  - Canvas call 0 runs on the layer's selection from the previous canvas instead of exact attention. Tiles wholly
    in that canvas's prefix keep their decision; every newer tile is kept.
  - It applies only as the direct continuation: the previous canvas, the prefix grown by exactly that canvas, and the
    same row-block layout. Otherwise call 0 stays exact.
  - It removes the extra dense call that made the step-1 ladder rungs cost 0.58 of dense work.
- Tests:
  - `tests/test_v31_group_select.py` (GPU): shared sets, budget, needle rows, the carried map against the expected
    map, each invalid-carry case, and option validation.
  - A CPU logic check with stubbed FA4 entry points passed on mpk before the GPU queue.

**Panel gs (overlay `ov_gs`, adapter e8391eb210c7; EXPLORATORY; queued after panel vk on mpk and dlm2).**
1. The overlay's GPU test suite. The panel aborts unless every test passes.
2. RULER v31 pool, panel x cells (mpk shard 1/3, dlm2 0/3), keep 12% per unit, selection at step 1:
   - a control rerun of `qblock_max`;
   - `kvblock_max` and `kvhead_max`;
   - with carry: `qblock_max`, `kvblock_max` and MAGE's own `kvhead`.
3. Per-call GPU cost (`PROFILE_MODE=events`, 3 LongBench-v2 64K cells per host): `qblock_max` vs `kvblock_max`,
   both with carry.
4. End to end on the LongBench-v2 64K + 96K confirmation cells (pf's cells and shards):
   - a fresh dense FULL reference on the same host;
   - `qblock_max` and `kvblock_max`, both with carry.

**Read-out rules, stated before the data.**
- **Control.** The control must reproduce panel x's `qblock_max` step-1 records token for token on each host.
  Otherwise the overlay changed the code path and nothing else in the panel is read.
- **Accuracy.** Official RULER score, paired per cell against the `qblock_max` control:
  - group sharing "keeps the method's gain" if the mean difference is within ±1 point;
  - it "loses it" if it falls toward MAGE's `kvhead` rung (about −1).
  - The carry is read the same way against the same unit without carry.
- **Speed.**
  - The hypothesis holds if a `kvblock_max` held call costs clearly less on the GPU than a `qblock_max` held call
    at the same kept fraction. The kernel audit's `corr0` vs `indep` patterns give the kernel-only expectation.
  - End to end, W, S/N, N/C and C are reported against the same-host dense run.
- **Next step.** Any winner here is exploratory. It enters the integration branch and the pre-registered
  confirmation (fresh RULER v33 pool and official LongBench-v2), not the paper directly.

**The same decision inside the method (`RISK_GROUP=kv`, added 09:05 UTC; module `v31_group_select.py`).**
- What it does:
  - On a fixed-fraction method config (m2c k12 mass), the core's per-(query head, block) top-k
    (`v27_dense_prefix.topk_skip`) is replaced at install time by `grouped_topk_skip`. The frozen core files are
    untouched.
  - One decision per (KV head, block): rank by the max over the group's heads of the worst-row value. A tile can
    be dropped only if every head of the group may drop it (eligible and finite).
  - The same (1 − keep) fraction is dropped, so each head keeps as many tiles as before. The canvas (tail) tiles
    keep the core's per-head decision.
  - The adapter refuses configs where the option would be silently ignored: no `risk_topk`, a `risk_budget`, or
    64-row blocks.
- Tests: `tests/test_v31_risk_group.py` (CPU), 6/6 on mpk.
  - Identical heads reproduce the per-head rule exactly.
  - One shared set per group, with the per-head drop count unchanged.
  - A needle in one head is kept for its group.
  - A tile that any head must keep is never dropped.
  - Padded rows never decide.
- **Panel gs2** (after gs, both hosts):
  - RULER v31 x cells: m2c k12 mass as a control on this overlay, and with `RISK_GROUP=kv`;
  - per-call profile on 3 LongBench-v2 64K cells;
  - LongBench-v2 64K + 96K end to end.
  - The read-out rules are the gs rules, with the control as the reference.
- On mpk, the kernel microbenchmark (vkb) now runs after gs2.

## Step inflation measured on the same canvases (branch `research/v31-forced-canvas-20261004`) — 2026-10-04 09:20 UTC

**Why.** In panel pf (72 LongBench-v2 64K cells, both hosts) the per-step cost S/N is measured to ±0.5%:
- MAGE k=4096: 0.844 [0.840, 0.848];
- m2c: 0.883 [0.878, 0.887].

Steps per canvas N/C is the dominant noise:
- MAGE: 1.075 [1.03, 1.13];
- m2c: 1.048 [1.00, 1.09].

At 96K (22 cells) the N/C intervals even reach below 1 (MAGE 0.92 [0.82, 1.01]). That is illogical for a sparsity
effect: free-running arms diverge after the first differing token and then denoise different texts.

The output length also differs wildly per cell. For example, dense ran to the 8192-token cap in 32 canvases where
the sparse arms stopped after 15. So W and the canvas count C mostly measure length noise. More seeds narrow the
interval only slowly, so the comparison is moved onto identical canvases instead.

**Forced-canvas mode (bench option, opt-in; `CanvasForcing` in `scripts/v31_vllm_paired_bench.py`).**
- Per-canvas reseeding: the CUDA generator is reseeded from (request seed, canvas index) before every commit step,
  which draws the next canvas's initial noise. Canvas i's noise therefore does not depend on how many steps earlier
  canvases took.
- `FORCE_RECORD`: the reference run writes its committed token ids per cell (private).
- `FORCE_REF`: when a canvas of the arm converges, the mode records:
  - the arm's denoising-step count;
  - the share of its converged argmax tokens equal to the reference's.

  It then overwrites the converged canvas (argmax canvas, canvas, draft tokens) with the reference tokens before
  the commit step. Every canvas therefore starts from the reference prefix with the same noise seed.
- Step counts and agreement are paired per canvas. Only the arm's attention differs.
- Records carry `forced_mode`, `forced_canvas_steps`, `forced_agree` and `forced_output_matches_ref`. Their timing
  fields are not valid, because the mode synchronizes.
- Tests: `tests/test_v31_forced_canvas.py` (CPU, fake sampler state machine), 3/3 on mpk. They cover:
  - step counts per canvas;
  - one reseed per commit;
  - the reference overwrite of the argmax canvas, canvas and draft tokens (the tail beyond a short last canvas is
    kept);
  - the agreement values;
  - refusal of a missing reference.
- Report: `scripts/v31_forced_canvas_report.py`. It gives the step ratio (sum) and the per-canvas geometric mean,
  each with a cell-clustered 95% CI, plus more / equal / fewer canvases and token agreement.

**Panel fc (after gs2 on both hosts; LongBench-v2 64K + 96K confirmation cells, pf's shards).**
1. Reference: dense PIECEWISE.
2. Self-check: the same configuration forced with its own record.
3. Calibration: dense FULL forced with the PIECEWISE record.
4. Arms: MAGE k=4096, m2c and m2c k12 mass, all forced with the same record.

**Read-out rules, stated before the data.**
- The self-check must reproduce every canvas's step count and agree 1.0. Otherwise the mode is not deterministic
  and its differences are not read.
- The FULL calibration gives the execution-mode noise floor: numerics alone, no sparsity.
- A sparse arm's step inflation is its step ratio. It counts as a real effect only when the CI excludes the
  calibration's ratio.
- Token agreement ranks attention fidelity on identical inputs. This is the paper's step-inflation measurement;
  no prior work quantifies it.
- The measurement says nothing about wall time. Time per step stays the S/N of the free-running panels.

**Also settled today (vt).**
- Dense FULL, the same configuration in two separate engines: 6/6 identical outputs at 64K (dlm2). The earlier
  FULL-vs-PIECEWISE disagreement (6/12) is a deterministic numerical difference between execution modes, not
  run-to-run noise.
- `LEAN_HELD` + `NO_META_SYNC=check`: identical outputs for m2c and MAGE (3/3 each, dlm2). The CPU metadata clock
  had no mismatch in 162 checks. The check mode adds GPU reads, so its S/N (about 1.00) is not a speed result.
- The lean branch's 13 fake-torch unit tests load the adapter by repository path, so they error in the flat overlay
  layout. They pass in the repository layout (36 tests, the 13 CUDA ones skipped without CUDA). The CUDA classes
  pass on dlm2 (22 ok).

**fc revised before it started (10:25 UTC).** Arms added:
- m2c with the step-20 dense rescue (`DENSE_WHEN=step:20`). The dense-rescue idea comes from Junyu Liao's work, so
  it is a COLLABORATION CANDIDATE.
- The MAGE-port per-head unit (`qblock_max`, 12%, step 1), without and with `MAGE_CARRY`. Panel gs could not test
  the carry, because RULER answers are one canvas long.

The `RISK_GROUP=kv` arm was dropped: panel gs rejected group-shared selection.

**First fc attempt failed on dlm2 (11:17 UTC); fixed and relaunched.**
- vLLM's engine start-up warms the sampler up with dummy decode steps before any request. The forcing wrapper ran
  on them before `begin()` and failed (`AttributeError: steps`). The reference run died in engine init, and the
  later steps failed in cascade.
- Fix: the wrapper passes calls through while no request is active (engine warm-up, possibly several slots, and
  between requests). The state is initialised at construction.
  - New CPU test `test_engine_warmup_calls_pass_through_before_begin`; 4/4 on mpk.
  - The chain now aborts if the reference record is empty.
- The failed partial outputs are kept in `failed_fc1/` on dlm2. No GPU result was produced.
- Order on dlm2: vks started early when the failed chain wrote its final status line, so fc now runs after vks.
  mpk: gb → fc.
