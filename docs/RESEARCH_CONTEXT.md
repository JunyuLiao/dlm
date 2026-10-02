# Research context: block-sparse GLOBAL attention for DiffusionGemma (M1/M2/M3 study)

Stable facts only. The current frontier is in `../HANDOFF.md`, decisions and negatives in `DECISIONS.md`, and
verified numbers in `RESULTS_LEDGER.md`. Detailed Chinese glossary: `results/m1_m2_m3_frontier_v27_20260929/progress_20260930.md`.

## 1. Question

Can block-sparse attention make DiffusionGemma decoding faster **end to end**, against the strongest official dense
baseline, with **no accuracy loss**? The setting is the model's native adaptive stopping: steps per canvas are decided
by the model, not fixed.
- The criteria are accuracy, performance and novelty.
- Trajectory or step analyses only explain those three; they are not goals.
- M1/M2/M3 are the study's original core. As of the user's 2026-10-01 update, follow-up panels may select only the
  stronger relevant arms, retaining dense and a matched optimized reference. Historical plain-method comparisons
  remain available. Engineering variants must be named honestly.

## 2. Model and decoding (what is fixed)

- **Model.** `google/diffusiongemma-26B-A4B-it`, revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`, bf16, one H100 80GB, batch 1.
  - Gemma 4 family: the config's vision tower is `gemma4_vision` and the processor is `Gemma4Processor`; the text
    model is `diffusion_gemma_text` (`DiffusionGemmaForBlockDiffusion`). Name it DiffusionGemma-26B-A4B.
  - 30 decoder layers, hidden 2816.
  - MoE: 128 experts, top-8, expert intermediate 704, about 46 GB of expert weights. Vocabulary 262,144.
- **Attention.**
  - 5 **GLOBAL** layers (5/11/17/23/29): head_dim 512, 16 Q heads, 2 KV heads, attend to the whole context.
  - 25 **LOCAL** layers: head_dim 256, 16 Q heads, 8 KV heads, sliding window 1024. Their cache keeps the last 1023
    tokens.
- **Block diffusion.** Output comes in canvases of 256 tokens.
  - Each canvas is denoised for up to 48 steps (decoder calls). Every call attends from the 256 canvas queries to the
    encoder KV cache plus the canvas.
  - After a canvas stops, the encoder appends its 256 committed tokens to the cache.
- **Official sampler and stop rule.** Do not change; the runner asserts them.
  - Entropy-bound acceptance (bound 0.1); non-accepted positions are renoised.
  - Temperature 0.8 → 0.4 linearly over 48 steps.
  - A canvas stops when (a) the argmax canvas equals the previous step's (stability 1) AND (b) the mean entropy of the
    processed logits is < 0.005.
  - The committed tokens are the argmax canvas of the last step.
- **Generation settings.** Thinking ON. AIME budget 8192 new tokens (32 canvases). LongBench-v2 uses its task budget.

## 3. Methods

Sparsity is applied **only to the GLOBAL layers' canvas (decoder) calls**:
- tile geometry is Q128 × KV64 per query head;
- canvas key tiles are always kept;
- LOCAL layers, the prompt prefill and the encoder canvas appends stay dense.

The kept tiles are executed by FA4's block-sparse interface (§4).

**Original methods (historical references; not required in every follow-up panel):**
- **M1.** For each candidate KV64 tile, estimate how much skipping it changes the attention output, from:
  - historical (observed) QK scores: an anchor every A calls; A8 is the plain method;
  - the CURRENT projected V (rank-32 Gaussian bank, seed 1729, per layer and KV head);
  - the current reference norm;
  - causal query sensitivity T, applied per row before the max over the Q128 tile.

  A tile whose log-risk is below the threshold is skipped. M1 redecides every call (R1). **The output always uses the
  CURRENT QK and the original V** over the kept tiles; skipped tiles get no QK or PV (`historical_route_preqk_current_output`).
  - Historical note: `method_contract.md` (the v5 contract, 2026-09-24) describes an older cached-score output.
    That is superseded.
- **M2.** M1 with the tile mean of V ("pooling V") in place of the attention-weighted projected V. Two versions:
  - `pooled`: row-wise reference (v26);
  - `pooled_compact` (M2c): one pooled vector per KV head and tile (v27, `m2_compact_contract.md`). Panels use M2c.
- **M3.** M1 redecision only every R calls (plain: R3), holding the bitmap in between.
- **Plain arms as run in v27:** `M1_R1_A8_fa4`, `M2c_R1_A8_fa4`, `M3_R3_A8_fa4`. These are Fan's definitions with
  execution through FA4. The pipelined selector (`route_pipeline`) is bit-identical in its decisions.

**Other comparison arms (never relabelled as M1/M2/M3):**
- **fresh T** (`T_scope`; fused version `fresh_fused`): Junyu's selection from the CURRENT QK plus projected V.
- **B** (`B_A8_matched`, `hold_only`): observe once and hold the bitmap with no redecision. With A64: one observation
  per canvas, held for the whole canvas. Close to SparseD (collision).
- **SparseD port** (`SparseD_k{keep}_s{skip}_fa4`): our labelled port of SparseD (ICLR 2026) on the same FA4 kernel.
  - The first `s` steps per canvas are dense; then tiles are ranked by average-pooled attention scores, the top
    `keep` fraction is kept, and the map is held to the canvas end.
  - In AIME panels the arm is named `SparseD_sXX` with XX the target sparsity.
- **G75** (`G75 F/S15/S30`): port of the vLLM-era historical-map method (GLOBAL 75% target; S15/S30 add LOCAL).
- **Bootstrap policy** `native_bootstrap2_observe1`, used by all v23+ method arms:
  - canvas call 0 is native dense;
  - call 1 is dense plus observation;
  - decisions apply from call 2.

**Schedule of the main configuration within one canvas (from `cache.py:plan` and `v27_dense_prefix.py`).**
- Call 0: native dense (with `carry_first`: the previous canvas's final map). Call 1: dense plus the fused
  observation, which also builds the prefix risk table (`_dp_build`, one scan over prefix tiles).
- Call 2: first decision; the map is held for calls 2–7. Call 8 re-decides (calls 8–13), call 14, … (`decision_interval=6`).
  With A64 and clock origin 1 there is no second observation within a canvas (canvases have at most 48 calls).
- A re-decision does not observe QK again and does not rebuild the prefix risk table (prefix K/V are fixed within a
  canvas). It re-projects only the canvas part of V, recomputes the V reference scale (RMS of V over all valid keys),
  and re-runs the threshold comparison (`_dp_decide`) with the CURRENT query sensitivity T, plus the canvas/boundary
  tail scan.
- **Query sensitivity T is active** (corrected 2026-10-02; an earlier note said T = 1). Every run wraps the router in
  `NativeReuseState('T', …)` with the bound parent config `beta=3.0, gamma=0.5, fast_t=True`: each step sets, per canvas
  position, T = clamp(1 + 3·e, 1, 4), where e is the exponential moving average (γ = 0.5) of "this position's argmax
  changed since the last step". log T (up to 1.39) is added to every row's risk before the worst-row max, so rows that
  are still changing keep more tiles. The reference scale does not change between decisions within a canvas.
- **Measured map drift** (`scripts/v27_map_drift_diag.py`, 64K/32K items 0–3): every re-decision changes the map;
  relative to the first decision of the canvas, 44–71% of its kept tiles change and the kept-set Jaccard is 0.59–0.70.
  Re-decisions follow T, so M3 R6 is not "select once per canvas".

**Our named variants (v27; each is an optional `v21` config key):**

| key / suffix | meaning |
|---|---|
| `score_period` A16/A64; `fused_observe` | re-observation period. A64 = once per canvas, fused into call 1's dense pass (prefix summaries plus score tail) |
| `decision_interval` R6/R12 (R3 = parent default) | M3 decision interval |
| `risk_state='dense_prefix'` (DP) | M1 risk against the dense-prefix state, so tiles are decided in parallel (0.27 ms/layer vs 2–8 ms) |
| `threshold_shift` (−3ln2 … +4ln2) | log-threshold shift; negative keeps more tiles |
| `async_route`, `route_pipeline` | selector on a side stream / pipelined. Bit-identical decisions; time only |
| `min_route_keys` (gate2k…gate32k) | below N keys the layer runs exactly dense |
| `route_layers`, `share_layers` | GLOBAL layer subsets / cross-layer shared support |
| `density_gate` (ent*, cap*, stall*, `stable1`) | the rest of the canvas runs dense after a sampler-state trigger |
| `risk_budget`, `risk_topk` (topkXX) | summed-risk budget / fixed target sparsity by risk rank; keeps k70/k60/k50/k30/k20/k12 (k12 = keep 12%, 88% sparsity, added 2026-10-01 for E9; arm names say topk88) |
| `carry_canvases` K | reuse a layer's decision for the next K−1 canvases (newer tiles kept) |
| `carry_first` (c0) | only canvas call 0 reuses the previous canvas's decision (newer tiles kept); call 1 still observes |
| `observe_step` (obs2) | dense calls 0..s−1, observation at call s |
| `protect_output` (po) | generated-token key tiles are never skipped |
| `proj_rank` (r16/r8/r4) | projected V at rank r: the first r columns of the rank-32 Gaussian bank × √(32/r) (nested JL), same kernels |
| `risk_value='mass'` | score-only control for the top-k selector: rank tiles by attention-mass share, no V term |

**Main configuration** "M3 R6 DP −ln2" = `M3_R6_A64_fused_dp_async_m1ln2_fa4`. It is parent `M3_R3_A8_current_output`
with `decision_interval=6`, `score_period=64`, `fused_observe`, `async_route`, `risk_state=dense_prefix`,
`threshold_shift=minus_ln2` (global threshold −3.874), FA4 consumer.

## 4. Baselines and execution substrates

**Dense controls.**
- **`D_fa4_allkept`** (headline): FlashAttention-4, the vLLM fork's CuTe DSL kernel with SM90 head_dim ≤ 512, called
  through its block-sparse interface with every tile kept.
  - Bitwise equal to FA4 dense.
  - Kernel 4–6% faster than plain FA4 dense; about 1.3% at decode level.
  - Sparse arms use the same kernel, so dense and sparse differ only in skipped tiles.
- **`D_fa4`**: the plain FA4 dense path.
- Superseded:
  - `D_c64`: our 64-row Triton dense, 1.9× slower than FA4;
  - `D_fast`: GLOBAL repeat-KV SDPA;
  - `D_native`: HF default path, repeat-KV plus SDPA mem-efficient, about 2.1× slower than FA4 at the 64K kernel;
  - `D_matched`: same-consumer all-kept, legacy.
- FlashInfer (FA2 backend) is tied with FA4. FA2/FA3/upstream FA4/cuDNN cannot run head_dim 512 on SM90.
- Source: `results/m1_m2_m3_frontier_v27_20260929/official_baseline/README.md`.

**Substrates** (`experiments/numerical_qk_reuse/v27_substrate.py`). All arms of a panel share one substrate; numerics
differ between substrates.

| substrate | content | used by |
|---|---|---|
| `eager` | HF eager path. Host-bound: GPU savings in attention do not reach wall time | every panel before 2026-09-30 (v20–v27 Tier 3, long RULER, LB-long) |
| `piecewise_v1` | decoder `torch.compile` reduce-overhead with eager GLOBAL boundaries (vLLM split) and compiled sampler | FA4 LB-long panels v1–v3 |
| `piecewise_v2` | + FA4 causal prefill and FA4 GLOBAL canvas append | quality q1v3, threshold sweep |
| `piecewise_v3` | + no per-call GLOBAL KV concat, no batch-1 sync (tokens identical to v2) | final panels, SparseD, variants v5, held-out |
| `piecewise_v4` | + static LOCAL shape (fixes the recompile-limit eager fallback on short prompts, e.g. AIME) | AIME v4 check, 64K traj |
| `piecewise_v5` | + compiled post-attention tail of encoder canvas appends | E1–E4 (current) |

## 5. Workloads

- **LongBench-v2 pool:** 12 items, 10–19K tokens.
- **LongBench-v2 length bins:** official items at natural length, no truncation.
  - 32K bin: 28–40K tokens; 64K bin: 56–76K tokens; 24 items each.
  - The first 12 by sha256(id) are the **formal** items, which were used to pick the −ln2 threshold. Items 13–24 are
    the **held-out** items.
  - 96K bin: 84–104K tokens; only prompts ≤ about 95K fit on one H100.
  - 128K bin: dense OOMs in prefill.
- **AIME26:** 30 problems; context ≤ about 8.4K keys.
- **RULER:** 4K and 32K/64K.
- **Seeds:** 101/202/303 (+404 for the final AIME panel); 404–909 for the E4 confirmation.
- **Scoring:** NeMo multiple-choice scorer for LongBench-v2 (v15 contract), exact match for AIME.

## 6. Metrics and conventions

- **Cells.** One cell is (item, seed); each arm contributes only its first output. All arms of a cell run on one host.
- **Ratios vs `D_fa4_allkept`.** Paired geometric mean with a question-clustered bootstrap 95% CI:
  - `W`: request wall;
  - `Wc`: W excluding pairs that captured new CUDA graphs while timed;
  - `S`: decode span, excluding prefill; `P`: prefill;
  - `N`: decoder calls; `NC` = N/C: steps per canvas; `T`: output tokens;
  - `S/N`: amortized per-step cost (not a direct per-forward price);
  - `S/C`: decode time per canvas (= per-step cost × steps per canvas).
- **Accuracy.** Strict-correct counts with +/− discordant cells and a paired sign test.
- **Realized sparsity.** Skipped GLOBAL key tiles over all GLOBAL tiles, by construction (canvas tiles kept, first
  calls dense). It is not per-tile telemetry.

## 7. Comparison rules (do not compare apples to oranges)

- **Substrate.**
  - Never compare ratios across substrates (eager vs piecewise_vN).
  - Eager-era end-to-end ratios (v20–v27 Tier 3, long RULER, LB-long on `D_c64`/`D_native`) hold only for that host-bound substrate.
  - Quality must be re-measured on each new substrate.
- **Baseline.** Ratios against `D_c64`, `D_fast` or `D_native` must be re-based on FA4 before any claim.
- **Replication.**
  - Same host, same seed, same substrate reproduces tokens exactly. Re-running is speed replication, not an
    independent accuracy replication.
  - Across hosts, tokens differ: about ±3/36 correct for the same arm.
  - Pool same-host method/dense cell ratios, never raw latencies from different hosts. Report host-specific
    diagnostics; different item/seed mixtures mean their differences alone cannot identify a hardware effect.
  - The FA4 timing summary now rejects duplicate first outputs, missing scored executions and joins that differ in
    host/GPU, substrate, frozen protocol, model revision or source hashes (see `INTAKE_AUDIT_20261001.md`).
- **Trajectory noise.** Dense-vs-dense across model loads differs by a median 1.17× in denoising calls
  (`research/adaptive-trajectory-characterization-20260921`). Step and request ratios from 3 seeds × 12 items are
  noisy; prefer NC and S/C for mechanism and large seed panels for request claims.
- **Selection bias.** The −ln2 threshold was chosen on the formal 32K/64K items; report held-out items separately.
- **Levels.** Kernel, attention-module and request speedups differ by about 6× at 64K; report all three. Amortized
  per-step costs are not per-forward prices.
- **Group members' results** come from other stacks (for example a JAX/FA3 run with a 2048-token cap). Do not compare
  them directly, and do not state their setups beyond what their slides or code say.

## 8. Cost structure (why gains are bounded at batch 1)

Dense, piecewise_v3–v5, from `substrate/time_breakdown.json`, `kernel_bench/module_profile_*.jsonl` and the panels'
per-step decode costs:
- **Per step:** about 37 ms at 32K and 45 ms at 64K (amortized).
  - MoE expert GEMM about 12.8 ms: all 128 experts are hit by 256 × 8 assignments, so each step reads about 46 GB and
    is HBM-bound.
  - GLOBAL attention (5 layers): 7.3 ms (about 20%) at 32K, 14.8 ms (about 33%) at 64K, ≤ about 5% on AIME.
  - Sampler over the 262K vocabulary: about 4.6 ms.
- **Request share at 64K:** prefill about 35%, GLOBAL decode attention about 21%. So even a free, step-neutral sparse
  attention caps the request gain at about 16% (32K) and 21% (64K).
- Batching was measured at 64K: B=1/2/4 keeps prefix attention at about 26/22/25% of a forward, with keep-0.12
  savings about 23/19/20%. It did not raise attention's share (`batch_scaling/README.md`).

## 10. Dataset coverage and the V-dimension question (2026-10-01 intake)

- The user relayed a classmate's hypothesis: RULER needs V dimensional information, AIME/LongBench may not, and
  HumanEval is unknown. This is a hypothesis, not a verified result of this branch. Actual attention output already
  uses full-dimensional V in every arm; projected V is only a selector input. Existing RULER8K peer evidence
  supports V direction versus mass-only, but does not establish that all 512 dimensions are necessary.
- HumanEval has 164 Python tasks with executable tests ([official paper](https://arxiv.org/html/2107.03374v2),
  [harness](https://github.com/openai/human-eval)). Six seeds mean 984 generations per arm. It is a proposed coding
  quality check, not yet a registered v27 task or an established sparse-speed workload.
- Full LongBench-v2 has 503 tasks ([official source](https://github.com/THUDM/LongBench)). The existing private
  all-item rendering log uses the pinned model tokenizer and unchanged task template, without truncation:
  min/median/max input = 10,334 / 107,706 / 5,174,028 tokens; 224 inputs are at most 95,074 tokens. This is a length
  eligibility screen, not evidence that all 224 requests fit the GPU.
- The pinned model's text config has `max_position_embeddings=262144`. Reserving 8,192 output tokens leaves 400
  of the 503 rendered inputs in range; 402 inputs alone fit, and 101 inputs alone exceed the configured limit.
  Chunked prefill may remove the current about-95K memory bottleneck but does not extend the positional context.
  A complete 503-task run needs an explicit truncation/retrieval protocol for over-limit inputs. Report that
  separately from an untruncated eligible subset. No such full-panel run has been launched.

## 9. Related work and novelty status (as of 2026-10-01)

- **SparseD** (arXiv 2509.24014, ICLR 2026): dense early steps, then reuse of a pooled-score pattern. Collides with B.
  In a head-to-head on FA4 there is no significant speed difference from its best setting.
- **PulseCol** (2605.20813) ≈ periodic refresh (M3-like). **Focus-dLLM** (2602.02159) keeps response tokens and prunes
  the prompt. **FlashBlock**, **LoSA** (2604.12056): reuse for stable tokens.
- All of these use **fixed** step counts or lengths, so none reports step inflation under adaptive stopping.
- Output-length inflation from sparse attention is documented for autoregressive reasoning models:
  **Lil** (2601.03043), **LessIsMore** (2508.07101).
- Step-reduction methods (orthogonal; would not be our contribution): Prophet (2508.19982), JoT (2602.11133),
  SchED (2512.02892), and EB-sampler (2505.24857), which DiffusionGemma already uses.
- In this branch's tested AIME/LongBench settings, no data supports "V-aware selection beats score-only selection".
  This does not settle the RULER hypothesis (§10). At high AIME sparsity our risk top-k is
  worse than SparseD.

## 11. Authorized group-context review (2026-10-01, 23:10 UTC−5)

The specified four-person group was read with user authorization. Sanitized background and paper links are in
`GROUP_CONTEXT_20261001.md`. Peer V-rank/step-inflation observations are external context, not new v27 results.
The reported BLASST control is not the repo's mass-only control. Same-forward spatial S/TS must not be treated as
available for pre-forward regroup; distinguish it from previous-step signals and from the decode-time metric S.
A/B refs remain unknown; no integration or GPU generation has occurred.

Related-work boundaries: HERALD uses CPU/GPU KV offloading and block reuse, with adaptive accuracy evaluation but
fixed-T performance on LLaDA/SDAR; PRR repairs speculative selections on AR DSA; SSV combines sparse speculative
verification and acceptance-aware orchestration. They constrain novelty but do not measure the current native
DiffusionGemma/FA4 protocol. BRISK-DLM's prefix-conditioned corrector is not assumed portable to native decoding.
