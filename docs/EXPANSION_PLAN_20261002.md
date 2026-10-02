# Expansion plan: datasets and models (2026-10-02)

The user asked which datasets comparable papers use (preferring new and long ones) and whether to add models such as
LLaDA2.1-mini or I-DLM, with the usual baseline rule: each model's dense baseline must be its official or widely
known SOTA serving path. This file records the survey and the proposal; decisions go to `DECISIONS.md`.

## Decisions (user, 2026-10-02)

- **Models added:** LLaDA2.1-mini and I-DLM-8B, next to DiffusionGemma.
  - UltraLLaDA is dropped under the user's rule ("old or does not fit one H100"). It is from October 2025 on the
    February 2025 LLaDA-8B base, and recomputes the whole sequence every step: minutes per request at 64K–128K, and
    the official generate's full-sequence logits alone are about 33 GB in bf16 at 128K.
- **Envs and weights:** separate envs under dyh with the official serving stacks; weights downloaded into dyh.
- **Datasets:** add LongBench Pro and RULER 4K/8K.
- Both new models top out near 32K (LLaDA2.1-mini 32,768; I-DLM-8B 40,960 positions). Their accuracy panels
  therefore use RULER 4K–32K, LongBench-v2 ≤ 32K and LongBench Pro levels 8K–32K. This matches the dLLM papers.
  DiffusionGemma keeps the 64K/96K bins.
- **Official serving paths (dense baselines):**
  - LLaDA2.1-mini: upstream SGLang 0.5.21 with `--dllm-algorithm JointThreshold --attention-backend flashinfer`;
  - I-DLM-8B: the SGLang bundled in the I-DLM repo, `--dllm-algorithm IDLMBlockN` (ISD), FlashInfer backend.
  - The sparse methods will modify the same FlashInfer attention (its block-sparse interface), so dense and sparse
    differ only in skipped blocks.
- **I-DLM caveat.** It is causal (SDAR architecture, 36 layers, 32 Q / 8 KV heads, hd 128). Our canvas observation
  and held-map structure must be adapted to its strided verify-and-advance decoding; the comparable sparse baselines
  are decode-sparsity methods (Quest/DSA family).
- **Setup** (coordinator chain `setup_chain.sh`, after the dllm diagnostics) into `/home/exouser/dyh/dlm_models_20261002`:
  - `envs/sglang_up` and `envs/sglang_idlm`, with pip freezes;
  - `models/`, `datasets/LongBench-Pro`, each with its HF revision recorded;
  - every pip/HF/XDG/tmp cache stays inside that directory.

## Baseline selection protocol (user rule: official or SOTA, presentable in a paper)

For every model:
- Benchmark all official dense serving candidates at the model's real geometry: batch 1, bf16, the panel's context
  lengths, same prompts and stopping.
- Use the fastest correctness-equivalent one as the dense reference, and state why each excluded candidate is out.
- Build the sparse method as a modification of that reference's attention kernel (its block-sparse interface), so
  dense and sparse differ only in skipped blocks.

| model | dense candidates | status |
|---|---|---|
| DiffusionGemma | (1) ours: HF model + piecewise_v5 substrate + FA4 all-kept (`D_fa4_allkept`); (2) HF official compiled path; (3) **vLLM 0.30.0 native DiffusionGemma** (official serving, merged 2026-06; FA4 at hd512 on SM90); (4) SGLang (forces Triton attention at hd512) | (2) is measured slower: 44–56 ms/step at 17K–32K, OOM at 60K (`official_baseline/official_compiled_path.jsonl`). **(3) has never been measured end to end, a gap in the current evidence.** If vLLM is faster per step at our lengths, port the method into vLLM's attention call or re-base the speed claims on it. vLLM env C is added to the setup. |
| LLaDA2.1-mini | (1) upstream SGLang 0.5.21 (JointThreshold) with FlashInfer / FA3 / Triton backends; (2) **dInfer** (inclusionAI's own framework; LLaDA2.1-mini path through its SGLang backend); (3) HF transformers remote code (correctness reference) | vLLM's dllm-plugin is excluded: it needs a vLLM fork branch, and its LLaDA2 model logic is unfinished per its docs |
| I-DLM-8B | its bundled SGLang (IDLMBlockN / ISD) with FlashInfer / FA3 / Triton backends | transformers loading is unsupported per the model card; this is the only official path |

Sparse baselines: official code where it exists, ported (and labelled as ports) where it does not.
- Block diffusion with a KV cache (LLaDA2.1-mini, I-DLM/SDAR): MAGE (no code; port), LoSA, Quest (official,
  FlashInfer), SparseD (official; full-recompute regime, port the rule), PulseCol.
- Each choice is re-checked against current preprints before the panel is frozen.

## What the closest papers evaluate (checked against the papers)

| paper | venue / date | models | accuracy benchmarks | lengths for accuracy | speed claim |
|---|---|---|---|---|---|
| SparseD (2509.24014) | ICLR 2026 | LLaDA-1.5, Dream-7B-Instruct | MMLU, GSM8K, HumanEval, RULER-4K/8K | ≤ 8K | 1.50× vs FlashAttention at 64K, 1,024 steps |
| PulseCol (2605.20813) | preprint, 2026-05 | LLaDA, Dream | GSM8K, HumanEval, RULER-4K/8K | ≤ 8K | up to 1.95× end to end vs FlashAttention at 64K |
| Block approximate sparse attention (2605.19726) | preprint, 2026-05 | LLaDA-1.5, UltraLLaDA (+ LLaDA-V, Wan2.1) | RULER 4K–32K, LongBench (16K) | ≤ 32K | attention operator up to 6.95× at 256K |
| Focus-dLLM (2602.02159) | preprint, 2026-02 | UltraLLaDA, Dream-7B | LongBench, NIAH | 8K–32K | throughput vs vanilla / Fast-dLLM |
| Prefilling-dLLM (2606.10537) | EMNLP 2026 | (see paper) | LongBench, InfiniteBench | 8K–32K | up to 28× end to end at 32K |
| Sparse-dLLM (2508.02558) | preprint, 2025-08 | LLaDA, Dream | LongBench (truncated to 4K) | 4K | throughput |
| UltraLLaDA (2510.10481) | ICLR 2026 | LLaDA-8B → 128K | NIAH-128K, LongBench-16K, RULER 4K–128K, PG19 PPL | ≤ 128K | — |

Recent AR sparse-attention work uses RULER (to 128K), LongBench v1/v2, HELMET and InfiniteBench. DeepSeek's DSA
reports AA-LCR and Fiction.LiveBench.

Reading:
- dLLM sparse-attention papers judge accuracy at ≤ 32K (mostly RULER-4K/8K and LongBench ≤ 16K). Only speed is shown
  at 64K–256K.
- Our accuracy panels (LongBench-v2 32K/64K/96K, RULER 32K/64K) are already longer than all of them.

## Dataset proposal

| dataset | year | lengths | role |
|---|---|---|---|
| LongBench-v2 (keep) | 2024-12 | our 32K/64K/96K bins | main natural long-context accuracy + speed |
| RULER (keep, add 4K/8K) | 2024 | 4K, 8K for direct comparison with SparseD/PulseCol; 32K/64K (R16) | standard synthetic check in every dLLM paper |
| **LongBench Pro** (add) | 2026-01 (2601.02872) | six levels, 8K–256K; 1,500 samples, EN/ZH, Apache-2.0 | newest natural long benchmark; use the levels inside each model's window. Check its per-task metrics first (exact match vs judge) |
| AIME26 (keep) | 2026 | short | reasoning accuracy check (no speed room) |
| HumanEval (keep, small) | 2021 | short | only for comparability with SparseD/PulseCol; old and likely contaminated |
| NoLiMa (optional) | 2025 (ICML) | up to 128K | harder retrieval (no literal overlap) if reviewers ask |

## Model proposal

| model | date | geometry | context | official / SOTA dense baseline | fit |
|---|---|---|---|---|---|
| DiffusionGemma-26B-A4B (current) | 2026 | 30 layers (5 GLOBAL hd512), MoE 128×top-8, canvas 256 | 96K+ on one H100 | FA4 (vLLM fork) all-kept | main |
| **LLaDA2.1-mini** (inclusionAI) | 2026-02 | 16B MoE (256 experts, top-8), 20 layers, 16 Q / 4 KV heads, hd 128, block diffusion with KV cache (block 32) | 32K | official SGLang (`--attention-backend flashinfer`, `--dllm-algorithm JointThreshold`) or dInfer | yes: newest mainstream open dLLM family; mini is the only LLaDA2.x that fits one H100 (flash = 100B). Limits: 32K window, 32 query rows per step |
| UltraLLaDA (LLaDA-8B → 128K; dropped, see Decisions) | ICLR 2026 | 32 layers, MHA hd 128, bidirectional over the whole sequence | 128K | FlashAttention dense, as SparseD/PulseCol compare; sparse baselines from official code (SparseD, Sparse-dLLM, Focus-dLLM) | yes: the long-context dLLM that Focus-dLLM and the block-approximate paper use |
| **I-DLM-8B** (from Qwen3-8B; added by user decision) | 2026-04 | SDAR, 36 layers, 32 Q / 8 KV, hd 128, strict causal attention, introspective strided decoding | 40,960 | I-DLM's bundled SGLang (IDLMBlockN), FlashInfer | added; needs an adapter for causal strided decoding |
| IDLM (inverse distillation, in chw/value_aware) | 2026-02 | small DiT on OpenWebText / TinyGSM | short | — | no: not a long-context LLM |

Prerequisites:
- No host env has SGLang or FlashInfer. The official LLaDA2.1 baseline needs a separate env under `dyh` (user decision).
- Weights: about 32 GB (LLaDA2.1-mini) and about 16 GB (UltraLLaDA), to dyh dirs. Space is ample on every host.

## Code structure for more models

- `core/`, model-independent:
  - risk table and dense-prefix route;
  - q64 refinement;
  - FA4 or FlashInfer block-sparse list builders;
  - carry and observe logic;
  - query sensitivity;
  - metric, receipt and panel tooling.
- `adapters/<family>/`: one thin adapter per model family. It owns:
  - the attention hook point and layer selection (which layers are "global");
  - the canvas/step bookkeeping (DiffusionGemma canvas 256; LLaDA2.1 block 32 with KV cache; UltraLLaDA whole-sequence
    diffusion);
  - the sampler statistics for T;
  - the official dense baseline call.
- The DiffusionGemma path must stay bit-identical: a regression test against existing receipts (tokens, calls).

## Setup status (2026-10-02 09:05 UTC-5, dllm, `/home/exouser/dyh/dlm_models_20261002`)

| item | result |
|---|---|
| `envs/sglang_up` | SGLang 0.5.21, torch 2.13.0+cu130, FlashInfer 0.6.18. The first try failed: `cuda-tile`'s wheel stub could not verify pypi.nvidia.com with the base Python's CA store. Fixed by pointing `SSL_CERT_FILE` at pip's certifi bundle for that process only |
| `envs/sglang_idlm` | I-DLM's bundled SGLang (`IDLMBlockN` import verified) |
| `envs/vllm` | vLLM 0.30.0, torch 2.13.0+cu130 |
| `models/LLaDA2.1-mini` | 31 GB, HF revision 20e64e2ad216 |
| `models/I-DLM-8B` | 16 GB, HF revision 3cecd8cd39b9 |
| `datasets/LongBench-Pro` | 513 MB, HF revision 4996884deae5 |

Pip freezes and revisions are stored next to each item; every cache stays inside that directory.

## LongBench Pro inventory (2026-10-02, from the downloaded `longbench_pro.json`, revision 4996884deae5)

- **Size and balance.** 1,500 samples: 250 at each length level (8k, 16k, 32k, 64k, 128k, 256k; counted with the
  Qwen tokenizer). 750 English and 750 Chinese.
- **Tasks.** 11 primary tasks, 25 secondary tasks (60 each).
- **Fields.**
  - `context`;
  - `question_nonthinking` and `question_thinking`;
  - `answer` (list);
  - `contextual_requirement` (Full 840 / Partial 660);
  - `difficulty` (Easy 482 / Moderate 289 / Hard 289 / Extreme 440);
  - `token_length`, `primary_task`, `secondary_task`, `language`.
- **Scoring is per task** (paper):
  - NDCG@k for retrieval/ranking;
  - pairwise accuracy for sequencing/clustering;
  - accuracy for QA;
  - F1 for citation and violations;
  - SubEM for single-answer generation;
  - SemSim + ROUGE-L for summarization, which needs an embedding model.
  Use the official evaluation code (GitHub `caskcsg/longcontext`) rather than a reimplementation. Check it before
  freezing a panel.
- **Panel plan.**
  - DiffusionGemma: the 32k/64k/128k levels, re-measured with its own tokenizer.
  - LLaDA2.1-mini and I-DLM-8B: 8k/16k/32k within their windows.
  - Both languages, `question_thinking` when thinking is on.
  - Gold stays private like all other gold.

## Official dense smoke status (2026-10-02 10:49 UTC−5, dllm)

`scripts/v27_sglang_dense_smoke.py` runs the official in-process SGLang engine on one public toy prompt.
LLaDA2.1-mini (upstream SGLang 0.5.21, `JointThreshold`, FlashInfer) does **not run yet**:
- deep_gemm (FP8 GEMM only): disabled with `SGLANG_ENABLE_JIT_DEEPGEMM=0`;
- deep_ep imports need `CUDA_HOME`: pointed at the env's pip CUDA 13 package `.../site-packages/nvidia/cu13`;
- FlashInfer JIT failed (CUDA compiler/headers incompatible): fixed by installing `flashinfer-jit-cache==0.6.18+cu130`
  (prebuilt kernels) into the env; `ninja` was also added to both SGLang envs;
- SGLang's own JIT kernels failed to link (`cannot find -lcudart`): the pip CUDA package has only
  `lib/libcudart.so.13`. Fixed with a CUDA_HOME shim inside dyh, `/home/exouser/dyh/dlm_models_20261002/cuda13_shim`:
  `bin`, `include`, `nvvm` and `cccl` link to the pip package, and `lib64` holds `libcudart.so` / `libnvrtc.so` links.
  The `activation` JIT kernel then built.
- **open:** a FlashInfer kernel outside the prebuilt cache still JIT-compiles and fails with "CUDA compiler and CUDA
  toolkit headers are incompatible": FlashInfer's bundled CCCL does not match the pip nvcc.
  Next options:
  - install the nvcc/CUDA 13.x version FlashInfer 0.6.18 expects into the env (or a full CUDA 13 toolkit in a dyh
    prefix);
  - find which kernel is missing from `flashinfer-jit-cache` and whether `flashinfer-cubin` covers it;
  - as a first smoke, try `--attention-backend fa3` (also official in SGLang). FlashInfer may still be used for
    non-attention ops.
- I-DLM-8B smoke: not run yet (same toolchain needs; config `src/I-DLM/inference/configs/idlm_blockN4_config.yaml`).

**Cache hygiene for every SGLang/vLLM run** (dyh-only rule): set `SGLANG_CACHE_DIR`, `SGLANG_JIT_CACHE_DIR` (JIT
kernel builds; its default `~/.cache/sglang/jit` ignores `SGLANG_CACHE_DIR`), `TVM_FFI_CACHE_DIR`, `XDG_CACHE_HOME`,
`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `CUDA_CACHE_PATH`, `FLASHINFER_WORKSPACE_BASE`, `VLLM_CACHE_ROOT`,
`HF_HOME` and `TMPDIR` to directories under `/home/exouser/dyh/dlm_models_20261002/`.
- SGLang ignores `XDG_CACHE_HOME` for its own cache and wrote `~/.cache/sglang` on import (14:02 UTC) and during the
  smoke test (15:47 UTC).
- One Triton cache entry (`~/.triton/cache/KJGB…`, 13:56 UTC) came from the I-DLM env's import check.
- Both were created by these runs (timestamps match) and were removed at 15:49 UTC.
- A later smoke attempt recreated `~/.cache/sglang/jit` (2.2 MB, 15:50 UTC), because `SGLANG_JIT_CACHE_DIR` was not yet
  set; it was removed at 15:52 UTC.
- No `~/.cache/tvm-ffi` and no new `~/.triton` entries exist.

## RULER: what it is and the R17 proposal (2026-10-02)

**RULER** (NVIDIA, COLM 2024, "What's the Real Context Size of Your Long-Context Language Models?") is a synthetic
benchmark generated to any target length with the model's own tokenizer. Its 13 tasks:
- 8 needle-in-a-haystack variants (single 1–3, multi-key 1–3, multi-value, multi-query);
- variable tracking (multi-hop);
- common-words and frequent-words extraction (aggregation);
- 2 QA tasks (SQuAD / HotpotQA with distractor paragraphs).

`scripts/data/prepare.py` fills a haystack (essays or noise) to `max_seq_length` minus the answer budget and inserts
the needles at random depths. Papers report 4K–128K.

**Our pools** (`/home/exouser/dyh/ruler_long_v27/readme.json`): pinned RULER commit c3f5e3b, seed 42, the
DiffusionGemma tokenizer, `length_mode total`, 13 tasks × 1 sample at 32K and 64K. Some rows were regenerated after a
chat-template-overhead fix. The closest dLLM papers use RULER only at 4K/8K (SparseD, PulseCol) or up to 32K.

**R17 proposal** (accuracy only; RULER answers take about 5 decoder calls, so prefill dominates and no speed claim is
possible):
- **Data:** RULER at 32K, 64K and 96K (96K = 98,304 total, which fits one H100 like the LB 96K items), 13 tasks ×
  10 samples = 130 per length. Same pinned generator and tokenizer, a new seed, gold kept private.
- **Arms:**
  - dense FA4;
  - M3 + c0 (main);
  - fixed 88% (k12) with projected-V rank 32 / 8 / 4 and mass-only (the group member's V-dimension question);
  - fixed 95% (k5) with rank 32 and mass-only, the sparsity where V effects are most likely to appear.
- **Size and order:** about 3,100 runs, about 3 h on three hosts. Run after E15.
- For the new models (windows ≤ 32K): RULER 4K/8K/16K/32K with each model's tokenizer.
