# Expansion plan: datasets and models (2026-10-02, proposal, nothing run yet)

The user asked which datasets comparable papers use (preferring new and long ones) and whether to add models such as
LLaDA2.1-mini or I-DLM, with the usual baseline rule: each model's dense baseline must be its official or widely
known SOTA serving path. This file records the survey and the proposal; decisions go to `DECISIONS.md`.

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
| **UltraLLaDA** (LLaDA-8B → 128K) | ICLR 2026 | 32 layers, MHA hd 128, bidirectional over the whole sequence | 128K | FlashAttention dense, as SparseD/PulseCol compare; sparse baselines from official code (SparseD, Sparse-dLLM, Focus-dLLM) | yes: the long-context dLLM that Focus-dLLM and the block-approximate paper use |
| I-DLM-8B (from Qwen3-8B) | 2026-04 | strict causal attention, introspective strided decoding | Qwen3 window | SGLang | no: AR-style causal decoding has no bidirectional canvas, so it is a different sparse-attention problem (decode sparsity, Quest/DSA family) |
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
