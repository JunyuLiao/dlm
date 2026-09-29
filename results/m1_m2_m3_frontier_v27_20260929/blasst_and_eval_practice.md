# BLASST and recent sparse-attention evaluation practice (survey, 2026-09-29)

**Sources and confidence.**
- Helper-agent web survey. BLASST facts come from the arXiv HTML of 2512.12087 and from GitHub/PR pages.
- Rows marked *[U]* in the survey (not verified from full text) are excluded here unless labelled.
- The PDFs could not be read in this session.

## BLASST (arXiv 2512.12087, MLSys'26 best paper)

**Mechanism.**
- The skip decision is made **inside** the FlashAttention inner loop, per KV tile, by reusing the online-softmax local and running maxima.
- A tile is skipped when `local_max − running_max < ln λ`.
- There is **no separate selector pass**, so the decision is essentially free.
- Prefill skips exp() and the PV GEMM for pruned tiles. Decode skips loading the V tile.
- Traversal is in forward order. Reverse traversal is only an appendix ablation, with a negligible, dataset-dependent effect.

**Kernels and hardware.** Hand-written CUDA in TensorRT-LLM (`skip_softmax`) and FlashInfer. Measured on H200 and B200.

**Speed claims (Table 5)** are **kernel-only against FlashAttention-3 BF16**, with prefill at batch 1 and 64K tokens, and decode on B200 at batch 148 and 32K.

| sparsity | ~0% | ~24% | ~50% | ~72% |
|---|---|---|---|---|
| H200 prefill | 1.00× | 1.08× | 1.27× | 1.52× |
| H200 decode | 0.96× | 1.08× | 1.20× | 1.40× |

- The "≈20% break-even" is simply the first sampled point above 0%. There is no knee in the curve, because the selector has no extra cost.
- **End to end**, their own Fig. 5 reports only **~1.1× TTFT and TPOT** on Qwen3-30B. A 1.5–1.8× kernel gain shrinks to 1.1× in their own paper.

**Datasets and calibration.**
- RULER 4K–128K, LongBench v2, MATH-500, AIME 2024, GPQA.
- Thresholds come from an exponential fit of λ·L = α·exp(β·s), made on about 1,000 RULER sequences, so they transfer across context lengths.
- Reported quality: RULER-32K at 75% sparsity 91.67 vs 92.33 dense.

## Why BLASST looks large and we look small

1. **What is measured.** BLASST's big numbers are attention-kernel-only at 32–64K on long single sequences. Our numbers are complete DiffusionGemma forwards.
2. **Where the time goes.** In DiffusionGemma, 5 of 30 layers are GLOBAL. The 25 LOCAL layers are 1K-window layers, which add up to 0.34% of a forward, and the forward is MoE-dominated (~43%). With an efficient dense kernel, GLOBAL attention is ~5% of a forward at 14–17K keys.
3. **Selector cost.** Our selector is a separate pass: historical QK plus a projected-V risk, with periodic re-observation. Its cost grows with the key extent, whereas BLASST's selector is free.
4. **Same compression pattern.** At 32K our kernel-level held-support saving is ~5% of a full forward against same-kernel dense, which is the same kind of kernel→E2E compression BLASST shows for itself.

## What transfers to our work

- **Report kernel-only, per-forward and whole-request numbers side by side**, all against the strongest dense kernel.
- **Calibrate thresholds automatically** as a function of context length (BLASST's Algorithm 2).
- **Evaluate the way they do:** RULER at 4K to 128K, LongBench v2, plus a no-regression reasoning set (AIME, MATH-500, GPQA).
- **A candidate hybrid method: skip in-kernel as well.** Our historical pre-QK bitmap removes QK *and* PV for dropped tiles. A BLASST-style in-kernel test on the tiles we keep could additionally drop PV and exp work at zero selection cost. This combination is not yet implemented.

## Other works (abstract-level, see survey caveats)

- **Speed baselines.**
  - MInference, XAttention, SeerAttention-R, NSA, SparseD and PulseCol report kernel or prefill speedups against FlashAttention at 64K–1M.
  - Diffusion-LM papers (SparseD 1.2–1.5×, PulseCol ~1.95× *[U]*) report end-to-end speedups against FlashAttention at 64K.
- **Statistics.** Almost none state repeats, seeds or CIs.
- **Quality rules.** These are qualitative ("comparable", "near-lossless"), not pre-registered thresholds.
