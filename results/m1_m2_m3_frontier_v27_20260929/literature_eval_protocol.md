# How recent sparse-attention / cache papers evaluate (survey, 2026-09-29)

Scope and caveats:
- This is a web survey by a helper agent. The arXiv ids were fetched directly.
- Numbers marked *[unverified]* must be re-checked against the primary PDF before any citation.
- It informs our protocol; it is not evidence about our method.

## Diffusion-LM papers

| paper | models / lengths | benchmarks | speed metric | sparsity knob | quality rule | baselines |
|---|---|---|---|---|---|---|
| SparseD (2509.24014) | LLaDA-1.5, Dream-7B; 4k–64k | MMLU, GSM8K, HumanEval, RULER 4k/8k | attention/E2E vs FlashAttention: 1.23–1.25× at 64k/128 steps, 1.48–1.50× at 64k/1024 steps | top-ρ% per query, block-averaged; ρ swept 5–50% | "lossless" (±0.3 pt) | dense FA, sliding window, StreamingLLM, dKV-Cache, Fast-dLLM |
| PulseCol (2605.20813) | LLaDA-1.5 | GSM8K, HumanEval, RULER *[tables unverified]* | ~1.95× vs FA at 64k/1024 steps *[unverified]* | column-sparse | *[unverified]* | FA, SparseD |
| Sparse-dLLM (2508.02558) | LLaDA-8B, Dream-7B; gen 256/512 | MMLU, ARC-c, PIQA, GPQA, GSM8K, MATH, HumanEval; LongBench (appendix) | tokens/s vs vanilla (avg 3.4×; the abstract's "up to 10×" is not reconciled) | retention ratio r swept 0.1–0.9 | "comparable" | vanilla, dLLM-Cache, dKV-Cache, Fast-dLLM |
| Fast-dLLM (2505.22618) | LLaDA, Dream; gen 256–1024 | GSM8K, MATH, HumanEval, MBPP | tokens/s vs vanilla, 5.7–27.6× (cache plus parallel decoding) | block cache + confidence threshold | 1–2 pt | vanilla, ablations |
| d2Cache (2509.23094) | LLaDA-8B, Dream-7B | GSM8K, MBPP, HumanEval, MATH-500, GPQA, MMLU-Pro | tokens/s vs vanilla (avg 3.5×) | fixed token budget per step | "comparable or better" | dLLM-Cache, Fast-dLLM, vanilla |
| Elastic-Cache (2510.14973) | LLaDA, LLaDA-1.5, Dream-7B; gen 256/512 | GSM8K, MATH, HumanEval, MBPP | tokens/s, up to 45× vs a slow no-cache reference | sliding window β | within 1–2% | dKV-Cache, dLLM-Cache, Fast-dLLM, dense |
| dLLM-Cache (2506.06295) | LLaDA-8B, Dream-7B | LongBench-HotpotQA (+ GSM8K, MATH, code, partly verified) | FLOPs reduction up to 9.1× | prompt/response cache intervals | "competitive" | vanilla |
| dKV-Cache (2505.15781), DPad (2508.14148), Fast-dLLM v2 (2509.26328) | *[details unverified]* | | | | | |

## Autoregressive comparison set

| paper | setting | reported speed |
|---|---|---|
| MInference | prefill up to 1M; InfiniteBench, RULER, Needle | up to 10× prefill |
| XAttention | RULER, LongBench, VideoMME | attention up to 13.5× |
| NSA | trained native sparse attention; 64k | forward 9× vs FA2 |
| SeerAttention-R | reasoning models; **AIME24/25, GPQA-D, MATH-500** | up to 9× vs FA3 at 90% sparsity |
| Quest, DuoAttention | long-context decode | decode 1.5–2.2× |
| FlexPrefill, SpargeAttn, RetrievalAttention, ShadowKV, TidalDecode | exist; not deep-read | — |

## De-facto protocol and what it means for us

1. **Two tables.**
   - A short-generation reasoning and code quality table: GSM8K, MATH/MATH-500, HumanEval, MBPP, often GPQA.
   - A separate long-context table where attention sparsity is actually shown: RULER 4k–64k and/or LongBench.
2. **Headline dLLM speedups are not attention-sparsity numbers.**
   - The 5–45× figures combine KV/cache reuse and parallel decoding, measured against a no-cache "vanilla" dLLM.
   - DiffusionGemma already has a native cache, so those baselines do not apply to us.
   - Papers that isolate sparse attention against FlashAttention report 1.2–2× at 64k (SparseD; PulseCol *[unverified]*).
   - This is consistent with our ~8–10% per-forward ceiling at 16k, where GLOBAL attention is ~16% of a forward.
3. **Sparsity is reported as a swept knob** (top-ρ%, retention ratio, budget) with an accuracy-vs-sparsity curve, not a single point.
4. **Quality rule:** "within ~1–2 points / comparable", reported per task.
5. **Steps:** the native schedule is kept, and speed is reported at that schedule. This matches our native adaptive-stopping contract. We additionally report calls/canvas and total calls.
6. **AIME** is not used in any surveyed dLLM sparsity paper. The AR reasoning-sparsity line (SeerAttention-R) does use AIME.
   - For us, AIME serves as a no-regression quality check.
   - Our measured result, with no per-forward speed gain at short context, agrees with the literature.

## Implications for our remaining plan

- Add a **long-context speed tier**: RULER 16k/32k/64k within the model's supported length. This is where GLOBAL attention becomes a large share of a forward.
- Report an **accuracy-vs-sparsity sweep** on the quality tasks (LB, AIME, plus GSM8K/MATH-500 if available), with **AIME no-regression** as a hard constraint.
- Baselines:
  - native dense and same-consumer dense (D_matched);
  - fresh T (a SparseD-like fresh selection);
  - B (a pattern-reuse hold).
  - Cache-based dLLM methods do not apply to DiffusionGemma's native KV cache as-is.
