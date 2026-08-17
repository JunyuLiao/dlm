# DiffusionGemma dense vs. BLASST on MATH500

## Executive result

| Metric | Dense | BLASST λ(local/global)=0.9/0.6 | BLASST − dense |
|---|---:|---:|---:|
| NeMo-Skills majority@10 | 92.91% | 93.23% | +0.33 pp |
| Deterministic majority@10 | 93.20% | 93.20% | +0.00 pp |
| pass@1, average of 10 draws | 76.54% | 71.38% | -5.16 pp |
| Oracle pass@10 | 96.20% | 96.20% | +0.00 pp |
| No extracted answer | 15.48% | 22.80% | +7.32 pp |
| Accuracy conditional on extracted answer | 90.56% | 92.46% | +1.90 pp |
| Local-layer physical tile sparsity | 0.00% | 28.62% | +28.62 pp |
| Global-layer physical tile sparsity | 0.00% | 19.97% | +19.97 pp |
| Mean completion length | 484.34 | 474.36 | -9.98 tokens |
| Mean full sequence length | 587.23 | 577.25 | -9.98 tokens |
| Mean model time/sample | 1.634 s | 2.581 s | +57.9% |
| Peak allocated CUDA memory | 50.30 GiB | 50.30 GiB | +0.00 GiB |

The standardized NeMo-Skills majority score is nominally **+0.327 percentage points** higher with BLASST. This advantage comes from NeMo-Skills' fractional treatment of tied modes: under the concrete deterministic tie-break, both systems solve **466/500 problems (93.20%)**. Thus this run does not show a deterministic majority-accuracy gain from sparse attention.

At the individual-draw level, BLASST is **5.16 points lower** and produces **7.32 points more** unextractable answers. Among draws where an answer is extracted, however, BLASST is 92.46% correct versus 90.56% for dense. This indicates that the pass@1 loss is dominated by answer-extraction failures rather than a lower correctness rate among parseable answers. Oracle pass@10 is unchanged at **96.20%**, so ten-sample coverage survives even though per-draw reliability falls.

## Paired analysis

All comparisons pair the same problem and sample index, with matching prompts and seeds.

| Paired outcome | Count |
|---|---:|
| Sample correct under both | 3086 / 5,000 |
| Correct only with BLASST | 483 / 5,000 |
| Correct only with dense | 741 / 5,000 |
| Incorrect under both | 690 / 5,000 |
| Problem majority correct under both | 456 / 500 |
| Majority correct only with BLASST | 10 / 500 |
| Majority correct only with dense | 10 / 500 |
| Majority incorrect under both | 24 / 500 |
| pass@10 only with BLASST | 8 / 500 |
| pass@10 only with dense | 8 / 500 |
| No answer under both | 343 / 5,000 |
| No answer only with BLASST | 797 / 5,000 |
| No answer only with dense | 431 / 5,000 |

- Across problems, BLASST has more correct draws on 109, dense has more on 229, and 162 tie.
- The paired problem-bootstrap 95% interval for BLASST − dense pass@1 is [-6.60, -3.74] percentage points.
- The corresponding interval for NeMo majority@10 is [-1.21, +1.93] points.
- Deterministic majority discordances are 10 BLASST-only versus 10 dense-only; exact paired McNemar p=1.000.

## Pass@1 by subgroup

| Subject | Problems | Dense pass@1 | BLASST pass@1 | Delta |
|---|---:|---:|---:|---:|
| Algebra | 124 | 85.16% | 80.08% | -5.08 pp |
| Counting & Probability | 38 | 82.37% | 81.84% | -0.53 pp |
| Geometry | 41 | 63.66% | 57.07% | -6.59 pp |
| Intermediate Algebra | 97 | 59.18% | 52.99% | -6.19 pp |
| Number Theory | 62 | 80.81% | 77.42% | -3.39 pp |
| Prealgebra | 82 | 85.61% | 83.29% | -2.32 pp |
| Precalculus | 56 | 75.00% | 63.21% | -11.79 pp |

| Level | Problems | Dense pass@1 | BLASST pass@1 | Delta |
|---|---:|---:|---:|---:|
| 1 | 43 | 88.14% | 86.05% | -2.09 pp |
| 2 | 90 | 88.44% | 81.89% | -6.56 pp |
| 3 | 105 | 82.00% | 75.24% | -6.76 pp |
| 4 | 128 | 74.38% | 71.02% | -3.36 pp |
| 5 | 134 | 62.61% | 56.94% | -5.67 pp |

BLASST's draw-level accuracy is lower in every subject and every difficulty level, so the aggregate decline is not attributable to a single MATH500 subgroup. The largest subject decline is in Precalculus; by difficulty, levels 2, 3, and 5 show the largest drops.

## Sparsity and efficiency

- BLASST skips **223,192,960/779,726,400 local tiles (28.62%)** and **31,721,788/158,810,400 global tiles (19.97%)**.
- Dense sparsity is 0% by definition; BLASST structurally masked tiles are excluded from the physical-sparsity denominator.
- BLASST takes 1.58× the dense model time in this implementation (2.581 versus 1.634 seconds/sample), while peak allocated memory is effectively unchanged.
- This timing is **not an optimized sparse-kernel result**: the reference BLASST backend applies the sparse mask and counts skippable tiles but still materializes dense QK scores. Its instrumentation adds overhead, so only accuracy and physical sparsity—not speedup—should be treated as the intended measurements.

## Length and termination

| Statistic | Dense | BLASST |
|---|---:|---:|
| Prompt tokens, mean / median / range | 102.89 / 81 / 40-881 | 102.89 / 81 / 40-881 |
| Completion tokens, mean / median / p95 / max | 484.34 / 403 / 1056 / 2048 | 474.36 / 401 / 979 / 2048 |
| Full sequence tokens, mean / median / p95 / max | 587.23 / 494 / 1243 / 2311 | 577.25 / 490 / 1207 / 2360 |
| Length-cap terminations | 4 (0.08%) | 5 (0.10%) |

Prompt length is identical by construction and includes the complete DiffusionGemma chat template and generation prefix. There is no natural-language system prompt and no few-shot example.

## Matched protocol and audit

- Model: `google/diffusiongemma-26B-A4B-it`, resolved revision `f7f5b7f5fa82ffc52addd066915886d497f5517b`, BF16, Transformers `5.12.1`.
- MATH500: 500 problems × 10 samples, temperature 0.6, top-p 0.95, base seed 42, maximum 2048 new tokens, thinking control disabled.
- NeMo-Skills commit `e06c9b900177be3f60d6a3f99135bb5de9af9bed`; identical dataset and prompt SHA-256 values in both configurations.
- Audit passed: 5,000 unique `(problem_index, sample_index)` keys and 500 unique self-consistency keys in each run; all paired prompts, prompt lengths, seeds, expected answers, and task identifiers match.
- Dense predictions SHA-256: `df7e76765fa02be9e60f55fa7b9fbc3242631aebbb92a50ff56434a15e4ce634`.
- BLASST predictions SHA-256: `200fd1b5dd2bcb00269244816b0e38c3c6cf29f979d4c784b6630a386d777828`.

## Source artifacts

- Dense summary: `results/diffusion_gemma_math500_dense/summary.json`
- Dense predictions: `results/diffusion_gemma_math500_dense/predictions.jsonl`
- BLASST summary: `results/diffusion_gemma_math500_blasst_l0p9_g0p6/summary.json`
- BLASST predictions: `results/diffusion_gemma_math500_blasst_l0p9_g0p6/predictions.jsonl`
