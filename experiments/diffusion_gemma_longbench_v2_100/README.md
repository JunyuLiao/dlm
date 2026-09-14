# LongBench v2:100 fresh problems, NeMo-Skills prompt/scoring

This isolated experiment reuses the frozen `diffusion_gemma_value_aware_gpu` attention implementation. It does not modify routing, kernels, native denoising, tile geometry or benchmark answers. Dense plus original BLASST, aggressive BLASST, mass-only and output-risk at50/75% target physical sparsity produces900 final outputs.

The100-problem final sample uses proportional domain and subdomain quotas from the full503-problem population. Separate12-problem calibration and6-problem development sets are excluded from final scoring; all previous experiment IDs are excluded. Sampling is score-blind, deterministic and not short-first.

The pinned official NeMo-Skills checkout supplies its actual LongBench prompt and default MCQ evaluator. Generation remains native DiffusionGemma, including its0.4–0.8 temperature schedule. This is not a claim of full NeMo serving/greedy protocol replication.

Literal model-reserved control tokens in source text (for example a code repository documenting `<|image|>`) are HTML-escaped before chat formatting. This prevents text-only documents from being misinterpreted as missing multimodal inputs; no selected examples are dropped. Replacement counts and raw-source hashes are recorded. The original6 pilot prompts are proven unchanged by an explicit cache-reuse audit, with the pre-escape code snapshot preserved.

Run from the repository root with `PYTHONPATH=src:.`, the existing `ljy_dlm` Python environment, and the available H100:

```bash
python -m experiments.diffusion_gemma_longbench_v2_100.pilot
python -m experiments.diffusion_gemma_longbench_v2_100.workflow prepare
python -m experiments.diffusion_gemma_longbench_v2_100.workflow launch
# CPU-only, exclusively from audited completed raw shards:
CUDA_VISIBLE_DEVICES='' python -m experiments.diffusion_gemma_longbench_v2_100.workflow report
```

The disjoint pilot checks memory at32K/48K/64K/128K and completion at4096/8192/16384 output tokens. It selects the highest safely tested context and the lowest output budget with no length limits and at least5/6 parsable answers. It never selects by correctness. The final dense sanity gate requires all100 outputs, at most5 length-limited and at least90 parsable answers. Failures are preserved and investigated before proceeding.

Existing calibration rules are reused, with historical scalars as proposals only and at most3 verified points on the new12 calibration problems. Original BLASST caps λ at1 and reports independently verified local/global unattainability. Aggressive BLASST permits λ>1. Mass/risk retain their existing empirical-CDF scalar calibration. No final data tunes thresholds.

Sparsity uses sums of eligible/skipped128×64 physical tile counts, separately for overall/global/local denoising attention, including prefix/canvas and separately classified boundary tiles. Dense prefill is outside this denominator. Retained mass uses the exact dense distribution at each sparse query state; cached dense generations supply positional output agreement. Missing/extra token positions disagree, including after first divergence.

`results/diffusion_gemma_longbench_v2_100_nemo/` holds immutable manifests, pilot evidence, calibration policies/traces, per-example generations, per-layer/head/step counts, actual NeMo scores, consolidated CSV/JSON, plots, audits and a Markdown report. The persistent supervisor checks job/GPU/disk status every900 seconds. Successful shards survive failed configurations and resumes. No hardware speedup is measured or claimed.
