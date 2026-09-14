# Engineering-accelerated value-aware50/75 rerun

User-authorized kernels are **only for reducing experiment runtime**. This is
not a FlashAttention/kernel/system benchmark. Old scientific sources, contracts
and usable results are untouched.

Scope: dense + original/aggressive BLASST and value/vector_mean, mass-only,
output-risk/mean, token-aligned value, each at50% and75% targets.13 conditions
on30 AIME26 and50 LongBench v2 questions =1040 final results. AIME reports full30
and noncalibration24. V2 keeps the earlier three domains, expands17/17/16 with
25easy/25hard and preserves existing30 IDs;20 additions exclude all tuning IDs.
All samples/prompts/token IDs are cached once. Native2048/128 output budgets,
32k v2 input cap, model revision, BF16, seed42 and denoising schedule stay fixed.
This preserves the prior v2 answer-completion limitation; no post-hoc scoring
or prompt changes are made.

`kernels.py`: custom Triton64-key block max/logsum/count/aligned-max reductions
and a128-query all-row streaming decision loop. Original/aggressive BLASST uses
the maximum of every preceding block; new risks use retained pre-block state.
Strict less-than, GQA, padding/structural validity and nonempty rows are unchanged.
Mass/risk decisions within2e-5 of the threshold use the original operator as a
numerical guard. Max-based exact ties do not need this guard.

`routing.py`: native QK/logit rounding and full-query FP32 softmax ->BF16 PV stay
in PyTorch/cuBLAS. The original same-state diagnostic operators and reduction
order are reused unchanged. A trial whole-PV diagnostic regrouping failed
long-context numerical checks and was discarded, not accepted by widening
tolerances. Mass/risk routing also uses the original block log-sums in the fused
streaming kernel, avoiding a second source of cancellation-sensitive rounding.
Per-layer/head/step physical counts are summed on GPU and copied at generation
end. Prefix/canvas/mixed-boundary counts are distinct; no theoretical sparsity
is substituted for measured skipped/eligible counts.

`calibration.py`: a backend adapter around the existing follow-up calibration
code.50% policies are imported unchanged after provenance auditing.75% reuses
compatible verified AIME policies or existing dense-calibration proposals and
the original inverse-L/scalar-risk empirical-CDF search (up to3 new verification
rounds). Only the original6 calibration examples per benchmark fit thresholds;
no expanded final-set scores enter fitting. Original BLASST uses exactlambda1
for each independently verified unattainable attention type.

Commands from the repository root, with the `ljy_dlm` Python environment:

```bash
PYTHONPATH=src:. HF_HUB_OFFLINE=1 python -m experiments.diffusion_gemma_value_aware_gpu.protocol
PYTHONPATH=src:. HF_HUB_OFFLINE=1 python -m experiments.diffusion_gemma_value_aware_gpu.runner smoke
PYTHONPATH=src:. HF_HUB_OFFLINE=1 python -m experiments.diffusion_gemma_value_aware_gpu.workflow launch
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_gpu.workflow report
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_gpu.verify
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m experiments.diffusion_gemma_value_aware_gpu.posthoc
```

The single-worker supervisor records status every900s. The workflow refuses
overlapping GPU jobs, requires matching CUDA validation before freezing/running,
reuses compatible native dense caches, continues independent configurations
after errors, checkpoints each successful example, and reports only1040/1040 as
complete. Exact duplicate accelerated operators may alias their completed
results with source hashes; different methods/budgets/thresholds never do.

Outputs: `results/diffusion_gemma_value_aware_gpu_50_75/`: manifests, frozen
execution/configs, CUDA proofs, policies/traces, per-sample generations, raw
layer/head/step counts, summary CSV/JSON, task scores, nearest-actual-sparsity
BLASST comparisons, attention mass/error/agreement, plots, report and audit.
V2 uses official micro accuracy with equal-domain macro separately; prior30
and new20 strata are also exported. Report regeneration does no inference.
The raw-only interpretation includes direct paired mass-only controls, actual
whole/global/local budget gaps, changes from50% to75%, calibration saturation,
and descriptive mass/agreement/error correlations. Report code does not select
or modify any inference threshold. Confidence intervals for mass controls are
explicitly in percentage points; budget proximity is not proof of equivalence.

Implementation references: [Triton libdevice](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html)
and [fused reductions](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
No performance comparisons from those tutorials are claimed here.
