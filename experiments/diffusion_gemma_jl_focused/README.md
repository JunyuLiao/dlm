# Focused JL output-aware experiment

User-narrowed successor to the broad JL study. The superseded worker and queued aligned addendum were explicitly stopped by approval; previous code, logs, outputs, and calibration traces remain untouched.

New candidates at 50% and 75% target physical sparsity:

- `jl_gaussian_r32`: attention-weighted centered online update, Gaussian rank32.
- `contribution_gaussian_r32`: uncentered projected contribution, the same Gaussian rank32/seed1729.
- `full_centered`: full-dimensional centered reference.

All30 AIME26 questions (2048 tokens) and a score-blind50-question subset of the completed newer100 LongBench v2 NeMo run (4096 tokens). Domain/subtask quotas use largest remainders; selection seed20260915. Both projected methods use rank32 to isolate centering. No final-score tuning or family/dimension/seed winner selection.

Reuse560 exact baseline outputs: dense plus original/aggressive BLASST and mass-only at both targets. Reuse verified centeredJL32/full-reference calibration and540 dense calibration QKV snapshots; newly calibrate only contribution32 with the existing six-per-benchmark512-token, max3 verified-point scalar local/global search. LongBench baseline calibration12 spans6 domains; reused JL calibration6 spans3 domains. This asymmetry is documented, not repaired using final data. AIME’s6 calibration problems overlap its30 final questions; report noncalibration24 separately.

The implementation adds scoped registry dispatch and dataset/cache/report orchestration only. Frozen custom Triton routing, structural masks, GQA,128×64 tiles, prefix+canvas eligibility, FP32 projections, strict tie handling and native retained BF16 output are unchanged. Full-dimensional PV and shared dense states are diagnostic-only for the projected routers. No FlashAttention or hardware-speedup claim.

The cached `mass` baseline is the existing conservative block-mass-bound method: its candidate numerator is valid-token-count times exp(block maximum), compared with the retained running normalizer. It is **not** the separate `mass_exact` ablation. The three new methods use the candidate’s actual within-block softmax normalizer. Thus a difference versus cached mass-only does not, by itself, isolate directional information; the centered versus uncentered rank32 comparison holds those details fixed. This limitation is preserved rather than silently replacing a cached baseline’s algorithm. Shared calibration-state sensitivity for Gaussian rank32 seeds1729/2718/31415 is reused; no seed is selected, and no end-to-end seed-robustness claim is made.

From the repository root, using the existing `ljy_dlm` Python environment:

```bash
PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_focused.workflow prepare
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=src:. OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false python -m experiments.diffusion_gemma_jl_focused.workflow launch
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_focused.workflow report
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH=src:. python -m experiments.diffusion_gemma_jl_focused.workflow verify
```

Output: `results/diffusion_gemma_jl_focused_lb100_subset50_v2/`. One GPU worker; supervisor health records every900 seconds. Resumes completed shards; independent configuration failures are recorded and do not stop other final configurations. Final reports require1040 audited outputs and shared diagnostic coverage; raw-only regeneration verifies identical artifacts. Main physical sparsity sums skipped and eligible tile counts across samples/layers/steps. Token agreement compares all positions after divergence, with missing/extra tokens counted as disagreements.
