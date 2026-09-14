# AIME26 BLASST thresholds above one

Run from the repository root with the `ljy_dlm` Python environment and `PYTHONPATH=src:.`:

```bash
python -m experiments.diffusion_gemma_aime26.run prepare
python -m experiments.diffusion_gemma_aime26.run run
python -m experiments.diffusion_gemma_aime26.run report
```

Default output: `results/diffusion_gemma_aime26_blasst_aggressive`.
The queued process checks the observed GPU workload every 900 seconds and only
starts after it exits and no compute applications remain on the GPU. Do not
start another run while the queue is live.

All 30 AIME26 questions are pinned to dataset revision
`79037aebdb6580008fb960d17cb21fd3099083e3`. Calibration uses IDs
2, 8, 14, 20, 23, 30, selected deterministically across consecutive five-question
bands. Full30 includes these six questions; heldout24 is separately scored.
Threshold selection never uses answer correctness. Each condition shares the
same prompts, seed42, native sampler and 2048-token generation budget.

Attention uses 128-query × 64-key tiles with prefix and canvas eligible. The
shared implementation requires an explicit `allow_lambda_above_one=True` opt-in;
existing configurations retain their former threshold range. A whole tile skips
only if every valid active query votes to skip. Comparisons use the maximum of
preceding KV blocks, including previously skipped blocks, exactly as the existing
router did. Each query's first valid block remains retained at finite thresholds.

This is an extension of the existing router, not a literal application of the
paper's Algorithm 1 for λ>1: that pseudocode updates the maximum before testing,
so its margins are nonpositive and λ>1 would skip everything. The calibration
uses [Algorithm 2](https://arxiv.org/pdf/2512.12087): fit the exponential relation
λL=α exp(γs), independently for local/global attention. Scores/margins are
collected once from dense generation of the calibration questions. Actual valid
KV lengths are incorporated per call as adjusted margins `margin + log(L)`.
Fits exclude sparsity below 1% and above 97%.

If the paper fit misses by more than two percentage points, a recorded monotonic
search adjusts its scale on the same dense calibration data. Up to eight sparse
calibration verification/refinement rounds then select local/global scales with
the smallest maximum target error. All fits, corrections, and verification
generations are saved before the final run. Deployment keeps the inverse-length
rule `λ = exp(log_scale) / L`. Unreached targets remain explicitly labeled;
the measured full-benchmark sparsity is never replaced by the requested target.

Calibration dense outputs are reused in the full30 dense baseline. Cached shards
require matching configuration/code fingerprints. Report regeneration checks
prompt hashes, seeds, budgets, layer coverage, finite-output checks, effective
thresholds, and count validity. It exports full30, heldout24 and calibration6
accuracy, count-weighted physical sparsity, λ ranges and row-vote sparsity.
Retained mass is dense softmax mass on each sparse run's corresponding Q/K state.
This is a reference-mask experiment without speedup claims.

Tests:

```bash
python -m pytest -q tests/test_aime26_aggressive.py tests/test_blasst_mask_migration.py
```
