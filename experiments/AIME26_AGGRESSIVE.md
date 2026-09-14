# AIME26 BLASST with lambda above one

From the repository root, use the `ljy_dlm` Python environment and `PYTHONPATH=src:.`.

- Initial calibration/evaluation: `python -m experiments.diffusion_gemma_aime26.run run`
- Resumable calibration-only refinement and evaluation: `python -m experiments.aime26_refinement`
- Canonical report and independent audit: `python -m experiments.aime26_finalize`
- Regression tests: `python -m pytest -q tests/test_aime26_aggressive.py`

The canonical final bundle is `results/diffusion_gemma_aime26_blasst_aggressive/refined/`.
It references immutable original dense/25%/50%/90% shards and contains the refined
75% condition. Keep the parent bundle together with it; symlinks avoid duplicating
the substantial cached attention statistics. Calibration traces and original
results are preserved, including the recovered JSON serialization failure.

Calibration uses questions 2, 8, 14, 20, 23, 30, never final accuracy. The full
30-question result includes these six; the report also gives the held-out 24.
The 90% target was not attained and is explicitly reported as such. The final
audit separately reports calibration attainment, full-evaluation completeness,
and the first-valid-tile bound on the saturated policy's actual trajectories.

This opts into the existing previous-running-maximum extension with finite lambda
above one, preserving default lambda validation for other callers. Physical tiles
are 128 query tokens by 64 KV tokens, with prefix and canvas eligible. This is not
a literal lambda-above-one application of the paper's updated-maximum pseudocode.
No optimized kernel or speedup claim is included.
