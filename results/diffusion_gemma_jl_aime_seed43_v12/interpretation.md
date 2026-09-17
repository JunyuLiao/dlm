# AIME26 seed43: empirical conclusions

Completed all **150/150 fresh outputs**, with no execution failures, missing shards or audit violations. Independent raw-only report regeneration passed. This run started only after the preceding 1,260-output low-rank sweep completed its independent audit.

Only the generation seed changed from 42 to 43. All 30 questions, prompts, 2,048-token output budgets, native decoding settings, model revision, routing implementations and previously calibrated thresholds were preserved. Gaussian8's projection seed remains 1729; its matrices did not change. Dense was rerun for seed43 and used as every sparse condition's baseline.

| Method | Seed42 correct/30 | Seed43 correct/30 | Seed43 accuracy | Whole sparsity | Global | Local | Retained mass | Token agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 17 | 17 | 56.7% | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| Aggressive BLASST | 11 | 8 | 26.7% | 50.46% | 49.29% | 50.75% | 70.94% | 2.30% |
| Mass-only bound | 14 | 16 | 53.3% | 51.48% | 53.18% | 51.07% | 74.52% | 4.76% |
| Full-dimensional centered | 15 | 17 | 56.7% | 51.11% | 50.90% | 51.16% | 72.78% | 4.91% |
| Gaussian8 centered | 15 | 15 | 50.0% | 50.59% | 48.85% | 51.02% | 72.88% | 5.09% |

All sparse targets were 50%. Physical sparsity is the ratio of summed skipped and eligible tiles, not an average of sample percentages. No thresholds were recalibrated for seed43.

## Interpretation

- Full-dimensional centered matches dense's correct count on this seed while skipping 51.1% of eligible tiles. This is encouraging, but does not prove equivalence: its paired accuracy difference versus dense is 0 points with a 95% interval of [−16.7, +16.7].
- Gaussian8 beats aggressive BLASST by seven questions, or **23.3 points [6.7, 40.0]**, at closely matched sparsity: whole/global/local gaps are +0.13/−0.43/+0.26 points. This supports the directional router relative to aggressive BLASST in this specific setting.
- Gaussian8 versus full-dimensional is −6.7 points [−23.3, +10.0]; versus mass-only it is −3.3 points [−23.3, +16.7]. Neither establishes a meaningful accuracy difference. Their global sparsities also differ by 2.05 and 4.33 points, respectively, despite similar whole-model sparsity.
- Gaussian8 scored 15/30 on both seeds, while full-dimensional moved 15→17, mass-only 14→16, aggressive BLASST 11→8 and dense remained 17. Equal correct counts do not mean equal outputs or identical correctly solved questions. Every within-method seed43-minus-seed42 interval includes zero. Two generation seeds with one fixed projection matrix are not a broad robustness study.
- Trajectory-local relative attention-output errors are approximately 0.479 for aggressive BLASST, 0.306 for mass-only, 0.258 for full-dimensional and 0.265 for Gaussian8. The centered methods reduce this error relative to those baselines, but the error ranking is not a substitute for downstream accuracy. Mass-only retains the most attention mass among these sparse methods but does not score highest.
- Aggressive BLASST reaches the output cap on 9/30 questions, versus 4 dense, 5 mass-only, 5 full-dimensional and 4 Gaussian8. This is an observed association, not an established causal explanation.

## Frozen threshold convention

Aggressive BLASST keeps `log(lambda) = log_scale - log(valid_KV_length)`, allowing lambda above one, with global/local log scales 7.779279118037186 / 8.091333801578767. No new calibration rule is introduced.

Natural-log scalar thresholds (global / local): mass-only −0.045639991760253906 / −0.018442153930664062; full-dimensional −1.2118581533432007 / −0.7402970194816589; Gaussian8 −1.268207311630249 / −0.7595282196998596. These have method-specific units and are not directly comparable importance values.

## Limits and provenance

All questions were previously examined and six historical calibration questions remain in headline30; noncalibration24 is separately reported. Bootstrap intervals are paired by question, exploratory and not multiplicity-corrected. The seed43 results did not select thresholds or projection directions.

Mass and execution-local output error use dense attention on each run's own QKV; token agreement uses the new seed43 dense output, includes positions after divergence, and counts missing/extra positions as disagreements. Shared-QKV diagnostics reuse the old seed42 states and are labeled separately; no fresh seed43 shared-state coverage is implied. No hardware-speedup claim is made.

The canonical report, JSON/CSV results, per-layer/head/step statistics, raw outputs, threshold provenance and independent verification are retained alongside this interpretation. This addendum does not modify the frozen numerical sources or audited artifacts.
