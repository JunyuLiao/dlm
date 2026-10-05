# T70 calibration refinement (v2, frozen before current-cohort final evaluation)

The v1 policy search is preserved at
`/home/exouser/ljy/dlm/results/query_adaptive_guardrail_v1`. On its 26 **calibration**
prompts, two T70 phase policies met the early, aggregate-sparsity and
trajectory guardrails. On the separate 26 **validation** prompts:

| Early log τ local/global | Late log τ local/global | Overall/global/local sparsity | Mean/p90 calls | Validation issue |
|---|---|---|---|---|
| −0.234/−2.291 | −0.084/−2.141 | 69.35/70.51/68.59% | 7.96/14 | p90 > 12 |
| −0.234/−2.291 | −0.084/−1.863 | 70.63/73.37/68.82% | 7.35/9.5 | global > 73% |

Both scored the same as matched-kernel dense on those 26 validation prompts.
The evidence suggests a narrow late-global-threshold interval between these
two policies. The exact 130 evaluation prompts and their accuracy were not
used to define this refinement.

The v1 validation outcomes informed the v2 candidate interval. Therefore,
although the 26 validation prompts remain disjoint from calibration and final
prompts, the v2 validation pass is a **development check**, not an unbiased
held-out confirmation. The two predeclared additional generation seeds reuse
the same final prompts and likewise do not provide fresh-prompt confirmation.

Version 2 keeps the v1 decoder, value router, sensitivity formula, physical
tile counters, H100 kernel and model unchanged. It fixes the first-two-call
local/global pair at −0.234/−2.291, then evaluates a small, recorded grid
around the archived late pair, including global offsets +0.05, +0.10, +0.15
and +0.20 in log threshold and limited ±0.07 local offsets. Every candidate
is a complete native adaptive generation on the same disjoint 26 calibration
prompts. Calibration feasibility and validation gates remain exactly v1’s:

- 70% pooled whole/global/local physical sparsity within ±2 percentage
  points on calibration, ±3 on validation;
- first-two-call whole <=75%, global/local <=77%;
- mean calls <=8, p90 <=12, at most one cap on 26 prompts;
- validation accuracy no more than 3 percentage points below matched-kernel
  dense. This tiny validation set makes the accuracy gate coarse and does not
  establish equivalence.

The best calibration-feasible point by maximum whole/global/local target
error is checked on validation. If no point passes, report the target as
unattainable under these guardrails; an evaluated fallback remains an
explicitly missed operating point. The 50% v1 policy is reused if it passes;
otherwise a separate bounded 50% refinement is performed. All thresholds
are frozen before running the 130-prompt final cohort. The previously used
130 examples remain exploratory, not fresh held-out confirmation.

## Reproduction commands

From `/home/exouser/ljy/dlm`, with `PYTHONPATH=src:.` and the
`/home/exouser/miniconda3/envs/ljy_dlm/bin/python` environment:

```bash
python -m experiments.value_direction_hopper.query_adaptive_guardrail_refine launch --root /home/exouser/ljy/dlm/results/query_adaptive_guardrail_v2
python -m experiments.value_direction_hopper.query_adaptive_guardrail_matrix launch-anchor-and-archive --root /home/exouser/ljy/dlm/results/query_adaptive_guardrail_v2
python -m experiments.value_direction_hopper.query_adaptive_guardrail_matrix launch-confirm --root /home/exouser/ljy/dlm/results/query_adaptive_guardrail_v2
python -m experiments.value_direction_hopper.query_adaptive_guardrail_matrix launch-calibrate --root /home/exouser/ljy/dlm/results/query_adaptive_guardrail_v2
python -m experiments.value_direction_hopper.query_adaptive_guardrail_matrix launch-final --root /home/exouser/ljy/dlm/results/query_adaptive_guardrail_v2
python -m experiments.value_direction_hopper.query_adaptive_guardrail_report --root /home/exouser/ljy/dlm/results/query_adaptive_guardrail_v2
```

Each GPU launcher uses a single worker and resumable, provenance-checked
per-example shards. Run one stage only after its predecessor has exited.
