# HumanEval v2 investigation plan

The v1 pilot cannot distinguish a routing-score effect from a calibration and
trajectory effect. The v2 run therefore freezes a stratified 100-question
manifest and calibrates complete sparse trajectories at measured 50% physical
sparsity before scoring any final completion.

## Tests of the earlier hypotheses

1. **Budget:** compare actual pooled whole, local, and global tile sparsity.
   A BLASST advantage is interpretable only against a matched T/C operating
   point.
2. **Trajectory length:** compare mean, median, P90, cap counts, accepted and
   renoised positions, and phase sparsity. More calls can indicate that the
   query weights are changing future logits and reopening tokens.
3. **Syntax stability:** split failures into `syntax_error` and `failed_tests`,
   measure prediction length by status, and compare normalized completions to
   the dense output. A score gap caused by parse failures has a different
   remedy from a semantic test failure.
4. **Task type:** report accuracy and status by the six fixed entry-point
   strata in the manifest. HumanEval does not ship official task-type labels;
   the taxonomy is a transparent keyword grouping, not a learned label.
5. **Routing versus value estimation:** if T/C still need more calls at the
   same sparsity, inspect whether the extra work is concentrated in low
   confidence or high-flip positions. If failures remain syntax-heavy despite
   denser early phases, confidence/flip sensitivity is not selecting the code
   structure that pass@1 requires.

## Interpretation rules

The v2 result will not be summarized as a universal benchmark ranking. The
strongest supported statement will be the paired outcome at matched measured
physical sparsity, followed by the failure and call-count mechanism. A syntax
advantage is considered causal only if it remains after matching thresholds,
task strata, and seeds and is visible in parse/failure diagnostics; fewer
syntax errors in one 18-question pilot is not enough.

