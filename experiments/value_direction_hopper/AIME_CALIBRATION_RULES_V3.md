# AIME26 phase-aware temporal calibration (v3)

This version replaces the previous rule that froze one shared early pair and
searched only the later threshold. It calibrates the first two denoising calls
separately from later calls while preserving early protection through threshold
ordering rather than a fixed sparsity ceiling.

## Threshold schedule

Each policy contains independent local/global thresholds for:

```text
call1: first denoising call of each 256-token canvas
call2: second denoising call
late:  all subsequent calls
```

The temporal sensitivity state remains causal: call 1 has no history, call 2
uses only the state available before call 2, and later calls use the method's
own previous predictions. Native stopping, acceptance, renoising, temperature,
and canvas semantics are unchanged.

## Sparsity objective

For each requested target `s` (for example 40% or 70%), calibration measures
pooled physical 128x64 eligible/skipped tiles separately for:

- call 1;
- call 2;
- calls 3 and later;
- the complete trajectory;
- local, global, and whole attention.

Every phase and every attention stratum is required to be within two
percentage points of `s` when a feasible policy exists. Counts are pooled
before division; per-example or per-step percentages are never averaged.

## Protection rule

Protection is an ordering constraint, not a fixed early sparsity cap. For each
attention stratum, the call-1 and call-2 thresholds must be no more aggressive
than the late threshold:

```text
threshold(call1) <= threshold(late)
threshold(call2) <= threshold(late)
```

Thus early calls may still reach the requested sparsity when the trajectory
supports it, but they cannot receive a more aggressive threshold than later
calls solely because one uniform threshold was used.

## Search procedure

1. Start from the previous policy, expanding its legacy `early` pair into
   call 1 and call 2.
2. Evaluate predeclared independent perturbations of the six coordinates:
   local/global × call1/call2/late.
3. Coordinate-refine each coordinate in 0.20 log-threshold increments using
   complete calibration generations.
4. Rank policies by the maximum absolute phase/stratum error, then by the
   protection-ordering violations, denoising steps, and executed tiles.
5. Freeze the selected policy before final seeds are run.

Changing an early threshold changes the temporal trajectory and therefore the
late sparsity; no score-histogram interpolation is used.

If no policy satisfies all phase and pooled constraints, the selected result is
labelled `unattainable_under_phase_constraints` and its actual operating point
is reported without relabelling it as the requested target.

The six AIME calibration IDs overlap the historical AIME30 cohort, so this is
development calibration rather than fresh held-out evidence.
