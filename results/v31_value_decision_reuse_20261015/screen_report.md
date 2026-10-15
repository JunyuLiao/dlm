# First same-state operator screen

Source-frozen independent operators; 36 H100 tests passed. Verified 670 immutable Q/K/V files; replayed 620 frames, with 570 usable frames (including map observations), two heads and one Q128 tile. No budget errors or CUDA captures. Official accuracy and model decisions have not been evaluated. Timing is paused.

The following table excludes all-kept units and anchor calls. Each ratio compares the identical head/query-tile state with its mass anchor; post-O is an exact single-head metric and does not yet include cross-head cancellation.

| Request | Method | Reused units | Post-O error ratio | Units with increased mean error |
|---|---|---:|---:|---:|
| request004 | correction_moments | 60 | 0.8834 | 17 |
| request004 | correction_residual | 60 | 0.7883 | 8 |
| request004 | correction_tangent | 60 | 1.4360 | 41 |
| request004 | full_swap2 | 60 | 0.9954 | 1 |
| request004 | full_swap8 | 60 | 0.9893 | 1 |
| request004 | lifetime_swap2 | 60 | 0.9942 | 2 |
| request004 | lifetime_swap8 | 60 | 0.9932 | 2 |
| request004 | mass_anchor | 60 | 1.0000 | 0 |
| request004 | online_swap2 | 60 | 0.9948 | 1 |
| request004 | online_swap8 | 60 | 0.9942 | 1 |
| request001 | correction_moments | 440 | 0.5117 | 4 |
| request001 | correction_residual | 440 | 0.6319 | 21 |
| request001 | correction_tangent | 440 | 0.7446 | 123 |
| request001 | full_swap2 | 440 | 0.9926 | 4 |
| request001 | full_swap8 | 440 | 0.9853 | 3 |
| request001 | lifetime_swap2 | 440 | 0.9930 | 7 |
| request001 | lifetime_swap8 | 440 | 0.9860 | 8 |
| request001 | mass_anchor | 440 | 1.0000 | 0 |
| request001 | online_swap2 | 440 | 0.9932 | 6 |
| request001 | online_swap8 | 440 | 0.9859 | 4 |
| request002 | correction_moments | 200 | 0.6854 | 43 |
| request002 | correction_residual | 200 | 0.6654 | 35 |
| request002 | correction_tangent | 200 | 1.0867 | 106 |
| request002 | full_swap2 | 200 | 0.9874 | 6 |
| request002 | full_swap8 | 200 | 0.9697 | 6 |
| request002 | lifetime_swap2 | 200 | 0.9883 | 12 |
| request002 | lifetime_swap8 | 200 | 0.9735 | 11 |
| request002 | mass_anchor | 200 | 1.0000 | 0 |
| request002 | online_swap2 | 200 | 0.9881 | 5 |
| request002 | online_swap8 | 200 | 0.9723 | 6 |

The causal previous-query extrapolation improves next-state error prediction on 0/100 sparse age-1 units. Mean post-O prediction error rises from 2.62502 to 4.74002. Central finite-difference tests passed, so this is evidence against the tested predictor on these states, not a derivative-sign failure.

Bounded swaps show small gains; cached correction shows larger mean reductions with harmful tails. Proceed to whole-layer signed-error replay. Do not promote the extrapolation or infer denoising/accuracy gains from these operator errors. Historical snapshots lack sampler statistics, so C_gate and acceptance/stopping gates remain pending.
