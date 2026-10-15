# Whole-layer same-state replay

Complete operator diagnostic: all 16 heads and both query tiles, five GLOBAL layers, 570 usable frames from 620 saved frames. Signed head errors are summed through the real O projection before taking norms. The table excludes anchor calls and all-kept states. It does not report model decisions or official task accuracy.

| Trajectory | Method | Sparse held frames | Mean error / mass control | Frames with worse mean | Frames with >2% worse maximum |
|---|---|---:|---:|---:|---:|
| request004 | correction_moments | 30 | 0.9059 | 15 | 20 |
| request004 | correction_residual | 30 | 0.8327 | 9 | 11 |
| request004 | full_swap8 | 30 | 0.9776 | 6 | 10 |
| request004 | mass_anchor | 30 | 1.0000 | 0 | 0 |
| request004 | online_swap8 | 30 | 0.9796 | 4 | 5 |
| request001 | correction_moments | 220 | 0.5398 | 21 | 126 |
| request001 | correction_residual | 220 | 0.6355 | 6 | 26 |
| request001 | full_swap8 | 220 | 0.9847 | 0 | 1 |
| request001 | mass_anchor | 220 | 1.0000 | 0 | 0 |
| request001 | online_swap8 | 220 | 0.9861 | 0 | 0 |
| request002 | correction_moments | 100 | 0.7694 | 48 | 68 |
| request002 | correction_residual | 100 | 0.7399 | 15 | 25 |
| request002 | full_swap8 | 100 | 0.9733 | 2 | 8 |
| request002 | mass_anchor | 100 | 1.0000 | 0 | 0 |
| request002 | online_swap8 | 100 | 0.9762 | 2 | 8 |

Requests001/002 are previously examined LongBench trajectories; request004 contributes the only sparse AIME trajectory. AIME K=27 is an operator counterfactual on source K=128 states. Request003 is all-kept and supplies no sparse-selection evidence.

The bounded full/online swaps retain modest improvements after cross-head cancellation. Cached correction has a much larger mean improvement, but its row tails often worsen. This supports testing a causal correction-confidence rule; it does not support deploying the unconditional correction or claiming accuracy preservation. The tested previous-query extrapolation failed the preceding screen and is not promoted.

Next gate: source-matched final-GLOBAL-layer suffix replay for logits, acceptance, entropy, and stopping, with same-mask numerical-floor controls and native trajectory/RNG invariance. Earlier-layer deployment requires a broader decision intervention. Clean timing remains paused.
