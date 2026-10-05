# Same-state query-sensitivity diagnostics

13 task-balanced RULER4K prompts, 25 step-2/3 snapshots. Each replay uses identical canvas, self-conditioning and prefix cache. The matched-kernel dense forward is returned to the native decoder, so alternate replays cannot create a different diagnostic trajectory.

| Method | Physical sparsity | Top-1 disagreement | Acceptance-set disagreement | Confidence loss | Sampled attention error | Mixed-query tiles | Reference/CUDA tile mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| CT_s50 | 47.5% | 2.2% | 2.9% | +0.0013 | 0.132 | 0 | 0/0 |
| CT_s70 | 68.8% | 4.1% | 15.5% | +0.0099 | 0.258 | 902 | 0/3562 |
| unweighted_s50 | 50.1% | 2.5% | 3.2% | +0.0028 | 0.143 | 0 | 0/0 |
| unweighted_s70 | 70.8% | 6.4% | 36.3% | +0.0314 | 0.288 | 788 | 0/3562 |

High-sensitivity versus low-sensitivity and returned-answer versus outside-answer damage rates are in `analysis.json`. Row-vote diagnostics use a small FP32 reference and must be qualified by their mask disagreement with the production kernel. These are replay associations, not evidence that selection caused downstream accuracy differences.
