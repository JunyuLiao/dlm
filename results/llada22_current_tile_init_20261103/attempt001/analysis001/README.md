# LLaDA2.2-mini HumanEval: current-tile initialization

Exploratory follow-up on the previously examined full164-problem cohort, seed42. Original native sampler/EOS/editing rules and1024-token cap; all20 layers, Gaussian32, Q128/KV64, refresh every fourth call. Only V2 initialization/order changes: current mandatory tile first, then chronological tile0 through J-2.

| Method | Correct /164 | Accuracy | Mean output tokens | N/request | C/request | N/C | Tile sparsity | EOS / cap | Syntax errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Native dense | 141 | 85.98% | 243.1 | 28.76 | 8.57 | 3.357 | 0.00% | 164 / 0 | 2 |
| All-kept FA4 | 138 | 84.15% | 249.2 | 31.42 | 8.77 | 3.581 | 0.00% | 163 / 1 | 2 |
| V31 | 83 | 50.61% | 218.9 | 36.76 | 7.84 | 4.692 | 49.98% | 164 / 0 | 31 |
| Original V2 | 10 | 6.10% | 62.9 | 14.21 | 3.07 | 4.634 | 49.22% | 164 / 0 | 98 |
| V3a | 103 | 62.80% | 245.3 | 42.32 | 8.67 | 4.881 | 49.99% | 163 / 1 | 21 |
| V2 current-tile init | 102 | 62.20% | 214.2 | 27.65 | 7.62 | 3.627 | 61.35% | 164 / 0 | 10 |

Calibration: six handwritten nonbenchmark development prompts, seed142, 4 threshold candidates; frozen threshold 0.29084308669, confirmed sparsity 50.93%. Calibration uses tile counts only. Full benchmark runs once, without retuning based on its achieved sparsity or accuracy.

Tile sparsity is1-sum(kept)/sum(eligible) on actual denoising consumers, including mandatory tiles, excluding selector discovery, dense prefill and commits. N/C is pooled; per-request statistics and complete counters are in summary.json. Times are instrumented accounting, not clean latency.

Paired accuracy differences for the new variant (95% problem-bootstrap intervals,10000 draws):
- Versus Native dense: -23.78pp [-31.10, -16.46], 3 wins/42 losses.
- Versus All-kept FA4: -21.95pp [-29.27, -15.24], 3 wins/39 losses.
- Versus V31: +11.59pp [+1.83, +21.95], 45 wins/26 losses.
- Versus Original V2: +56.10pp [+48.17, +64.02], 92 wins/0 losses.
- Versus V3a: -0.61pp [-10.38, +9.15], 34 wins/35 losses.

Validation: independent CPU tests and30 CUDA oracle/consumer cases;12 exact original-control replays; all164 new records source-bound and reconciled with private token hashes, four complete41-cell shards, same prompt identities/caps and official scorer fingerprint. Native dense and all-kept FA4 are separate historical references. Repeated HumanEval is exploratory; no held-out confirmation or universal repair claim. Raw prompts, gold and generations remain private.
