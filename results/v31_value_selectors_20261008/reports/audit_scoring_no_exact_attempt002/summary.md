# Official audit scoring — eight-arm LongBench panel

The audit stage is complete for all eight authorized arms. Clean timing was paused by the user after 419/4024 clean records; no clean-latency result is reported.

| Arm | Correct / 503 | Accuracy | Δ vs current control | 95% cluster interval | GLOBAL sparsity | N |
|---|---:|---:|---:|---|---:|---:|
| dense_full_fix51994 | 247 | 49.1% | -4.2 pp | [-8.0, -0.4] pp | 0.00% | 163345 |
| dense_piecewise | 247 | 49.1% | -4.2 pp | [-8.0, -0.4] pp | 0.00% | 163345 |
| allkept_fa4 | 252 | 50.1% | -3.2 pp | [-6.6, +0.2] pp | 0.00% | 147260 |
| current_v31_control | 268 | 53.3% | +0.0 pp | [+0.0, +0.0] pp | 83.16% | 140693 |
| value_v1_online_discard_mass | 247 | 49.1% | -4.2 pp | [-7.6, -0.8] pp | 80.04% | 120301 |
| value_v2_online_preserve_mass | 266 | 52.9% | -0.4 pp | [-3.8, +3.2] pp | 84.11% | 126072 |
| value_v3a_singleton_delete | 253 | 50.3% | -3.0 pp | [-6.4, +0.4] pp | 83.08% | 137212 |
| value_v3b_approx_batch8 | 242 | 48.1% | -5.2 pp | [-9.1, -1.4] pp | 84.97% | 171652 |

GLOBAL sparsity is audit-accounted physical eligible-tile skipping. LOCAL sparsity is zero by protocol. The intervals are paired question-cluster bootstrap intervals for descriptive comparison; nonsignificance is not a noninferiority claim.

The native dense control scored 247/503 and PIECEWISE also scored 247/503. The all-kept FA4 consumer scored 252/503 and differs in trajectories from native dense despite zero skipped tiles, so both controls remain visible.

The exact-greedy arm is excluded from this report by user instruction. Its completed 128-cell shard is retained under `diagnostics/exact_completed_shard01/attempt001/`.
