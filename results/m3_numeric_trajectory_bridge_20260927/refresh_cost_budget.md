# Measured refresh cost and a bounded opportunity estimate

Source: published [byte-preserving profile records](../fan_m1_m3_multidataset_20260927/profile_records/README.md) and their [SHA-256 manifest](../fan_m1_m3_multidataset_20260927/profile_records/manifest.json), specifically `selected_001_{mpk,dllm}.json.gz` and the shorter RULER `ruler_selected_001_{mpk,dllm}.json.gz`. These are actual-state, same-GPU replay measurements with three accepted blocks. `model_forward` times the complete decoder/logits forward with `state.begin` outside its boundary; direct epoch times the sequence including fixture state restoration. They are not natural request wall or a measured attention-kernel-only cost. The selected N16 replay reached only 7–14 calls on AIME; RULER reached four calls and has only the supplemental N4 profile. No nonexistent calls are padded.

The frozen R2 clock is A,H,D,H,D,H,D,H,A,H,D,H… when reached: A is current-score observation, H uses held support, and D redecides on prior summaries. A at call 0 includes initialization and is not an ordinary later score anchor. The `phase_deltas` counters show five GLOBAL-layer calls at each phase under the selected scope, with A score refresh, D decision refresh and H held consumption. They are used to *identify* real phases, not to sum phase median bins into a fictional request. [v20 method contract](../fan_m1_m3_multidataset_20260927/method_contract.json), lines 19–32; [per-forward report](../fan_m1_m3_multidataset_20260927/per_forward.md).

| Host / actual capture | Reached | R2 A/D/H complete calls | Real D call indices | Direct R2 / B / native sequence epoch (ms) | Optimistic removable D excess (ms) |
|---|---:|---:|---|---:|---:|
| mpk, AIME canvas 0 | 12 | 2/4/6 | 2,4,6,10 | 1682.7 / 1668.7 / 1636.5 | 9.5 |
| dllm, AIME canvas 0 | 8 | 1/3/4 | 2,4,6 | 1001.5 / 993.5 / 979.4 | 6.7 |
| mpk, AIME canvas 8 | 14 | 2/5/7 | 2,4,6,10,12 | 1894.4 / 1882.5 / 1840.8 | 12.2 |
| dllm, AIME canvas 8 | 7 | 1/3/3 | 2,4,6 | 836.1 / 829.5 / 820.1 | 5.9 |
| mpk, LongBench canvas 0 | 16 | 2/6/8 | 2,4,6,10,12,14 | 2493.3 / 2420.3 / 2598.1 | 63.0 |
| dllm, LongBench canvas 0 | 16 | 2/6/8 | 2,4,6,10,12,14 | 2315.5 / 2237.1 / 2547.0 | 64.9 |
| mpk, LongBench canvas 4 | 16 | 2/6/8 | 2,4,6,10,12,14 | 2439.4 / 2365.5 / 2552.4 | 61.9 |
| dllm, LongBench canvas 4 | 16 | 2/6/8 | 2,4,6,10,12,14 | 2308.4 / 2218.1 / 2496.5 | 67.6 |
| mpk / dllm, RULER canvas 0 | 4 | 1/1/2 | 2 | 563.6 / 556.8 / 554.2; 530.1 / 525.6 / 520.0 | 4.5 / 4.8 |

**Computation of the last column.** For each *actual* D index `i` in one captured sequence, take the directly timed complete-call event median `R2[i]` and the same-index B-A8 held-call median `B[i]`; sum `max(0, R2[i] − B[i])`. This is an optimistic observational ceiling on the existing indexed D-call excess if redecision could be made to cost like the B held call while **all other calls and the output trajectory stayed fixed**. It is neither a measured intervention nor a rigorous bound on a new implementation. The largest observed sum is 67.6 ms over 16 LongBench forwards (2.9% of that R2 direct epoch), on dllm canvas 4. The complete-call R2/B direct epoch difference there is 90.3 ms (3.9% of R2); the difference includes A/H/controller and fixture effects, so it cannot all be credited to removing D. AIME excess tops out at 12.2 ms over 14 calls, and the four-call RULER evidence offers only 4.8 ms. The A/D/H prices are whole decoder calls, not QK-only prices.

## Separate R3 successor cost screen

The frozen successor is **R3**, not R2. The same published replay gives these real R3 D indices and a separately computed optimistic positive excess against the **same-index B held call**. The per-index calculation is identical to the R2 formula above, substituting R3; no phase-bin medians are added. R3's A anchor at call 8 resets decision age, so the next D is at call 11 when reached.

| Host / actual capture | Reached | R3 A/D/H complete calls | Real R3 D indices | Positive indexed R3−B D excess (ms) |
|---|---:|---:|---|---:|
| mpk, AIME canvas 0 | 12 | 2/3/7 | 3,6,11 | 8.1 |
| dllm, AIME canvas 0 | 8 | 1/2/5 | 3,6 | 4.4 |
| mpk, AIME canvas 8 | 14 | 2/3/9 | 3,6,11 | 7.7 |
| dllm, AIME canvas 8 | 7 | 1/2/4 | 3,6 | 4.6 |
| mpk, LongBench canvas 0 | 16 | 2/4/10 | 3,6,11,14 | 40.7 |
| dllm, LongBench canvas 0 | 16 | 2/4/10 | 3,6,11,14 | 45.2 |
| mpk, LongBench canvas 4 | 16 | 2/4/10 | 3,6,11,14 | 44.1 |
| dllm, LongBench canvas 4 | 16 | 2/4/10 | 3,6,11,14 | 45.7 |
| mpk / dllm, RULER canvas 0 | 4 | 1/1/2 | 3 | 3.9 / 5.0 |

For example, on mpk LongBench canvas 0 the **measured sum of complete-call event medians** is 2487.6 ms for R3, 2439.8 ms for B, and 2615.9 ms for native. The B/R3 ratio of these observed sums is **0.9808**; the independently timed whole replay epochs are 2468.8/2420.3/2598.1 ms for R3/B/native. Those sums and epochs are direct replay prices on one captured state sequence; neither is a geometric mean of natural requests. R3/native sum ratios across the four LongBench captures range **0.906–0.951**, with different contexts and hosts. This is a same-state forward cost advantage, not a request-level result.

The published natural LongBench panel has **2514 R3 versus 2360 native** decoder calls (work ratio **1.0653**) and **2514 versus 2223 fresh-T** (work ratio **1.1309**). Holding all other work and trajectories artificially fixed, a per-call R3/native price ratio would need to be below **2360/2514 = 0.9387**, and R3/T below **2223/2514 = 0.8842**, merely to offset these observed call-count multipliers. These reciprocals are arithmetic break-even screens, not forecasts: contexts, canvas counts, prefill, sampler/commit, correctness, host drift and the paired geometric request ratios do not collapse into a single amortized per-forward price. The [natural panel report](../fan_m1_m3_multidataset_20260927/adaptive_panel_report.md) retains the request-level paired geometric ratios separately; the replay ratio 0.9808 must not be substituted for one.

At selected LongBench canvas 4, the real R2 D complete-call medians span **141.0–164.9 ms** on mpk and **139.1–155.5 ms** on dllm. H medians span **126.0–154.8 ms** and **123.9–144.9 ms** respectively. These change with context and call index. We deliberately do not add their phase medians or extrapolate a synthetic 16-call AIME/RULER sequence. A new policy could change support, logits, stopping, and future states, so even the indexed difference is a cost-screening number, not predicted request speedup or quality.

The principal old vLLM **G75/L30 S30 FIXED16** estimate is positive under its different KV32 grouped selector and controlled schedule: 16 LongBench questions × two seeds × two loads gave LB D/S30 whole 1.0656, generation 1.1783 and TPOB 1.1673, with dense 28/64 and S30 30/64 correct. A separate later four-question GLOBAL-only G75 overlay yielded TPOB 1.129 and generation 1.299; it is not the S30 composition. The old adaptive G75/L30 fairness screen was near neutral on eight LB development questions. The v20 R2/B/native replay prices above have a different model integration, Q128/KV64 support, selected GLOBAL-only scope, task panel and timing boundary. None of these measurements reverses or replicates an old estimate. [Contract matrix](old_new_contract_matrix.md); [old 16-question confirmation](../../../adaptive_trajectory_characterization_20260921/results/layer_step_budget_20260921/paper_decision.md); [v20 held evidence](../fan_m1_m3_multidataset_20260927/held_bitmap_evidence.md).

The only current actionable cost budget is small: spend at most the observed same-index D excess when considering elimination of redecision at fixed support; require a fresh *complete-forward and native-step* measurement before claiming a benefit. No R-interval sweep or natural speed/quality prediction follows from this replay.
