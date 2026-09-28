# v22 CP-A: bootstrap fill, frozen Q16 calibration, validation and attention-share ceiling

Sources: v22 stages at `372acfa3` (calibration, `--bootstrap-calibration --qualify-tests`) and `4cfbf7a9` (validation, `--q16-validation --qualify-tests`), fresh immutable deploys and freshly bound configs on mpk and dllm. Every stage wrote `completeness.status = complete` (no failed layer, no recorded error, five points per screen); the 15 geometry tests passed on each H100 inside the charged stage. Private receipts are pinned by SHA-256 in [q16_calibration.json](q16_calibration.json) and [q16_validation.json](q16_validation.json). Error is FP32 attention-output relative L2 against full attention on the same state. It is not a task-quality measure, and no answer or gold was read.

## 1. Missing bootstrap states: filled

All six call-0 states (AIME, LB, RULER × two hosts) now have qualified LOCAL0 and GLOBAL5 records under the production convention `T=None -> neutral ones`. Coarse route parity matched on all 12. The 12 b85fa23 failures remain preserved as diagnostic failures of the old capture, not production failures.

## 2. Frozen two-state calibration (GLOBAL layer 5, LB call 3, seed 101)

Pooled `E = sqrt(sum ||O-O_ref||^2 / sum ||O_ref||^2)` over mpk canvas0 (`...067c5456`, 13,703 keys) and dllm canvas1 (`...acb2a9de`, 17,540 keys). Coarse Q128/K64 at the inherited threshold: aggregate retained legal pairs 45,974,144, pooled E 0.2685.

| Q16/K64 offset | Aggregate work / coarse | Pooled E / coarse | Per-state E / coarse (mpk, dllm) | Row p99 rel-L2 (mpk, dllm) | Removed-mass row p99 (mpk, dllm) |
|---:|---:|---:|---|---|---|
| coarse | 1.000 | 1.000 | 1, 1 | 0.675, 1.043 | 0.641, 0.666 |
| 0 | 0.557 | 1.516 | 1.725, 1.363 | 1.226, 1.205 | 0.906, 0.852 |
| -0.25 | 0.676 | 1.383 | 1.589, 1.232 | 1.200, 1.140 | 0.899, 0.800 |
| -0.5 | 0.815 | 1.225 | 1.395, 1.102 | 1.088, 1.099 | 0.853, 0.759 |
| **-1.0** | **1.158** | **0.918** | 1.020, 0.845 | 0.929, 0.932 | 0.768, 0.666 |
| -2.0 | 1.990 | 0.440 | 0.486, 0.408 | 0.567, 0.571 | 0.544, 0.424 |

Retained work was monotone in the offset on both states. **Rule 1** (closest aggregate work) and **Rule 2** (least work with pooled E ≤ 1.05× and each state ≤ 1.15×) both select **offset -1.0**. It retains **15.8% more** legal pairs than coarse for 8.2% lower pooled error. No measured Q16 point uses less work than coarse inside the error band. The best-less-work point (-0.5) is 1.225× the coarse error. A descriptive (unmeasured) log-linear interpolation of the pooled frontier at equal work gives 1.046× coarse error.

## 3. Validation on all other reached GLOBAL states (no reselection)

The same five offsets were rerun on every previously captured GLOBAL state plus call 0 (22 states; 2 are the calibration states). The descriptive equal-work error ratio is a log-linear interpolation between measured offsets, not a measured point.

| Task | Validation states | Coarse kept (median) | Offset -1.0 work / coarse | Offset -1.0 error / coarse | States with less work AND error | Equal-work error / coarse, median [range] |
|---|---:|---:|---:|---:|---:|---|
| LongBench-v2 | 6 | 27.8% | 1.19 | 0.86 | 2 | 0.91 [0.83, 1.00] |
| RULER-4K | 7 | 34.0% | 1.18 | 0.80 | 0 | 0.93 [0.86, 1.30] |
| AIME26 | 8 | 97.5% | 0.99 | 1.04 | 3 | 0.68 [0.48, 1.79] |

Q16 is modestly better at equal work on most long-context states (≈9% lower attention error). Equivalently, at equal error it retains roughly 4–5% fewer GLOBAL legal pairs. The calibration pair happened to include the least favourable LB state (mpk call 3, 1.14×). Under the frozen threshold, however, fine geometry buys lower error with **more** work, not less. AIME GLOBAL is almost dense under both geometries.

## 4. Attention-share ceiling (native complete forward, same states)

In-place CUDA events around every native attention call inside `model.forward(...).logits` (1 warmup, 3 rotated repetitions, digest-stable). The `no_global_attention` oracle returns zeros for the five GLOBAL layers. It bounds what any GLOBAL support policy could save at zero selection and consumer cost, and its logits are invalid. LOCAL and all-zero oracles are reported in JSON but are confounded: zeroed attention changes MoE routing, and several such forwards got *slower*.

| Task | Keys | Native forward ms (median) | GLOBAL attention in forward | LOCAL attention in forward | Forward with GLOBAL attention deleted / native |
|---|---:|---:|---:|---:|---:|
| LongBench-v2 | 13.7–17.5K | 140–169 | **15.5%** (21–26 ms) | 1.2% | **0.84** [0.81, 0.88] |
| RULER-4K | 4.1–4.2K | 117–141 | 5.1% | 1.5% | 0.97 [0.95, 1.00] |
| AIME26 | 0.4–0.6K | 115–140 | 1.3% | 1.4% | 0.98 [0.96, 1.01] |

## 5. Decision

**Production fine-geometry promotion is stopped** under §6.3 of the v22 contract.

1. The frozen rule's candidate does not save work.
2. At matched error, the validated work saving is ≈4–5% of the retained GLOBAL pairs. That is ~30% of legal pairs on LB, whose fixed-support consumer costs ≈1.5–1.7 ms per layer, so the upper bound is well under 1 ms per forward (<0.5%) before any extra selector, bitmap or refresh cost.
3. The precise limiting factor is the GLOBAL attention share itself. Deleting all five GLOBAL attention calls saves at most ~16% of a native LB forward at 14–18K keys, ~3% on RULER-4K and ~2% on AIME. Coarse M3 R3 already realised 5–9% per forward on LB (v20 same-state replay), which is roughly half of that ceiling.
4. The 10–15% complete-forward aspiration therefore sits at or above the attention ceiling on the current panel. It cannot be reached through support geometry on these contexts.

CP-B (fine kernel), CP-C (fine full-forward) and CP-D (the 8-arm geometry panel) were not run. No sixth offset, R/A change, old-grouping substitution or new heuristic was introduced. Close prior art also covers finer support with periodic refresh ([novelty addendum](novelty_gap.md)).
