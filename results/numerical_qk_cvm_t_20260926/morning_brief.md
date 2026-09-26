# v14 morning brief — CVM-T under native adaptive denoising (AIME26, GLOBAL-only)

**Bottom line.** CVM-T is a measured negative on AIME.
- Its ordinary-step decision is genuinely cheaper than dense GLOBAL attention at long prefixes.
- That saving is ~0 at the level of a complete AIME request.
- Its live protection did not beat the matched frozen bitmap (B8_P) on quality or time in the 120-execution pilot.

Everything is committed; no GPU jobs are left. Budget used: ~2.4 h wall of 8, 1.77 GPU-h of 6, 123 complete executions of 224 (3 smoke + 120 pilot).

## Q1. Did decision + observation cost less than the removed attention work?
**Yes at the GLOBAL-call level beyond ~2K prefix; no at short prefixes.** Measurements are in-situ per-call CUDA events on real steps, 5 GLOBAL layers per step, and are reproducible across runs.

| prefix | native | CVM ordinary (planner + consumer) | B8_P ordinary | anchor (export) |
|---:|---:|---:|---:|---:|
| 365 | 1.80 ms | 2.11 | 1.69 | 3.3 |
| 2157 | 4.71 | 3.63 | 2.66 | 5.7 |
| 6275 | 10.75 | 7.17 | 4.97 | 11.2 |
| 15.7–17.9K (LongBench) | 24.5–27.6 | 12.1–14.0 | 7.0–7.5 | 22.9–25.7 |

- Launch inventory: ordinary CVM steps run exactly 5 planner + 5 consumer kernels, with no value-direction or dense GLOBAL launch. The anchors run v5.
- The CVM−B8_P gap is mostly restored QK/PV work, not the planner. CVM restores support to ≈ fresh-T density (executed prefix 0.70–0.75 vs B8_P 0.52–0.55).

## Q2. Did that survive the complete forward?
**Not on AIME; yes in the long-context regime.**
- GLOBAL attention is ≤8% of an AIME step.
- Phase-weighted CVM_T c = 1.020 / 1.009 / 0.989 at prefixes 365 / 2157 / 6275. The request estimate is 1.003.
- In the pilot, the order-randomized per-call factor vs native is CVM_T 1.003, B8_P 1.005, T_P 1.013, T_G 1.018.
- An order-controlled profile shows ~2% process-level drift, so AIME forward differences ≤2% are not resolvable by profiles. The binding plus the exact fast-T controller cost ~0 (an earlier "~1.5 ms/step T overhead" statement is retracted in `gate_decisions.md`).
- Real LongBench-v2 16–18K prompts (unscored regime diagnostic): CVM_T c = 0.929 / 0.942, B8_P 0.901 / 0.917.

## Q3. Did adaptive extra work erase it?
**Not in the identical-state canvas diagnostic; the pilot is dominated by trajectory length.**
- Canvas diagnostic, 5 real states run to the native stop: D 54 calls, T_G 49, T_P 43, B8_P 43, CVM_T 46. No caps, and ≤1 stable-but-not-confident step per arm.
- Pilot calls: D 3514, T_G 2874, T_P 3707, B8_P 3114, CVM_T 3955.
  - CVM_T/B8_P time 1.23 [1.08, 1.50] = calls ×1.27 × per-call 0.998.
  - CVM_T/D 1.15 [0.98, 1.49] = calls ×1.13 × per-call 1.003.
- These are trajectory effects: wrong answers are 8192-token caps (CVM_T 6 caps, B8_P 3). They are not execution overhead.

## Q4. Did correct complete answers survive?
Correct out of 12 (6 dev IDs × seeds 17/29, first outputs, frozen scorer):

| arm | correct | caps | EOS-but-wrong |
|---|---:|---:|---:|
| B8_P | 9 | 3 | 0 |
| T_G_original | 8 | 4 | 0 |
| D_native | 6 | 6 | 0 |
| CVM_T | 6 | 6 | 0 |
| T_P | 5 | 5 | 2 |

- CVM_T equals native (paired 1 vs 1) and trails B8_P (paired 1 vs 4; question-cluster Δ [−0.58, 0]).
- These are development IDs and a small pilot: intervals are descriptive, not noninferiority, and not proof of harm.
- Integrity checks:
  - D_native and T_G_original attempt-0 tokens equal the v13 cells 12/12 and 12/12, which validates the exact T fast path end to end.
  - 0 failures; 60/60 warm repeats reproduced their attempt 0.

## Q5. Single next action
**Stop CVM-T on AIME.**
- No method in this family can show a measured AIME E2E execution saving: every arm's per-call cost is within 1.00–1.02 of native, and time is set by trajectory length.
- Live protection added nothing measurable over B8_P, whose one-observation reuse is prior art (MAGE).

**The one next experiment**, if Fan/PI want to keep the runtime claim: run the same frozen five arms, unchanged, on a small scored LongBench-v2 10–20K panel through the v13/v14 seed-safe driver. This is the only measured regime with per-forward headroom (c 0.90–0.94). It needs PI approval because it makes a long-context benchmark primary.

## Not done / limits
- Extension /1,/6,/16,/25 was NOT run (gate 5.1 failed on AIME); the IDs were committed before the scores.
- No repair (neither A2 nor a fused planner had a supported diagnosis).
- There are no request-level executed-tile fractions (minimal telemetry). Tile fractions come from the canvas diagnostic only.
- The adaptive diagnostic is n = 5 states × 1 trajectory. Its divergence counts are descriptive.
- ~90% of divergent rows diverge while s = 1 (T bootstrap / no prior flip), so T's causal protection is structurally unavailable for them.

Files: `cost_budget.csv`, `complete_forward_profile.json`, `adaptive_canvas_diagnostic.json`, `launch_inventory.json`, `overhead_attribution.json`, `sanitizer_receipt.txt`, `qualification.json`, `gate_decisions.md`, `frozen_protocol*.json`, `complete_request_results.{csv,json}`, `private_receipt_index.json`, `fan_update.md`.


---
*Appended during v15:* scoped corrections to this report are in `corrections_v15.md` (LongBench phase-weighting used the AIME canvas mix; wording, divergence-cause, geometric-vs-pooled and drift clarifications). The text above is unchanged.
