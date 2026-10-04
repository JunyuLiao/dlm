# Paper outline (MLSys 2027, deadline 2026-10-30 20:00 UTC) — working draft, 2026-10-04

Working title: **Adapt in time, balance in space: progress-aware sparse attention for diffusion LM serving.**

The paper must stand on all three legs: performance, accuracy, novelty. Every design element answers a MEASURED
observation. Status: ✓ = evidence in hand; ◐ = partial; ○ = pending.

## Observations (characterization)

| | Observation | Evidence | Status |
|---|---|---|---|
| O1 | In DLM serving every denoising forward (256-row canvas, 12–20 forwards per canvas) re-reads the whole prefix in the global layers. Prefix attention is the length-dependent cost: 19–27% of a step at 128K. | Amdahl / step profiles; all-kept baseline (kernel path within 1–2% of dense) | ✓ |
| O2 | Prefix attention is sparse, but a canvas's queries drift as it denoises (masks → content). A selection made at step 1 goes stale. That hurts find-while-writing tasks: on MRCR, lean answers come out truncated, 3.4K vs 8.2K tokens. | pa: re-select +0.051 vs lean; S1: +0.091 [+0.031, +0.167]; GraphWalks +0.057 (n.s.) | ✓ |
| O3 | On H100 an FA4 attention kernel runs 1 CTA per SM in one wave, so a call takes as long as its longest CTA. Per-query adaptive budgets (threshold / coverage selectors, C-gate allocation) lose the speed they save: skew is 3.5–6× slower at the same mean. | nsys geometry (`fa4_grid_summary.json`); kernel audit; panels c / l | ✓ |
| O4 | Approximate attention can stall canvases near the convergence threshold, which inflates denoising steps. Unforced ratios are noisy across hosts; the forced-canvas measurement of lean is pending. | panels m / n; sc1x forced arms | ◐ |

## Design

| | Design | Answers | Status |
|---|---|---|---|
| D1 | Balanced selection: per (query head, 128-row block) worst-row prefix-mass top-k with equal k. The FA4 pass observes in-kernel (observing mask writes the tile log-mass during the dense pass). The selection is held for the canvas and carried into the next canvas's first call. | O1, O3 | ✓ (+1.40 vs MAGE on RULER, pre-registered; same per-step cost as MAGE) |
| D2 | Progress clock: the sampler's own acceptance or C-gate settledness decides when to re-observe. Two-level re-observation runs inside the block-sparse kernel over a candidate pool, at balanced CTAs and pool-sized cost. | O2 | ◐ (MRCR / GraphWalks gains; LongBench-think −6 n.s.; pool GPU check pending) |
| D3 | Budget adapts in time, not in space: one k per step for all units, set by progress or coverage. | O3 | ○ (sc2) |
| D4 | Step control from the C gate: stall rescue (attention side) and a stop rule (sampler side, evaluated on dense too and reported as a separate component). | O4 | ○ (sc1x / sc2) |

C gate and query sensitivity are Junyu Liao's ideas, a COLLABORATION CANDIDATE. Our parts are D1, the clock as a
trigger, the pool re-observation, and the balance principle.

## Evaluation plan

- **Accuracy (official scorers).**
  - Long: RULER (32–128K), LongBench-v2 `0shot_think`, MRCR or GraphWalks.
  - Short: AIME26, HumanEval.
  - The benchmark choice is fixed before the confirmation. All screened benchmarks are reported, the rest in an
    appendix.
- **Baselines.** Official dense FULL, all-kept (same kernel path), MAGE (same per-step cost), dense + the same step
  rule (for D4).
- **Performance.**
  - Per step, end to end, and the step count.
  - Context scaling, if feasible to 256K.
  - Batched throughput, if feasible.
- **Kernel analysis (nsys).**
- **Ablations:** D1 granularity / carry, D2 trigger / pool, D3 budget, D4 rescue / stop.
- **Confirmation.** Pre-registered on hold-out items (RULER v33, LongBench `0shot_think` hold-out 343, MRCR /
  GraphWalks bin complements, HumanEval hold-out 124), at least 2 seeds.

## Risks

- **The end-to-end gain is bounded by O1 (about 18–25% per step at 128K).** Feasibility study under way (branch
  `research/v31-scale-feasibility-20261004`):
  - 256K context, where the attention share grows;
  - batched throughput, where shared weight reads make attention a larger share.
- **D2 must not hurt long-thinking tasks.** Relative-progress trigger and pool variants are in sc2.
- **A second model.** LLaDA2.1-mini has no batch-1 headroom at ≤32K. A survey of other DLMs is under way (branch
  `research/dlm-model-headroom-20261004`).
